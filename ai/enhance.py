import os
import json
import sys
import dotenv
import argparse
import time
import threading
import signal
import queue
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, DefaultDict
from collections import defaultdict, deque

import langchain_core.exceptions
from langchain_google_genai import ChatGoogleGenerativeAI
# **新增**: 明确导入需要的异常类型
from google.api_core import exceptions as google_exceptions
from langchain.prompts import ChatPromptTemplate
from structure import Structure

# 加载环境变量
if os.path.exists('.env'):
    dotenv.load_dotenv()

# --- 文件加载 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(script_dir, "template.txt"), "r", encoding="utf-8") as f:
        template_content = f.read()
    with open(os.path.join(script_dir, "system.txt"), "r", encoding="utf-8") as f:
        system_prompt_template = f.read()
except FileNotFoundError as e:
    print(f"错误：找不到必需的模板文件: {e}。搜索路径: {script_dir}", file=sys.stderr)
    sys.exit(1)

# 全局关闭事件
shutdown_event = threading.Event()

class SlidingWindowRateLimiter:
    """
    一个线程安全的速率限制器，使用滑动窗口算法。
    它允许多个线程在不超过指定速率限制的情况下并发执行。
    """
    def __init__(self, rate_limit: int, time_period: int = 60):
        self.rate_limit = rate_limit  # 例如: 60 (次)
        self.time_period = time_period # 例如: 60 (秒)
        self.timestamps = deque()
        self.lock = threading.Lock()

    def acquire(self):
        """获取一个许可，如果达到速率限制则阻塞等待。"""
        with self.lock:
            now = time.time()
            # 移除时间窗口之外的旧时间戳
            while self.timestamps and self.timestamps[0] <= now - self.time_period:
                self.timestamps.popleft()

            if len(self.timestamps) >= self.rate_limit:
                wait_time = (self.timestamps[0] + self.time_period) - now
                if wait_time > 0:
                    time.sleep(wait_time)
            
            self.timestamps.append(time.time())

@dataclass
class ProcessingTask:
    """处理任务的数据结构"""
    idx: int
    data: Dict
    retry_count: int = 0
    
@dataclass
class ProcessingResult:
    """处理结果的数据结构"""
    idx: int
    data: Dict
    success: bool
    error_message: Optional[str] = None

class APIKeyManager:
    """API密钥管理器，负责管理单个密钥的速率限制和调用"""
    
    def __init__(self, api_key: str, key_name: str, model_names: List[str], prompt_template,
                 retries: int = 3, timeout: int = 1, api_call_interval: float = 1.1):
        self.api_key = api_key
        self.key_name = key_name
        self.model_names = model_names
        self.prompt_template = prompt_template
        self.retries = retries
        self.timeout = timeout
        self.lock = threading.Lock()

        # [核心优化] 使用滑动窗口速率限制器
        self.rate_limiter = SlidingWindowRateLimiter(rate_limit=int(60 / api_call_interval))
        # 跟踪每个模型的配额状态
        self.model_quota_exhausted = {model: False for model in model_names}
        # [新] 跟踪每个模型的冷却状态
        self.model_cooldown_until = {model: 0 for model in model_names}
        
        # 初始化模型链
        self.model_chains = {}
        self._init_model_chains()
    
    def _init_model_chains(self):
        """初始化所有模型链"""
        for model_name in self.model_names:
            try:
                llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=self.api_key)
                structured_llm = llm.with_structured_output(Structure)
                chain = self.prompt_template | structured_llm
                self.model_chains[model_name] = chain
                print(f"模型已为<{self.key_name}>成功设置: {model_name}", file=sys.stderr)
            except Exception as e:
                self.model_chains[model_name] = None
                print(f"警告：无法为<{self.key_name}>初始化模型 {model_name}。错误：{e}", file=sys.stderr)
    
    def is_model_available(self, model_name: str) -> bool:
        """检查模型是否可用（未耗尽配额）"""
        with self.lock:
            if self.model_quota_exhausted.get(model_name, False):
                return False
            # [新] 检查模型是否在冷却期
            if time.time() < self.model_cooldown_until.get(model_name, 0):
                return False
            return True
    
    def mark_model_exhausted(self, model_name: str):
        """标记模型配额已耗尽"""
        with self.lock:
            self.model_quota_exhausted[model_name] = True
        print(f"  ! 模型 {model_name} 在<{self.key_name}>上配额耗尽", file=sys.stderr)
    
    def set_model_cooldown(self, model_name: str, duration: int = 65):
        """
        当遇到临时速率限制时，为模型设置一个冷却期。
        默认为65秒，以安全地度过“每分钟请求数”的限制。
        """
        with self.lock:
            self.model_cooldown_until[model_name] = int(time.time() + duration)
        print(f"  ! 速率限制: <{self.key_name}> - {model_name} 进入冷却 {duration} 秒", file=sys.stderr)

    def enforce_rate_limit(self):
        """
        [核心优化] 使用滑动窗口算法来执行速率限制。
        """
        self.rate_limiter.acquire()
    
    def process_paper_with_model(self, paper_data: Dict, language: str, model_name: str) -> Tuple[Optional[Dict], Optional[str]]:
        """使用指定模型处理单篇论文"""
        if not self.is_model_available(model_name):
            # [改进] 提供更明确的不可用原因
            if time.time() < self.model_cooldown_until.get(model_name, 0):
                return None, f"模型 {model_name} 正在冷却中"
            return None, f"模型 {model_name} 配额已耗尽"
            
        chain = self.model_chains.get(model_name)
        if not chain:
            return None, f"模型 {model_name} 未初始化"
            
        # [优化] 在确认模型可用后，再执行速率限制，避免不必要的等待
        self.enforce_rate_limit()
            
        for attempt in range(self.retries):
            print(f"  线程<{self.key_name}>使用: {model_name} (尝试 {attempt + 1}/{self.retries})", file=sys.stderr)
            try:
                response_object = chain.invoke({
                    "title": paper_data['title'],
                    "content": paper_data['summary'],
                    "language": language
                })
                if response_object and is_response_valid(response_object):
                    return response_object.model_dump(), None
                    
            except (google_exceptions.ResourceExhausted, google_exceptions.NotFound) as e:
                # [核心改进] 区分处理不同类型的API错误
                if isinstance(e, google_exceptions.NotFound):
                    error_type = "模型未找到"
                    print(f"  ! {error_type}: <{self.key_name}> - {model_name}. 将永久禁用此模型。", file=sys.stderr)
                    self.mark_model_exhausted(model_name) # 永久禁用
                    return None, f"{error_type}: {model_name}"

                # 对于 ResourceExhausted，我们区分是临时速率限制还是每日配额
                error_str = str(e).lower()
                if "per day" in error_str or "daily" in error_str:
                    error_type = "每日配额耗尽"
                    print(f"  ! {error_type}: <{self.key_name}> - {model_name}", file=sys.stderr)
                    self.mark_model_exhausted(model_name) # 永久禁用
                else:
                    # 假设是临时速率限制（如每分钟请求数）
                    error_type = "速率限制"
                    self.set_model_cooldown(model_name) # 进入临时冷却
                
                return None, f"{error_type}: {model_name}"
                
            except Exception as e:
                print(f"  > 发生瞬时性错误: {e}", file=sys.stderr)
                if attempt < self.retries - 1:
                    time.sleep(self.timeout)
        
        return None, f"模型 {model_name} 尝试失败"

class ModelScheduler:
    """模型调度器，管理按优先级分层的任务分配"""
    
    def __init__(self, api_managers: List[APIKeyManager], model_names: List[str]):
        self.api_managers = api_managers
        self.model_names = model_names
        self.current_model_index = 0

        self.last_used_manager_indices: DefaultDict[str, int] = defaultdict(int)
        self.lock = threading.Lock()
        
        # 为每个模型优先级创建可用的密钥管理器列表
        self.model_key_managers = {
            model: [manager for manager in api_managers if model in manager.model_names]
            for model in model_names
        }
        
        print(f"--- 模型调度器初始化 ---", file=sys.stderr)
        for i, model in enumerate(model_names):
            available_keys = len(self.model_key_managers[model])
            print(f"  优先级 {i+1}: {model} - 可用密钥: {available_keys}个", file=sys.stderr)
        print("-------------------------", file=sys.stderr)

    def get_available_manager_for_current_model(self) -> Tuple[Optional[APIKeyManager], Optional[str]]:
        """获取当前优先级模型的可用密钥管理器"""
        with self.lock:
            if self.current_model_index >= len(self.model_names):
                return None, "所有模型优先级已耗尽"
            
            current_model = self.model_names[self.current_model_index]
            available_managers = [
                manager for manager in self.model_key_managers[current_model]
                if manager.is_model_available(current_model)
            ]
            
            if available_managers:
                # [改进] 使用轮询（Round-Robin）在可用密钥之间进行负载均衡
                num_available = len(available_managers)
                current_manager_idx = self.last_used_manager_indices[current_model]
                manager = available_managers[current_manager_idx % num_available]
                self.last_used_manager_indices[current_model] = (current_manager_idx + 1)
                return manager, current_model
            else:
                # 当前模型在所有密钥上都耗尽，切换到下一个模型
                print(f"  ! 模型 {current_model} 在所有密钥上配额耗尽，切换到下一优先级", file=sys.stderr)
                self.current_model_index += 1
                return self.get_available_manager_for_current_model()
    
    def get_current_model_info(self) -> Dict:
        """获取当前模型使用情况"""
        with self.lock:
            info = {
                "current_model_index": self.current_model_index,
                "model_availability": {}
            }
            
            for model in self.model_names:
                available_count = sum(
                    1 for manager in self.model_key_managers[model]
                    if manager.is_model_available(model)
                )
                total_count = len(self.model_key_managers[model])
                info["model_availability"][model] = {
                    "available": available_count,
                    "total": total_count,
                    "exhausted": total_count - available_count
                }
            
            return info

def result_writer_thread(result_queue: queue.Queue, output_filename: str, total_tasks: int,
                         initial_next_idx: int, initial_processed_count: int):
    """
    一个专门的线程，用于从结果队列中获取数据并实时写入文件。
    它会缓冲乱序的结果，以确保文件内容是按原始顺序写入的。
    """
    print(f"--- 写入线程已启动，将结果保存到 {output_filename} ---", file=sys.stderr)
    
    results_buffer = {}
    next_expected_idx = initial_next_idx
    processed_count = initial_processed_count
    failure_count = 0 # 失败计数在每次运行时重置

    # 当 (收到关闭信号 且 结果队列已空) 或 (所有任务都已处理完) 时，循环结束
    while not (shutdown_event.is_set() and result_queue.empty()) and processed_count < total_tasks:
        try:
            result = result_queue.get(timeout=1)
            
            processed_count += 1
            if not result.success:
                failure_count += 1

            # 如果收到关闭信号，只处理队列中剩余的结果，不再等待新结果
            if shutdown_event.is_set():
                print(f"写入线程(关闭模式): 正在清空结果队列... 剩余 {result_queue.qsize()} 项", file=sys.stderr)
            
            # 将结果存入缓冲区
            results_buffer[result.idx] = result.data

            # 检查是否可以写入连续的结果
            with open(output_filename, "a", encoding="utf-8") as f:
                while next_expected_idx in results_buffer:
                    data_to_write = results_buffer.pop(next_expected_idx)
                    f.write(json.dumps(data_to_write, ensure_ascii=False) + "\n")
                    next_expected_idx += 1
            
            result_queue.task_done()
        except queue.Empty:
            # 在正常运行时，队列为空就继续等待
            # 如果收到了关闭信号，并且队列已空，上面的 while 条件会使其退出
            continue
    
    print(f"\n--- 写入线程完成。共处理 {processed_count} 个任务，其中失败 {failure_count} 个。---", file=sys.stderr)

def _handle_task_failure(task: ProcessingTask, result_queue: queue.Queue, error_message: str, log_prefix: str):
    """统一处理任务的最终失败，记录错误并放入结果队列。"""
    print(f"{log_prefix}: {error_message}", file=sys.stderr)
    task.data['AI'] = {field: error_message for field in Structure.model_fields.keys()}
    result_queue.put(ProcessingResult(task.idx, task.data, False, error_message))

def worker_thread(task_queue: queue.Queue, result_queue: queue.Queue, 
                 scheduler: ModelScheduler, language: str, thread_id: int):
    """工作线程函数，会检查全局关闭事件"""
    while not shutdown_event.is_set():
        try:
            task = task_queue.get(timeout=1)
        except queue.Empty:
            continue

        if task is None:
            break  # 结束信号

        try:
            log_prefix = f"线程{thread_id} (任务 {task.data['id']})"

            # 1. 检查是否超过最大重试次数
            max_task_retries = int(os.environ.get("MAX_TASK_RETRIES", 5))
            if task.retry_count >= max_task_retries:
                error_message = f"任务超过最大重试次数 ({max_task_retries}次)，已放弃"
                _handle_task_failure(task, result_queue, error_message, log_prefix)
                continue

            print(f"线程{thread_id}正在处理: {task.idx + 1} - {task.data['id']} (尝试次数: {task.retry_count + 1})", file=sys.stderr)

            # 2. 从调度器获取可用模型
            manager, model_name = scheduler.get_available_manager_for_current_model()
            if not manager or not model_name:
                print(f"{log_prefix}: 暂时无可用模型，任务将重新排队", file=sys.stderr)
                task.retry_count += 1
                task_queue.put(task)
                time.sleep(5)
                continue

            # 3. 处理任务
            result, error = manager.process_paper_with_model(task.data, language, model_name)

            # 4. 根据结果进行处理
            if result:
                task.data['AI'] = result
                result_queue.put(ProcessingResult(task.idx, task.data, True))
                print(f"{log_prefix}: 成功处理，使用 {model_name}", file=sys.stderr)
            elif "配额耗尽" in str(error) or "速率限制" in str(error) or "冷却中" in str(error):
                print(f"{log_prefix}: 处理失败，重新排队 - {error}", file=sys.stderr)
                task.retry_count += 1
                task_queue.put(task)
            else:
                error_message = f"AI分析失败 ({error})"
                _handle_task_failure(task, result_queue, error_message, log_prefix)

        except Exception as e:
            print(f"线程{thread_id}在处理任务 {task.idx if task else 'N/A'} 时发生意外异常: {e}", file=sys.stderr)
            if task:
                task.retry_count += 1
                task_queue.put(task) # 发生未知异常时也重新入队
        finally:
            task_queue.task_done() # 这是唯一调用task_done的地方

def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="使用AI摘要增强arXiv数据。")
    parser.add_argument("--data", type=str, required=True, help="要处理的JSONL数据文件。")
    parser.add_argument("--retries", type=int, default=3, help="对每个模型任务的最大重试次数。")
    parser.add_argument("--timeout", type=int, default=1, help="失败尝试之间的等待秒数。")
    parser.add_argument("--max-workers", type=int, default=16, help="最大并发工作线程数。")
    return parser.parse_args()

def is_response_valid(result: Structure):
    """验证响应，确保所有字段都为非空字符串。"""
    if not result:
        return False
    result_dict = result.model_dump()
    all_fields = Structure.model_fields.keys()
    for field in all_fields:
        value = result_dict.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            return False
    return True

def signal_handler(sig, frame):
    """处理 Ctrl+C 信号"""
    if not shutdown_event.is_set():
        print("\nCtrl+C detected! Initiating graceful shutdown...", file=sys.stderr)
        shutdown_event.set()

def setup_resume_logic(output_filename: str, id_to_idx_map: Dict[str, int]) -> Tuple[set, int, int]:
    """
    处理断点续传逻辑。
    检查输出文件，加载已处理的任务，并确定下一次写入的起始点。
    返回: (已处理ID集合, 下一个写入索引, 已处理任务计数)
    """
    processed_ids = set()
    initial_next_idx = 0
    initial_processed_count = 0
    if os.path.exists(output_filename):
        print(f"发现已存在的输出文件: {output_filename}。进入续传模式。", file=sys.stderr)
        temp_results = {}
        try:
            with open(output_filename, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        processed_data = json.loads(line)
                        paper_id = processed_data.get('id')
                        if paper_id and paper_id in id_to_idx_map:
                            original_idx = id_to_idx_map[paper_id]
                            temp_results[original_idx] = processed_data
                            processed_ids.add(paper_id)
                    except (json.JSONDecodeError, KeyError):
                        print(f"警告: 在 {output_filename} 中发现无效的JSON行，已跳过。", file=sys.stderr)
            
            while initial_next_idx in temp_results:
                initial_next_idx += 1
            
            if initial_next_idx < len(temp_results):
                print(f"警告: 输出文件 {output_filename} 存在不连续的条目。将从最后一个连续条目 {initial_next_idx - 1} 处截断并继续。", file=sys.stderr)
                contiguous_lines = [json.dumps(temp_results[i], ensure_ascii=False) + "\n" for i in range(initial_next_idx)]
                with open(output_filename, 'w', encoding='utf-8') as f:
                    f.writelines(contiguous_lines)
                processed_ids = {temp_results[i]['id'] for i in range(initial_next_idx)}
        except Exception as e:
            print(f"错误: 读取已存在的输出文件时发生错误: {e}。将作为新任务运行。", file=sys.stderr)
    return processed_ids, initial_next_idx, len(processed_ids)

def main():
    """主函数，运行增强过程。"""
    args = parse_args()
    
    # --- [核心改造] 加载统一的密钥和模型优先级列表 ---
    google_api_keys_str = os.environ.get("GOOGLE_API_KEYS") # e.g., "key1,key2,key3"
    model_priority_list_str = os.environ.get("MODEL_PRIORITY_LIST")
    # [新] 从环境变量加载API调用间隔，默认为1.1秒以遵循Google Free Tier常见的 60 RPM (每分钟请求数) 限制
    api_call_interval = float(os.environ.get("API_CALL_INTERVAL", 1.1))

    if not google_api_keys_str or not model_priority_list_str:
        print("错误: 请在 .env 文件中设置 GOOGLE_API_KEYS 和 MODEL_PRIORITY_LIST 环境变量。", file=sys.stderr)
        sys.exit(1)

    api_keys = [key.strip() for key in google_api_keys_str.split(',') if key.strip()]
    model_names = [name.strip() for name in model_priority_list_str.split(',') if name.strip()]

    if not api_keys or not model_names:
        print("错误: GOOGLE_API_KEYS 或 MODEL_PRIORITY_LIST 环境变量不能为空。", file=sys.stderr)
        sys.exit(1)

    # 可以使用比密钥数量更多的工作线程，因为现在是基于任务的调度
    max_workers = min(args.max_workers, len(api_keys) * 2)  # 最多是密钥数量的2倍
    
    print(f"--- 分层并发处理配置 ---", file=sys.stderr)
    print(f"API密钥数量: {len(api_keys)}", file=sys.stderr)
    print(f"模型优先级: {model_names}", file=sys.stderr)
    print(f"工作线程数: {max_workers}", file=sys.stderr)
    print(f"速率限制 (每个密钥): {60/api_call_interval:.1f} RPM (每分钟请求数)", file=sys.stderr)
    print("-----------------------------", file=sys.stderr)

    language = os.environ.get("LANGUAGE", 'Chinese')

    # 读取和预处理数据
    try:
        with open(args.data, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        print(f"错误: 处理文件 {args.data} 时出错: {e}", file=sys.stderr)
        return
    
    seen_ids = set()
    unique_data = [item for item in data if item.get('id') not in seen_ids and not seen_ids.add(item['id'])]
    data = unique_data
    print(f"从 {args.data} 加载了 {len(data)} 篇不重复的论文", file=sys.stderr)

    # 创建提示模板
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt_template),
        ("human", template_content)
    ])

    # 创建API管理器
    api_managers = []
    for i, api_key in enumerate(api_keys):
        key_name = f"密钥_{i + 1}"
        api_manager = APIKeyManager(
            api_key=api_key,
            key_name=key_name,
            model_names=model_names,
            prompt_template=prompt_template,
            retries=args.retries,
            timeout=args.timeout,
            api_call_interval=api_call_interval)
        api_managers.append(api_manager)

    # 创建模型调度器
    scheduler = ModelScheduler(api_managers, model_names)

    # 创建任务队列和结果队列
    task_queue = queue.Queue()
    result_queue = queue.Queue()
    
    # [核心改造] 准备输出文件并启动写入线程
    output_filename = args.data.replace('.jsonl', f'_AI_enhanced_{language}.jsonl')
    total_tasks = len(data)
    id_to_idx_map = {item['id']: i for i, item in enumerate(data)}
    
    # [新] 调用辅助函数处理断点续传
    processed_ids, initial_next_idx, initial_processed_count = setup_resume_logic(output_filename, id_to_idx_map)

    if initial_processed_count > 0:
        print(f"已加载 {initial_processed_count} 个已处理任务。将从索引 {initial_next_idx} 开始写入。剩余任务: {total_tasks - initial_processed_count}", file=sys.stderr)
    else:
        print(f"未发现输出文件。将开始全新处理。", file=sys.stderr)
        # 确保文件是空的，为新运行做准备
        with open(output_filename, "w", encoding="utf-8") as f:
            pass

    # 将未处理的任务加入任务队列
    for idx, paper_data in enumerate(data):
        if paper_data.get('id') not in processed_ids:
            task_queue.put(ProcessingTask(idx, paper_data))

    # 启动工作线程
    threads = []
    
    # [核心改造] 启动写入线程
    writer_thread = threading.Thread(
        target=result_writer_thread,
        args=(result_queue, output_filename, total_tasks, initial_next_idx, initial_processed_count)
    )
    writer_thread.start()
    
    for i in range(max_workers):
        thread = threading.Thread(
            target=worker_thread,
            args=(task_queue, result_queue, scheduler, language, i+1)
        )
        thread.start()
        threads.append(thread)

    # 定期显示调度状态
    def status_monitor():
        while not task_queue.empty():
            time.sleep(30)  # 每30秒显示一次状态
            info = scheduler.get_current_model_info()
            if info["current_model_index"] < len(model_names):
                current_model = model_names[info["current_model_index"]]
                print(f"--- 当前使用模型: {current_model} ---", file=sys.stderr)
                for model, avail_info in info["model_availability"].items():
                    print(f"  {model}: 可用 {avail_info['available']}/{avail_info['total']}, "
                          f"耗尽 {avail_info['exhausted']}", file=sys.stderr)
                print("剩余任务:", task_queue.qsize(), file=sys.stderr)

    # 启动状态监控线程
    monitor_thread = threading.Thread(target=status_monitor)
    monitor_thread.daemon = True
    monitor_thread.start()

    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)

    # 记录处理开始时间
    start_time = time.time()

    # 等待所有任务完成，或直到收到关闭信号
    try:
        while not task_queue.empty() and not shutdown_event.is_set():
            time.sleep(1) # 允许主线程响应信号

        if not shutdown_event.is_set():
            print("所有任务已进入队列，等待处理完成...", file=sys.stderr)
            task_queue.join() # 正常完成
        else:
            print("关闭信号已接收，等待工作线程完成当前任务...", file=sys.stderr)

    except (KeyboardInterrupt, SystemExit):
        shutdown_event.set()

    # 发送结束信号给所有线程
    for _ in threads:
        task_queue.put(None)
    for thread in threads:
        thread.join()
    result_queue.join() # 等待结果队列被完全处理
    writer_thread.join() # 等待写入线程完成

    # 记录处理结束时间
    end_time = time.time()
    
    # 显示最终统计
    final_info = scheduler.get_current_model_info()
    print(f"\n--- 最终模型使用统计 ---", file=sys.stderr)
    for model, avail_info in final_info["model_availability"].items():
        print(f"  {model}: 耗尽 {avail_info['exhausted']}/{avail_info['total']} 个密钥", file=sys.stderr)
    
    if final_info["current_model_index"] < len(model_names):
        print(f"  最终使用模型: {model_names[final_info['current_model_index']]}", file=sys.stderr)
    else:
        print(f"  所有模型优先级已使用完毕", file=sys.stderr)
    
    # 显示性能统计
    actual_elapsed_time = end_time - start_time
    print(f"\n--- 性能与耗时统计 ---", file=sys.stderr)
    print(f"实际总耗时: {actual_elapsed_time:.1f}秒 ({actual_elapsed_time/60:.1f}分钟)", file=sys.stderr)

    if total_tasks > 0:
        estimated_time_sequential = total_tasks * api_call_interval
        effective_concurrency = len(api_keys)
        estimated_concurrent_time = (total_tasks * api_call_interval) / effective_concurrency
        
        print(f"理论顺序处理时间 (估算): {estimated_time_sequential:.1f}秒 ({estimated_time_sequential/60:.1f}分钟)", file=sys.stderr)
        print(f"理论并发处理时间 (估算): {estimated_concurrent_time:.1f}秒 ({estimated_concurrent_time/60:.1f}分钟)", file=sys.stderr)
        if actual_elapsed_time > 0 and total_tasks > initial_processed_count:
            print(f"实际处理速度: {total_tasks / actual_elapsed_time:.2f} 篇/秒", file=sys.stderr)

if __name__ == "__main__":
    main()
