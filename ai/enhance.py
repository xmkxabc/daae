import os
import json
import sys
import dotenv
import argparse
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, DefaultDict
from collections import defaultdict

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


@dataclass
class ProcessingTask:
    """处理任务的数据结构"""
    idx: int
    data: Dict
    
@dataclass
class ProcessingResult:
    """处理结果的数据结构"""
    idx: int
    data: Dict
    success: bool
    error_message: Optional[str] = None

class APIKeyManager:
    """API密钥管理器，负责管理单个密钥的速率限制和调用"""
    
    def __init__(self, api_key: str, key_name: str, model_names: List[str], 
                 prompt_template, retries: int = 3, timeout: int = 1, 
                 api_call_interval: int = 10):  # 改为10秒间隔
        self.api_key = api_key
        self.key_name = key_name
        self.model_names = model_names
        self.prompt_template = prompt_template
        self.retries = retries
        self.timeout = timeout
        self.api_call_interval = api_call_interval
        self.last_call_time = 0
        self.lock = threading.Lock()
        
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
    
    def _wait_for_rate_limit(self):
        """等待速率限制"""
        with self.lock:
            current_time = time.time()
            time_since_last_call = current_time - self.last_call_time
            if time_since_last_call < self.api_call_interval:
                sleep_time = self.api_call_interval - time_since_last_call
                time.sleep(sleep_time)
            self.last_call_time = time.time()
    
    def is_model_available(self, model_name: str) -> bool:
        """检查模型是否可用（未耗尽配额）"""
        if self.model_quota_exhausted.get(model_name, False):
            return False
        # [新] 检查模型是否在冷却期
        if time.time() < self.model_cooldown_until.get(model_name, 0):
            return False
        return True
    
    def mark_model_exhausted(self, model_name: str):
        """标记模型配额已耗尽"""
        self.model_quota_exhausted[model_name] = True
        print(f"  ! 模型 {model_name} 在<{self.key_name}>上配额耗尽", file=sys.stderr)
    
    def set_model_cooldown(self, model_name: str, duration: int = 65):
        """
        当遇到临时速率限制时，为模型设置一个冷却期。
        默认为65秒，以安全地度过“每分钟请求数”的限制。
        """
        self.model_cooldown_until[model_name] = int(time.time() + duration)
        print(f"  ! 速率限制: <{self.key_name}> - {model_name} 进入冷却 {duration} 秒", file=sys.stderr)
    
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
            
        self._wait_for_rate_limit()
        
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

def worker_thread(task_queue: queue.Queue, result_queue: queue.Queue, 
                 scheduler: ModelScheduler, language: str, thread_id: int):
    """工作线程函数"""
    while True:
        try:
            task = task_queue.get(timeout=1)
            if task is None:  # 结束信号
                break
                
            # [新] 引入任务级重试，从之前的版本继承
            max_task_retries = int(os.environ.get("MAX_TASK_RETRIES", 5))
            if task.retry_count >= max_task_retries:
                error_message = f"任务超过最大重试次数 ({max_task_retries}次)，已放弃"
                print(f"线程{thread_id}放弃任务: {task.data['id']} - {error_message}", file=sys.stderr)
                task.data['AI'] = {field: error_message for field in Structure.model_fields.keys()}
                processing_result = ProcessingResult(task.idx, task.data, False, error_message)
                result_queue.put(processing_result)
                task_queue.task_done()
                continue

            print(f"线程{thread_id}正在处理: {task.idx + 1} - {task.data['id']} (尝试次数: {task.retry_count + 1})", file=sys.stderr)
            
            # 获取可用的管理器和模型
            manager, model_name = scheduler.get_available_manager_for_current_model()
            
            if not manager or not model_name:
                # [核心改进] 如果暂时没有可用模型（可能都在冷却），不要立即失败。
                # 将任务重新排队，并等待一小段时间，避免CPU空转。
                error_message = "暂时无可用模型，任务将重新排队"
                print(f"线程{thread_id}发现: {error_message}", file=sys.stderr)
                task.retry_count += 1
                task_queue.put(task)
                time.sleep(5) # 等待5秒，给密钥一些冷却恢复的时间
                task_queue.task_done()
                continue
            
            # 使用选定的管理器和模型处理论文
            result, error = manager.process_paper_with_model(task.data, language, model_name)
            
            if result:
                task.data['AI'] = result
                processing_result = ProcessingResult(task.idx, task.data, True)
                print(f"线程{thread_id}成功处理: {task.data['id']} 使用 {model_name}", file=sys.stderr)
            else:
                error_message = "错误：AI分析失败。"
                task.data['AI'] = {field: error_message for field in Structure.model_fields.keys()}
                processing_result = ProcessingResult(task.idx, task.data, False, error)
                print(f"线程{thread_id}处理失败: {task.data['id']} - {error}", file=sys.stderr)
                
                # [改进] 如果是配额耗尽或速率限制，将任务重新放回队列
                if "配额耗尽" in str(error) or "速率限制" in str(error) or "冷却中" in str(error):
                    task.retry_count += 1
                    task_queue.put(task)  # 重新排队
                    task_queue.task_done()
                    continue
                
            result_queue.put(processing_result)
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"线程{thread_id}发生异常: {e}", file=sys.stderr)
        finally:
            task_queue.task_done()

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

def main():
    """主函数，运行增强过程。"""
    args = parse_args()
    
    # --- [核心改造] 加载统一的密钥和模型优先级列表 ---
    google_api_keys_str = os.environ.get("GOOGLE_API_KEYS") # e.g., "key1,key2,key3"
    model_priority_list_str = os.environ.get("MODEL_PRIORITY_LIST")
    # [新] 从环境变量加载API调用间隔，默认为10秒以遵循Google Free Tier常见的 6 RPM (每分钟请求数) 限制
    api_call_interval = int(os.environ.get("API_CALL_INTERVAL", 10))

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
    print(f"API调用间隔: {api_call_interval}秒 (约每分钟 {60/api_call_interval:.1f} 次)", file=sys.stderr)
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
            api_call_interval=api_call_interval
        )
        api_managers.append(api_manager)

    # 创建模型调度器
    scheduler = ModelScheduler(api_managers, model_names)

    # 创建任务队列和结果队列
    task_queue = queue.Queue()
    result_queue = queue.Queue()

    # 将所有任务加入队列
    for idx, paper_data in enumerate(data):
        task_queue.put(ProcessingTask(idx, paper_data))

    # 启动工作线程
    threads = []
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

    # 等待所有任务完成
    task_queue.join()

    # 发送结束信号给所有线程
    for _ in threads:
        task_queue.put(None)

    # 等待所有线程结束
    for thread in threads:
        thread.join()

    # 收集结果
    results = []
    total_failures = 0
    
    while not result_queue.empty():
        try:
            result = result_queue.get_nowait()
            results.append(result)
            if not result.success:
                total_failures += 1
        except queue.Empty:
            break

    # 按原始顺序排序结果
    results.sort(key=lambda x: x.idx)
    enhanced_data = [result.data for result in results]

    # 输出结果
    output_filename = args.data.replace('.jsonl', f'_AI_enhanced_{language}.jsonl')
    with open(output_filename, "w", encoding="utf-8") as f:
        for d_item in enhanced_data:
            f.write(json.dumps(d_item, ensure_ascii=False) + "\n")

    print(f"\n处理完成。成功处理: {len(enhanced_data) - total_failures}/{len(enhanced_data)}。输出文件: {output_filename}")
    
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
    if len(enhanced_data) > 0:
        # 考虑分层使用模型的时间计算
        estimated_time_sequential = len(enhanced_data) * api_call_interval
        # 并发处理时间取决于实际可用的密钥数量
        effective_concurrency = len(api_keys)
        estimated_concurrent_time = (len(enhanced_data) * api_call_interval) / effective_concurrency
        print(f"顺序处理预计时间: {estimated_time_sequential:.1f}秒 ({estimated_time_sequential/60:.1f}分钟)", file=sys.stderr)
        print(f"并发处理预计时间: {estimated_concurrent_time:.1f}秒 ({estimated_concurrent_time/60:.1f}分钟)", file=sys.stderr)
        print(f"预计节省时间: {(estimated_time_sequential - estimated_concurrent_time)/60:.1f}分钟", file=sys.stderr)

if __name__ == "__main__":
    main()
