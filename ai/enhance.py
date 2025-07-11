import os
import json
import sys
import dotenv
import argparse
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, Any
from collections import defaultdict
import threading
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime

import langchain_core.exceptions
from langchain_google_genai import ChatGoogleGenerativeAI
from google.api_core import exceptions as google_exceptions
from langchain.prompts import ChatPromptTemplate
from structure import Structure  # 确保 structure.py 存在且定义了 Structure Pydantic 模型

# 配置日志
logging.basicConfig(
    level=logging.INFO, # 默认INFO级别
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)  # 只输出到控制台
    ]
)
logger = logging.getLogger(__name__)

# 加载环境变量
if os.path.exists('.env'):
    dotenv.load_dotenv()

# 常量定义
AI_CALL_FAILED_MARKER = "AI_CALL_FAILED"
MAX_TOTAL_ATTEMPTS_PER_PAPER = 50  # 每篇论文的最大总尝试次数
MAX_PROCESSING_TIME_PER_PAPER = 1800  # 每篇论文的最大处理时间（30分钟）
# 针对非永久性失败的指数退避冷却时间（例如，TooManyRequests）
TEMPORARY_FAILURE_BASE_COOLDOWN = 5 # 临时失败基础冷却时间（秒）
TEMPORARY_FAILURE_MAX_COOLDOWN = 300 # 临时失败最大冷却时间（秒，5分钟）


@dataclass
class TaskConfig:
    """任务配置类"""
    key_name: str
    api_key: str
    model_name: str
    priority: int
    available: bool = True # 用于标记初始化失败或永久性API问题
    last_failure_time: Optional[float] = None # 最近一次非永久性失败的时间
    failure_count: int = 0 # 非永久性失败的连续次数


@dataclass
class ProcessingStats:
    """处理统计信息"""
    start_time: float = field(default_factory=time.time)
    processed_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    
    def update_stats(self, success: bool = True):
        """更新统计信息"""
        with self.lock:
            self.processed_count += 1
            if success:
                self.success_count += 1
            else:
                self.failure_count += 1
    
    def get_stats(self) -> Dict:
        """获取当前统计信息"""
        with self.lock:
            elapsed_time = time.time() - self.start_time
            return {
                'processed': self.processed_count,
                'success': self.success_count,
                'failure': self.failure_count,
                'success_rate': self.success_count / max(1, self.processed_count) * 100,
                'elapsed_time': elapsed_time,
                'papers_per_minute': self.processed_count / max(1, elapsed_time / 60)
            }


class ConfigValidator:
    """配置验证器"""
    
    @staticmethod
    def validate_environment():
        """验证环境变量"""
        google_api_keys_str = os.environ.get("GOOGLE_API_KEYS")
        model_priority_list_str = os.environ.get("MODEL_PRIORITY_LIST")
        
        if not google_api_keys_str:
            raise ValueError("GOOGLE_API_KEYS 环境变量未设置")
        if not model_priority_list_str:
            raise ValueError("MODEL_PRIORITY_LIST 环境变量未设置")
        
        api_keys = [key.strip() for key in google_api_keys_str.split(',') if key.strip()]
        model_names = [name.strip() for name in model_priority_list_str.split(',') if name.strip()]
        
        if not api_keys:
            raise ValueError("至少需要一个有效的API密钥")
        if not model_names:
            raise ValueError("至少需要一个有效的模型名称")
        
        return api_keys, model_names
    
    @staticmethod
    def validate_files(script_dir: str) -> Tuple[str, str]:
        """验证必需的文件"""
        template_file = Path(script_dir) / "template.txt"
        system_file = Path(script_dir) / "system.txt"
        
        if not template_file.exists():
            raise FileNotFoundError(f"模板文件不存在: {template_file}")
        if not system_file.exists():
            raise FileNotFoundError(f"系统提示文件不存在: {system_file}")
        
        try:
            template_content = template_file.read_text(encoding="utf-8")
            system_content = system_file.read_text(encoding="utf-8")
            return template_content, system_content
        except Exception as e:
            raise IOError(f"读取模板文件时出错: {e}")


class RateLimiter:
    """智能频率限制器"""
    
    def __init__(self, min_interval: float = 6.0):
        self.min_interval = min_interval
        self.last_call_times = defaultdict(float)
        self.lock = threading.Lock()
    
    def wait_if_needed(self, key: str):
        """根据密钥智能等待"""
        with self.lock:
            current_time = time.time()
            elapsed = current_time - self.last_call_times[key]
            if elapsed < self.min_interval:
                wait_time = self.min_interval - elapsed
                logger.debug(f"API密钥 {key} 需要等待 {wait_time:.2f} 秒")
                time.sleep(wait_time)
            self.last_call_times[key] = time.time()


class TaskManager:
    """任务管理器"""
    
    def __init__(self, task_configs: List[TaskConfig]):
        self.task_configs = task_configs
        self.lock = threading.Lock()
        logger.info(f"任务管理器初始化完成，共 {len(task_configs)} 个任务配置")
    
    def get_next_available_task(self) -> Optional[TaskConfig]:
        """获取下一个可用任务"""
        with self.lock:
            # 过滤掉初始化失败或永久性不可用，以及仍在冷却期的任务
            available_tasks = [
                task for task in self.task_configs 
                if task.available and self._is_task_ready(task)
            ]
            
            if not available_tasks:
                return None
            
            # 按优先级和失败次数排序
            available_tasks.sort(key=lambda x: (x.priority, x.failure_count))
            return available_tasks[0]
    
    def _is_task_ready(self, task: TaskConfig) -> bool:
        """检查任务是否准备好使用（针对非永久性失败）"""
        if task.last_failure_time is None:
            return True
        
        # 指数退避：基础等待时间 * 2^失败次数，有最大值限制
        cooldown_time = min(TEMPORARY_FAILURE_MAX_COOLDOWN, TEMPORARY_FAILURE_BASE_COOLDOWN * (2 ** task.failure_count))
        return time.time() - task.last_failure_time > cooldown_time
    
    def mark_task_failed(self, task: TaskConfig, permanent: bool = False):
        """标记任务失败"""
        with self.lock:
            if permanent:
                task.available = False # 永久性失败，将此任务标记为不可用
                logger.warning(f"任务 {task.key_name}-{task.model_name} 被标记为永久不可用。")
            else:
                task.failure_count += 1
                task.last_failure_time = time.time()
                logger.debug(f"任务 {task.key_name}-{task.model_name} 失败次数: {task.failure_count}，进入冷却。")
    
    def mark_task_success(self, task: TaskConfig):
        """标记任务成功"""
        with self.lock:
            if task.failure_count > 0: # 仅当之前有失败时才重置
                task.failure_count = 0
                task.last_failure_time = None
                logger.debug(f"任务 {task.key_name}-{task.model_name} 成功，重置失败计数。")
    
    def get_available_tasks_count(self) -> int:
        """获取当前可被调度的任务（非永久失败且不在冷却中）数量"""
        with self.lock:
            return len([t for t in self.task_configs if t.available and self._is_task_ready(t)])
    
    def get_task_status(self) -> Dict:
        """获取任务状态统计"""
        with self.lock:
            total = len(self.task_configs)
            permanently_unavailable = len([t for t in self.task_configs if not t.available])
            
            temporarily_unavailable_cooldown = 0
            for task in self.task_configs:
                if task.available and task.last_failure_time is not None and not self._is_task_ready(task):
                    temporarily_unavailable_cooldown += 1

            available_now = total - permanently_unavailable - temporarily_unavailable_cooldown
            
            return {
                'total_tasks': total,
                'permanently_unavailable': permanently_unavailable,
                'temporarily_unavailable_cooldown': temporarily_unavailable_cooldown,
                'currently_available_for_scheduling': available_now
            }


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="使用AI摘要增强arXiv数据",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data", type=str, required=True, help="要处理的JSONL数据文件")
    parser.add_argument("--retries", type=int, default=3, help="对每个模型任务的最大重试次数")
    parser.add_argument("--timeout", type=int, default=2, help="失败尝试之间的等待秒数")
    cpu_count = os.cpu_count() or 4
    parser.add_argument("--max-workers", type=int, default=min(8, cpu_count * 2),
                        help="最大并发工作线程数")
    parser.add_argument("--batch-size", type=int, default=10, help="批处理大小")
    parser.add_argument("--output", type=str, help="输出文件名（已禁用，程序不保存文件）。")
    parser.add_argument("--resume", action="store_true", help="恢复处理（已禁用，因为不保存文件）。")
    parser.add_argument("--log-level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                        default='INFO', help="日志级别")
    return parser.parse_args()


def is_response_valid(result: Structure) -> bool:
    """验证响应，确保所有字段都为非空字符串。"""
    # Langchain的with_structured_output(Structure, response_model_strict=True) 
    # 已经处理了大部分结构和类型验证，这里主要检查非空字符串。
    if not result:
        logger.debug("AI响应对象为空。")
        return False
    
    try:
        result_dict = result.model_dump()
        all_fields = Structure.model_fields.keys()
        
        for field_name in all_fields:
            value = result_dict.get(field_name)
            if value is None or (isinstance(value, str) and not value.strip()):
                logger.debug(f"响应字段 '{field_name}' 为空或无效字符串。")
                return False
        return True
    except Exception as e:
        logger.error(f"验证AI响应数据结构时发生内部错误: {e}")
        return False


def process_single_paper(
    paper_data: dict,
    model_chains: Dict[Tuple[str, str], Any],
    task_manager: TaskManager,
    rate_limiter: RateLimiter,
    language: str,
    max_retries_per_task: int, # 对每个 TaskConfig 的重试次数
    timeout: int,
    stats: ProcessingStats
) -> dict:
    """处理单篇论文"""
    paper_id = paper_data.get('id', 'unknown')
    start_time = time.time()
    total_attempts_for_paper = 0
    
    logger.info(f"开始处理论文: {paper_id}")
    
    try:
        # 持续尝试，直到成功、达到总尝试上限或超过总处理时间
        while total_attempts_for_paper < MAX_TOTAL_ATTEMPTS_PER_PAPER:
            # 检查总处理时间
            if time.time() - start_time > MAX_PROCESSING_TIME_PER_PAPER:
                logger.warning(f"论文 {paper_id} 处理超时 ({MAX_PROCESSING_TIME_PER_PAPER}秒)，已用时 {time.time() - start_time:.2f} 秒。")
                break # 跳出循环，标记为失败
            
            # 获取下一个可用任务
            task = task_manager.get_next_available_task()
            if not task:
                # 如果所有任务都暂时不可用或永久不可用
                if task_manager.get_available_tasks_count() == 0:
                    logger.error(f"论文 {paper_id}: 所有任务都已永久不可用，无法继续处理。")
                    break # 跳出循环，标记为失败
                else:
                    logger.debug(f"论文 {paper_id}: 没有可用任务，等待冷却中的任务恢复。")
                    time.sleep(10)  # 等待一段时间，让一些任务从冷却中恢复
                    continue # 继续尝试获取任务
            
            # 尝试使用当前任务进行处理
            final_result = None
            for attempt_within_task in range(max_retries_per_task):
                try:
                    # 频率限制
                    rate_limiter.wait_if_needed(task.api_key)
                    
                    logger.debug(f"论文 {paper_id}: 使用 {task.key_name}-{task.model_name} (任务内尝试 {attempt_within_task + 1}/{max_retries_per_task})")
                    
                    # 获取模型链
                    key_tuple = (task.api_key, task.model_name)
                    chain = model_chains.get(key_tuple)
                    if not chain: # 如果模型链初始化时就失败了
                        task_manager.mark_task_failed(task, permanent=True)
                        logger.error(f"论文 {paper_id}: 模型链 '{task.key_name}-{task.model_name}' 未成功初始化，标记为永久失败。")
                        break # 跳出当前任务的重试循环，尝试下一个 TaskConfig
                    
                    # 调用AI模型
                    response_object = chain.invoke({
                        "title": paper_data['title'],
                        "content": paper_data['summary'],
                        "language": language
                    })
                    
                    if response_object and is_response_valid(response_object):
                        final_result = response_object.model_dump()
                        break # 成功则跳出当前任务的重试循环
                    else:
                        logger.warning(f"论文 {paper_id}: AI响应无效 (尝试 {attempt_within_task + 1}/{max_retries_per_task})。")
                        # 响应无效也视为瞬时失败，进行短暂等待
                        if attempt_within_task < max_retries_per_task - 1:
                            time.sleep(timeout)

                except (google_exceptions.ResourceExhausted, google_exceptions.NotFound) as e:
                    error_type = "配额耗尽" if isinstance(e, google_exceptions.ResourceExhausted) else "模型未找到"
                    logger.error(f"论文 {paper_id}: {error_type} - {task.key_name}-{task.model_name}。标记为永久不可用。")
                    task_manager.mark_task_failed(task, permanent=True)
                    final_result = None # 确保在永久失败时结果为None
                    break # 永久失败，直接跳出当前任务的重试循环，尝试下一个 TaskConfig
                    
                except google_exceptions.TooManyRequests as e:
                    logger.warning(f"论文 {paper_id}: 请求过于频繁 (TooManyRequests) - {task.key_name}-{task.model_name}。")
                    task_manager.mark_task_failed(task, permanent=False) # 标记为临时失败，进行冷却
                    final_result = None # 确保结果为None
                    time.sleep(timeout * 2)  # 更长的等待时间
                    break # 跳出当前任务的重试循环，让任务进入冷却
                    
                except langchain_core.exceptions.OutputParserException as e:
                    logger.error(f"论文 {paper_id}: 输出解析错误 (Pydantic验证失败) - {e}")
                    # Pydantic验证失败通常意味着模型输出了非结构化内容或格式不正确
                    # 视为瞬时错误，尝试退避
                    task_manager.mark_task_failed(task, permanent=False)
                    final_result = None
                    if attempt_within_task < max_retries_per_task - 1:
                        time.sleep(timeout)
                    break # 跳出当前任务重试循环，尝试下一个TaskConfig

                except Exception as e:
                    logger.error(f"论文 {paper_id}: 处理时发生未知瞬时错误: {e}")
                    task_manager.mark_task_failed(task, permanent=False) # 标记为临时失败
                    final_result = None
                    if attempt_within_task < max_retries_per_task - 1:
                        time.sleep(timeout)
                    else:
                        logger.warning(f"论文 {paper_id}: 任务 {task.key_name}-{task.model_name} 在所有重试中均失败。")
                    break # 所有重试失败，跳出当前任务的重试循环，尝试下一个 TaskConfig
            
            total_attempts_for_paper += 1 # 每次尝试一个 TaskConfig 都算一次总尝试
            
            if final_result: # 如果当前 TaskConfig 处理成功
                task_manager.mark_task_success(task)
                stats.update_stats(True)
                logger.info(f"论文 {paper_id} 处理成功。")
                return _merge_ai_result(paper_data, final_result)
        
        # 如果循环结束仍未成功，则标记为失败
        logger.error(f"论文 {paper_id} 最终处理失败，已达到总尝试上限或处理超时。")
        stats.update_stats(False)
        return _merge_ai_result(paper_data, None)
        
    except Exception as e:
        logger.critical(f"论文 {paper_id} 处理函数外部发生严重错误: {e}")
        stats.update_stats(False)
        return _merge_ai_result(paper_data, None)


def _merge_ai_result(paper_data: dict, ai_result: Optional[dict]) -> dict:
    """合并AI结果到论文数据"""
    result = paper_data.copy()
    
    if ai_result:
        result['AI'] = ai_result
    else:
        # AI处理失败，创建错误标记
        error_message = "错误：AI分析失败"
        # 填充所有Structure字段为错误信息
        result['AI'] = {field_name: error_message for field_name in Structure.model_fields.keys()}
        result['AI'][AI_CALL_FAILED_MARKER] = "True" # 增加一个失败标记，使用字符串类型
    
    return result


@contextmanager
def safe_thread_pool(max_workers: int):
    """安全的线程池上下文管理器"""
    executor = None
    try:
        executor = ThreadPoolExecutor(max_workers=max_workers)
        yield executor
    finally:
        if executor:
            executor.shutdown(wait=True)
            logger.info("线程池已关闭。")


def process_papers_batch(
    papers_batch: List[dict],
    model_chains: Dict[Tuple[str, str], Any],
    task_manager: TaskManager,
    rate_limiter: RateLimiter,
    language: str,
    max_retries: int,
    timeout: int,
    max_workers: int,
    stats: ProcessingStats
) -> List[dict]:
    """批量处理论文"""
    results = []
    
    with safe_thread_pool(max_workers) as executor:
        # 提交所有任务，并关联原始论文数据，以便异常时也能找到对应论文
        future_to_paper = {
            executor.submit(
                process_single_paper,
                paper.copy(), # 确保传入副本
                model_chains,
                task_manager,
                rate_limiter,
                language,
                max_retries,
                timeout,
                stats
            ): paper for paper in papers_batch
        }
        
        # 收集结果
        # 设置一个较大的超时，以防某个future长时间不返回
        for future in as_completed(future_to_paper, timeout=MAX_PROCESSING_TIME_PER_PAPER * 1.5): 
            original_paper = future_to_paper[future]
            paper_id = original_paper.get('id', 'unknown')
            try:
                result_data = future.result() # result_data 已经是处理后的字典，包含了AI结果或失败标记
                results.append(result_data)
            except Exception as e:
                logger.error(f"论文 {paper_id} 的处理线程发生未预期异常: {e}")
                # 即使线程本身出现异常，也确保该论文有失败标记
                failed_result = _merge_ai_result(original_paper, None)
                results.append(failed_result)
                stats.update_stats(False) # 确保统计被更新

    return results


def setup_logging(log_level: str):
    """设置日志级别"""
    # 重新设置root logger的级别，确保所有handler都遵循
    logger.setLevel(getattr(logging, log_level.upper()))
    for handler in logger.handlers:
        handler.setLevel(getattr(logging, log_level.upper()))
    logger.info(f"日志级别设置为: {log_level}")


def load_and_validate_data(data_file: str, resume_file: Optional[Path] = None) -> Tuple[List[dict], Set[str]]:
    """加载并验证数据文件"""
    processed_ids = set()
    
    # 不支持恢复模式，因为不保存文件
    if resume_file:
        logger.warning("不支持恢复模式，因为程序不保存输出文件")

    try:
        with open(data_file, "r", encoding="utf-8") as f:
            all_data = [json.loads(line) for line in f if line.strip()]
        
        # 去重
        seen_ids = set()
        data_to_process = []
        for item in all_data:
            item_id = item.get('id')
            if item_id:
                if item_id in seen_ids:
                    logger.debug(f"发现重复论文ID: {item_id}，已跳过。")
                    continue
                seen_ids.add(item_id)

            data_to_process.append(item)
        
        logger.info(f"从 {data_file} 加载了 {len(all_data)} 篇论文，其中 {len(seen_ids)} 篇不重复。")
        logger.info(f"将处理 {len(data_to_process)} 篇论文。")
        return data_to_process, processed_ids
        
    except Exception as e:
        logger.critical(f"加载原始数据文件 {data_file} 时出错: {e}")
        raise


def initialize_model_chains(
    task_manager: TaskManager, # 传入TaskManager以便标记初始化失败的任务
    task_configs: List[TaskConfig], 
    template_content: str, 
    system_prompt_template: str
) -> Dict[Tuple[str, str], Any]:
    """初始化模型链"""
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt_template),
        ("human", template_content)
    ])
    
    model_chains = {}
    
    for task in task_configs:
        key = (task.api_key, task.model_name)
        if key in model_chains: # 避免重复初始化，如果同一个key/model组合出现在task_configs中多次
            continue
        
        try:
            llm = ChatGoogleGenerativeAI(
                model=task.model_name, 
                google_api_key=task.api_key,
                temperature=0.1,  # 降低随机性
                max_output_tokens=2048,  # 限制输出长度
                response_model_strict=True # 开启严格模式，更早捕获Pydantic验证错误
            )
            # with_structured_output 默认会使用Pydantic进行解析和验证
            structured_llm = llm.with_structured_output(Structure)
            chain = prompt_template | structured_llm
            model_chains[key] = chain
            logger.info(f"模型链初始化成功: {task.key_name}-{task.model_name}")
        except Exception as e:
            logger.error(f"模型链初始化失败: {task.key_name}-{task.model_name}, 错误: {e}")
            model_chains[key] = None # 标记为 None，表示初始化失败
            task_manager.mark_task_failed(task, permanent=True) # 初始化失败也视为永久失败

    return model_chains


def save_results(enhanced_data: List[dict], output_filename: str):
    """显示结果统计信息，不保存到文件"""
    try:
        total_papers = len(enhanced_data)
        successful_papers = len([item for item in enhanced_data if not item.get('AI', {}).get(AI_CALL_FAILED_MARKER)])
        failed_papers = total_papers - successful_papers
        
        logger.info("=== 处理结果统计 ===")
        logger.info(f"总论文数: {total_papers}")
        logger.info(f"成功处理: {successful_papers}")
        logger.info(f"处理失败: {failed_papers}")
        logger.info(f"成功率: {(successful_papers / max(1, total_papers) * 100):.1f}%")
        logger.info("结果已完成处理，未保存到文件")
    except Exception as e:
        logger.error(f"显示结果统计时出错: {e}")


def main():
    """主函数"""
    final_enhanced_data = []  # 确保无论异常发生前后都已定义
    output_filename = None    # 确保始终定义，避免未绑定错误
    try:
        # 解析参数
        args = parse_args()
        setup_logging(args.log_level)
        
        logger.info("=== arXiv论文AI增强程序启动 ===")
        logger.info(f"命令行参数: {args}")
        
        # 验证配置
        api_keys, model_names = ConfigValidator.validate_environment()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        template_content, system_prompt_template = ConfigValidator.validate_files(script_dir)
        
        # 构建任务配置
        task_configs = []
        priority = 0
        for model_name in model_names:
            for i, api_key in enumerate(api_keys):
                task_configs.append(TaskConfig(
                    key_name=f"Key_{i+1}",
                    api_key=api_key,
                    model_name=model_name,
                    priority=priority
                ))
            priority += 1
        
        logger.info(f"构建了 {len(task_configs)} 个任务配置")
        
        # 初始化组件
        task_manager = TaskManager(task_configs)
        rate_limiter = RateLimiter(float(os.environ.get("API_CALL_INTERVAL", 6)))
        stats = ProcessingStats()
        language = os.environ.get("LANGUAGE", 'Chinese')
        
        # 确定输出文件名 (仅用于恢复模式检查，实际不保存文件)
        output_filename = "not_saved.jsonl"  # 占位符文件名
        
        # 加载数据
        data_to_process, previously_processed_ids = load_and_validate_data(
            args.data, None  # 不支持恢复模式
        )
        
        # 初始化模型链
        model_chains = initialize_model_chains(task_manager, task_configs, template_content, system_prompt_template)
        
        # 检查是否有可用的模型链
        # 这里的检查应该基于TaskManager中标记为可用的TaskConfig
        if task_manager.get_available_tasks_count() == 0:
            logger.critical("没有可用的模型链或所有任务都已永久失败，程序退出。请检查API密钥和模型名称。")
            return
        
        logger.info(f"成功初始化 {len(model_chains)} 个模型链，其中 {task_manager.get_available_tasks_count()} 个任务当前可用。")
        
        # 批量处理
        # final_enhanced_data 已在函数顶部初始化
        
        # 不支持恢复模式，因为不保存文件
        if args.resume:
            logger.warning("不支持恢复模式，因为程序不保存输出文件")
        
        total_batches = (len(data_to_process) + args.batch_size - 1) // args.batch_size
        if len(data_to_process) == 0:
            logger.info("没有新的论文需要处理。程序退出。")
            return
            
        logger.info(f"开始批量处理 {len(data_to_process)} 篇论文，共 {total_batches} 个批次。")
        
        for i in range(0, len(data_to_process), args.batch_size):
            batch = data_to_process[i:i + args.batch_size]
            batch_num = i // args.batch_size + 1
            
            logger.info(f"处理批次 {batch_num}/{total_batches} ({len(batch)} 篇论文)")
            
            # 显示当前任务状态
            task_status = task_manager.get_task_status()
            logger.info(f"任务状态: {task_status}")
            
            batch_results = process_papers_batch(
                batch, model_chains, task_manager, rate_limiter,
                language, args.retries, args.timeout, args.max_workers, stats
            )
            
            final_enhanced_data.extend(batch_results)
            
            # 显示进度统计
            current_stats = stats.get_stats()
            logger.info(f"批次 {batch_num} 完成。当前统计: "
                       f"成功率 {current_stats['success_rate']:.1f}%, "
                       f"处理速度 {current_stats['papers_per_minute']:.1f} 篇/分钟")


        # 显示最终统计信息，但不保存文件
        save_results(final_enhanced_data, output_filename)
        
        # 最终统计 (基于新处理的数据)
        final_stats = stats.get_stats()
        logger.info("=== 处理完成 ===")
        logger.info(f"总输入论文数: {len(data_to_process) + len(previously_processed_ids)}")
        logger.info(f"本次处理论文数: {len(data_to_process)}")
        logger.info(f"本次成功处理: {final_stats['success']}")
        logger.info(f"本次失败数量: {final_stats['failure']}")
        
        # 成功率应基于本次处理的论文
        if final_stats['processed_count'] > 0:
            logger.info(f"本次处理成功率: {final_stats['success_rate']:.1f}%")
        else:
            logger.info("本次没有论文被处理。")

        logger.info(f"平均处理速度 (本次): {final_stats['papers_per_minute']:.1f} 篇/分钟")
        logger.info(f"总耗时 (本次): {final_stats['elapsed_time']:.2f} 秒")
        logger.info("程序完成，结果未保存到文件")
        
    except KeyboardInterrupt:
        logger.info("程序被用户中断。")
        # 显示中断时的统计信息
        if final_enhanced_data:
            save_results(final_enhanced_data, "interrupted")
            logger.info("中断时已处理数据的统计信息已显示")
    except Exception as e:
        logger.critical(f"程序执行过程中发生未预期错误: {e}", exc_info=True) # 打印堆栈信息
        sys.exit(1)


if __name__ == "__main__":
    main()