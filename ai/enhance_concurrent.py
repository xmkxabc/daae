import os
import json
import sys
import dotenv
import argparse
import time
import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
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
class WorkerConfig:
    """工作线程配置"""
    worker_id: int
    key_name: str
    api_key: str
    model_name: str
    rpm_limit: int = 6  # 每分钟请求数限制
    

@dataclass
class ProcessingStats:
    """处理统计信息"""
    total_papers: int = 0
    processed_papers: int = 0
    successful_papers: int = 0
    failed_papers: int = 0
    worker_stats: Dict[int, Dict[str, int]] = None
    start_time: float = 0
    
    def __post_init__(self):
        if self.worker_stats is None:
            self.worker_stats = defaultdict(lambda: {'processed': 0, 'success': 0, 'failed': 0})


class RateLimiter:
    """为单个API密钥实现率限制器"""
    
    def __init__(self, rpm: int = 6):
        self.rpm = rpm
        self.interval = 60.0 / rpm  # 请求间隔（秒）
        self.last_request_time = 0
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """如果需要，等待以遵守速率限制"""
        with self.lock:
            current_time = time.time()
            time_since_last_request = current_time - self.last_request_time
            
            if time_since_last_request < self.interval:
                sleep_time = self.interval - time_since_last_request
                time.sleep(sleep_time)
            
            self.last_request_time = time.time()


class PaperProcessor:
    """并发论文处理器"""
    
    def __init__(self, worker_configs: List[WorkerConfig], prompt_template, 
                 retries: int = 3, timeout: int = 1, language: str = "Chinese"):
        self.worker_configs = worker_configs
        self.prompt_template = prompt_template
        self.retries = retries
        self.timeout = timeout
        self.language = language
        
        # 为每个worker创建模型链和率限制器
        self.worker_chains = {}
        self.rate_limiters = {}
        self.worker_status = {}  # 跟踪worker状态
        
        self._initialize_workers()
        
        # 线程安全的统计信息
        self.stats = ProcessingStats()
        self.stats_lock = threading.Lock()
        
    def _initialize_workers(self):
        """初始化所有worker的模型链和率限制器"""
        for config in self.worker_configs:
            try:
                llm = ChatGoogleGenerativeAI(model=config.model_name, google_api_key=config.api_key)
                structured_llm = llm.with_structured_output(Structure)
                chain = self.prompt_template | structured_llm
                
                self.worker_chains[config.worker_id] = chain
                self.rate_limiters[config.worker_id] = RateLimiter(config.rpm_limit)
                self.worker_status[config.worker_id] = 'active'
                
                print(f"Worker {config.worker_id} ({config.key_name}) 已初始化: {config.model_name}", file=sys.stderr)
                
            except Exception as e:
                self.worker_chains[config.worker_id] = None
                self.rate_limiters[config.worker_id] = None
                self.worker_status[config.worker_id] = 'failed'
                print(f"警告：Worker {config.worker_id} ({config.key_name}) 初始化失败: {e}", file=sys.stderr)
    
    def _update_stats(self, worker_id: int, success: bool):
        """线程安全地更新统计信息"""
        with self.stats_lock:
            self.stats.processed_papers += 1
            self.stats.worker_stats[worker_id]['processed'] += 1
            
            if success:
                self.stats.successful_papers += 1
                self.stats.worker_stats[worker_id]['success'] += 1
            else:
                self.stats.failed_papers += 1
                self.stats.worker_stats[worker_id]['failed'] += 1
    
    def _print_progress(self):
        """打印处理进度"""
        with self.stats_lock:
            elapsed_time = time.time() - self.stats.start_time
            rate = self.stats.processed_papers / (elapsed_time / 60) if elapsed_time > 0 else 0
            
            print(f"\n=== 处理进度 ===", file=sys.stderr)
            print(f"已处理: {self.stats.processed_papers}/{self.stats.total_papers} "
                  f"(成功: {self.stats.successful_papers}, 失败: {self.stats.failed_papers})", file=sys.stderr)
            print(f"处理速度: {rate:.1f} 篇/分钟", file=sys.stderr)
            print(f"运行时间: {elapsed_time:.1f} 秒", file=sys.stderr)
            
            # 打印每个worker的统计信息
            for worker_id, stats in self.stats.worker_stats.items():
                if stats['processed'] > 0:
                    config = next(c for c in self.worker_configs if c.worker_id == worker_id)
                    print(f"  Worker {worker_id} ({config.key_name}): {stats['processed']} 处理, "
                          f"{stats['success']} 成功, {stats['failed']} 失败", file=sys.stderr)
            print("================", file=sys.stderr)
    
    def process_single_paper(self, paper_data: Dict[str, Any], worker_id: int) -> Dict[str, Any]:
        """使用指定worker处理单篇论文"""
        config = next(c for c in self.worker_configs if c.worker_id == worker_id)
        chain = self.worker_chains.get(worker_id)
        rate_limiter = self.rate_limiters.get(worker_id)
        
        if not chain or not rate_limiter or self.worker_status[worker_id] != 'active':
            return self._create_error_result(paper_data, "Worker不可用")
        
        paper_id = paper_data['id']
        
        for attempt in range(self.retries):
            try:
                # 遵守速率限制
                rate_limiter.wait_if_needed()
                
                print(f"  Worker {worker_id} ({config.key_name}) 处理 {paper_id} (尝试 {attempt + 1}/{self.retries})", file=sys.stderr)
                
                response_object = chain.invoke({
                    "title": paper_data['title'],
                    "content": paper_data['summary'],
                    "language": self.language
                })
                
                if response_object and self._is_response_valid(response_object):
                    result = paper_data.copy()
                    result['AI'] = response_object.model_dump()
                    self._update_stats(worker_id, True)
                    print(f"  ✓ Worker {worker_id} 成功处理 {paper_id}", file=sys.stderr)
                    return result
                
            except (google_exceptions.ResourceExhausted, google_exceptions.NotFound) as e:
                error_type = "配额耗尽" if isinstance(e, google_exceptions.ResourceExhausted) else "模型未找到"
                print(f"  ! Worker {worker_id} {error_type}: {paper_id}", file=sys.stderr)
                # 标记worker为不可用
                self.worker_status[worker_id] = 'quota_exhausted'
                break
                
            except Exception as e:
                print(f"  × Worker {worker_id} 瞬时错误处理 {paper_id}: {e}", file=sys.stderr)
                if attempt < self.retries - 1:
                    time.sleep(self.timeout)
        
        # 所有尝试都失败了
        self._update_stats(worker_id, False)
        return self._create_error_result(paper_data, f"Worker {worker_id} 处理失败")
    
    def _is_response_valid(self, result: Structure) -> bool:
        """验证响应，确保所有字段都为非空字符串"""
        if not result:
            return False
        result_dict = result.model_dump()
        all_fields = Structure.model_fields.keys()
        for field in all_fields:
            value = result_dict.get(field)
            if value is None or (isinstance(value, str) and not value.strip()):
                return False
        return True
    
    def _create_error_result(self, paper_data: Dict[str, Any], error_message: str) -> Dict[str, Any]:
        """创建错误结果"""
        result = paper_data.copy()
        result['AI'] = {field: f"错误：{error_message}" for field in Structure.model_fields.keys()}
        return result
    
    def process_papers_concurrent(self, papers: List[Dict[str, Any]], 
                                max_workers: Optional[int] = None) -> List[Dict[str, Any]]:
        """并发处理论文列表"""
        self.stats.total_papers = len(papers)
        self.stats.start_time = time.time()
        
        if max_workers is None:
            max_workers = len([w for w in self.worker_configs if self.worker_status.get(w.worker_id) == 'active'])
        
        print(f"\n开始并发处理 {len(papers)} 篇论文，使用 {max_workers} 个worker", file=sys.stderr)
        
        results = [None] * len(papers)  # 保持顺序
        
        # 创建任务队列：(paper_index, paper_data)
        task_queue = Queue()
        for i, paper in enumerate(papers):
            task_queue.put((i, paper))
        
        def worker_thread(worker_id: int):
            """worker线程函数"""
            while True:
                try:
                    # 获取任务，超时1秒
                    paper_index, paper_data = task_queue.get(timeout=1)
                    
                    # 检查worker是否仍然可用
                    if self.worker_status[worker_id] != 'active':
                        # Worker不可用，将任务放回队列
                        task_queue.put((paper_index, paper_data))
                        break
                    
                    # 处理论文
                    result = self.process_single_paper(paper_data, worker_id)
                    results[paper_index] = result
                    
                    task_queue.task_done()
                    
                    # 每处理5篇论文打印一次进度
                    if self.stats.processed_papers % 5 == 0:
                        self._print_progress()
                        
                except Empty:
                    # 队列为空，退出
                    break
                except Exception as e:
                    print(f"Worker {worker_id} 遇到未预期错误: {e}", file=sys.stderr)
                    break
        
        # 启动worker线程
        active_workers = [w.worker_id for w in self.worker_configs 
                         if self.worker_status.get(w.worker_id) == 'active']
        
        if not active_workers:
            print("错误：没有可用的worker", file=sys.stderr)
            return [self._create_error_result(paper, "没有可用的worker") for paper in papers]
        
        threads = []
        for worker_id in active_workers[:max_workers]:
            thread = threading.Thread(target=worker_thread, args=(worker_id,))
            thread.start()
            threads.append(thread)
        
        # 等待所有任务完成
        task_queue.join()
        
        # 等待所有线程结束
        for thread in threads:
            thread.join()
        
        self._print_progress()
        
        # 处理任何剩余的None结果（失败的任务）
        for i, result in enumerate(results):
            if result is None:
                results[i] = self._create_error_result(papers[i], "处理超时或失败")
        
        return results


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="使用AI摘要增强arXiv数据（并发版本）。")
    parser.add_argument("--data", type=str, required=True, help="要处理的JSONL数据文件。")
    parser.add_argument("--retries", type=int, default=3, help="对每个模型任务的最大重试次数。")
    parser.add_argument("--timeout", type=int, default=1, help="失败尝试之间的等待秒数。")
    parser.add_argument("--max-workers", type=int, default=None, help="最大并发worker数量。")
    parser.add_argument("--rpm-limit", type=int, default=6, help="每个API密钥的每分钟请求数限制。")
    return parser.parse_args()


def main():
    """主函数，运行增强过程。"""
    args = parse_args()
    
    # --- 加载统一的密钥和模型优先级列表 ---
    google_api_keys_str = os.environ.get("GOOGLE_API_KEYS")
    model_priority_list_str = os.environ.get("MODEL_PRIORITY_LIST")

    if not google_api_keys_str or not model_priority_list_str:
        print("错误: 请在 .env 文件中设置 GOOGLE_API_KEYS 和 MODEL_PRIORITY_LIST 环境变量。", file=sys.stderr)
        sys.exit(1)

    api_keys = [key.strip() for key in google_api_keys_str.split(',') if key.strip()]
    model_names = [name.strip() for name in model_priority_list_str.split(',') if name.strip()]

    if not api_keys or not model_names:
        print("错误: GOOGLE_API_KEYS 或 MODEL_PRIORITY_LIST 环境变量不能为空。", file=sys.stderr)
        sys.exit(1)
        
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

    # 创建worker配置（使用第一个模型作为主要模型）
    worker_configs = []
    worker_id = 0
    
    # 为每个API密钥创建worker
    for i, api_key in enumerate(api_keys):
        config = WorkerConfig(
            worker_id=worker_id,
            key_name=f"密钥_{i+1}",
            api_key=api_key,
            model_name=model_names[0],  # 使用优先级最高的模型
            rpm_limit=args.rpm_limit
        )
        worker_configs.append(config)
        worker_id += 1
    
    print(f"\n=== 并发处理配置 ===", file=sys.stderr)
    print(f"API密钥数量: {len(api_keys)}", file=sys.stderr)
    print(f"主要模型: {model_names[0]}", file=sys.stderr)
    print(f"RPM限制: {args.rpm_limit} 请求/分钟/密钥", file=sys.stderr)
    print(f"理论最大处理速度: {len(api_keys) * args.rpm_limit} 篇/分钟", file=sys.stderr)
    print("====================", file=sys.stderr)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt_template),
        ("human", template_content)
    ])

    # 创建并发处理器
    processor = PaperProcessor(
        worker_configs=worker_configs,
        prompt_template=prompt_template,
        retries=args.retries,
        timeout=args.timeout,
        language=language
    )

    # 处理论文
    enhanced_data = processor.process_papers_concurrent(data, args.max_workers)

    # 输出结果
    output_filename = args.data.replace('.jsonl', f'_AI_enhanced_{language}.jsonl')
    with open(output_filename, "w", encoding="utf-8") as f:
        for d_item in enhanced_data:
            f.write(json.dumps(d_item, ensure_ascii=False) + "\n")

    # 最终统计
    total_time = time.time() - processor.stats.start_time
    final_rate = len(enhanced_data) / (total_time / 60) if total_time > 0 else 0
    
    print(f"\n=== 处理完成 ===", file=sys.stderr)
    print(f"总论文数: {len(enhanced_data)}", file=sys.stderr)
    print(f"成功处理: {processor.stats.successful_papers}", file=sys.stderr)
    print(f"处理失败: {processor.stats.failed_papers}", file=sys.stderr)
    print(f"总耗时: {total_time:.1f} 秒", file=sys.stderr)
    print(f"平均处理速度: {final_rate:.1f} 篇/分钟", file=sys.stderr)
    print(f"输出文件: {output_filename}", file=sys.stderr)
    print("==================", file=sys.stderr)


if __name__ == "__main__":
    main()