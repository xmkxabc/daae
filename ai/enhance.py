import os
import json
import sys
import dotenv
import argparse
import time

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


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="使用AI摘要增强arXiv数据。")
    parser.add_argument("--data", type=str, required=True, help="要处理的JSONL数据文件。")
    parser.add_argument("--retries", type=int, default=3, help="对每个模型任务的最大重试次数。")
    parser.add_argument("--timeout", type=int, default=1, help="失败尝试之间的等待秒数。")
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
    google_api_keys_str = os.environ.get("GOOGLE_API_KEYS")
    model_priority_list_str = os.environ.get("MODEL_PRIORITY_LIST")
    # [新] 从环境变量加载API调用间隔，默认为6秒以遵循10 RPM的限制
    api_call_interval = int(os.environ.get("API_CALL_INTERVAL", 6))


    if not google_api_keys_str or not model_priority_list_str:
        print("错误: 请在 .env 文件中设置 GOOGLE_API_KEYS 和 MODEL_PRIORITY_LIST 环境变量。", file=sys.stderr)
        sys.exit(1)

    api_keys = [key.strip() for key in google_api_keys_str.split(',') if key.strip()]
    model_names = [name.strip() for name in model_priority_list_str.split(',') if name.strip()]

    if not api_keys or not model_names:
        print("错误: GOOGLE_API_KEYS 或 MODEL_PRIORITY_LIST 环境变量不能为空。", file=sys.stderr)
        sys.exit(1)
        
    # --- [核心改造] 构建级联调用计划 ---
    # 策略: 优先使用最高优先级的模型，轮询所有密钥。
    cascade_plan = []
    # 外层循环遍历模型列表 (Outer loop for models)
    for model_name in model_names:
        # 内层循环遍历密钥列表 (Inner loop for keys)
        for i, api_key in enumerate(api_keys):
            cascade_plan.append({
                "key_name": f"密钥_{i+1}",
                "api_key": api_key,
                "model_name": model_name
            })
            
    if not cascade_plan:
        print("错误: 无法根据环境变量构建有效的调用计划。", file=sys.stderr)
        sys.exit(1)
        
    print("--- 调用计划已构建 ---", file=sys.stderr)
    for i, task in enumerate(cascade_plan):
        print(f"  优先级 {i+1}: <{task['key_name']}> - {task['model_name']}", file=sys.stderr)
    print("----------------------", file=sys.stderr)


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

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt_template),
        ("human", template_content)
    ])

    # 预先初始化所有需要的调用链
    model_chains = {}
    for task in cascade_plan:
        key = (task["api_key"], task["model_name"])
        if key in model_chains: continue
        try:
            llm = ChatGoogleGenerativeAI(model=task["model_name"], google_api_key=task["api_key"])
            structured_llm = llm.with_structured_output(Structure)
            chain = prompt_template | structured_llm
            model_chains[key] = chain
            print(f"模型已为<{task['key_name']}>成功设置: {task['model_name']}", file=sys.stderr)
        except Exception as e:
            model_chains[key] = None
            print(f"警告：无法为<{task['key_name']}>初始化模型 {task['model_name']}。错误：{e}", file=sys.stderr)

    enhanced_data = []
    total_failures = 0
    
    current_task_index = 0

    for idx, d in enumerate(data):
        print(f"\n正在处理 {idx + 1}/{len(data)}: {d['id']}", file=sys.stderr)
        final_result = None
        
        # 保存当前任务索引，以便在论文级别进行重置
        paper_start_task_index = current_task_index
        
        while paper_start_task_index < len(cascade_plan):
            task = cascade_plan[paper_start_task_index]
            key = (task["api_key"], task["model_name"])
            chain = model_chains.get(key)

            if not chain:
                print(f"  ! 跳过已失败的任务: <{task['key_name']}> - {task['model_name']}", file=sys.stderr)
                paper_start_task_index += 1
                # 更新全局任务索引
                current_task_index = paper_start_task_index
                continue

            for attempt in range(args.retries):
                print(f"  使用: <{task['key_name']}> - {task['model_name']} (尝试 {attempt + 1}/{args.retries})", file=sys.stderr)
                try:
                    response_object = chain.invoke({
                        "title": d['title'],
                        "content": d['summary'],
                        "language": language
                    })
                    if response_object and is_response_valid(response_object):
                        final_result = response_object.model_dump()
                        print("  > 尝试成功", file=sys.stderr)
                        break 

                # **核心升级**: 将 NotFound 和 ResourceExhausted 视为同类永久性错误
                except (google_exceptions.ResourceExhausted, google_exceptions.NotFound) as e:
                    error_type = "配额耗尽" if isinstance(e, google_exceptions.ResourceExhausted) else "模型未找到"
                    print(f"  ! {error_type}: <{task['key_name']}> - {task['model_name']}", file=sys.stderr)
                    # 永久切换到下一个任务
                    paper_start_task_index += 1
                    # 更新全局任务索引
                    current_task_index = paper_start_task_index
                    # 跳出重试循环，让外层while循环决定下一步
                    break 

                except Exception as e:
                    print(f"  > 发生瞬时性错误: {e}", file=sys.stderr)
                    if attempt < args.retries - 1:
                        time.sleep(args.timeout)
            
            if final_result:
                break
        
        if not final_result:
            total_failures += 1
            print(f"  处理 {d['id']} 失败。所有可用任务均已尝试失败。", file=sys.stderr)
            error_message = "错误：AI分析失败。"
            d['AI'] = {field: error_message for field in Structure.model_fields.keys()}
        else:
            d['AI'] = final_result
            
        enhanced_data.append(d)
        # [核心改造] 使用可配置的延迟，确保不超过API频率限制
        print(f"  ...等待 {api_call_interval} 秒...", file=sys.stderr)
        time.sleep(api_call_interval)

    output_filename = args.data.replace('.jsonl', f'_AI_enhanced_{language}.jsonl')
    with open(output_filename, "w", encoding="utf-8") as f:
        for d_item in enhanced_data:
            f.write(json.dumps(d_item, ensure_ascii=False) + "\n")

    print(f"\n处理完成。成功处理: {len(enhanced_data) - total_failures}/{len(enhanced_data)}。输出文件: {output_filename}")

if __name__ == "__main__":
    main()
