import os
import json
import sys
import logging
from typing import Dict, Any

import dotenv
import argparse

import langchain_core.exceptions
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from structure import Structure

# 日志配置（可选）
logging.basicConfig(filename='ai_enhance.log', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

if os.path.exists('.env'):
    dotenv.load_dotenv()

# 读取模板和系统提示
with open("template.txt", "r", encoding="utf-8") as f:
    template = f.read()
with open("system.txt", "r", encoding="utf-8") as f:
    system = f.read()

REQUIRED_FIELDS = ["tldr", "motivation", "method", "result", "conclusion"]

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="jsonline data file")
    return parser.parse_args()

def fill_missing_fields(ai_dict: Dict[str, Any]) -> Dict[str, Any]:
    """补全缺失字段，保证输出完整结构"""
    for k in REQUIRED_FIELDS:
        if k not in ai_dict or ai_dict[k] is None:
            ai_dict[k] = "Error"
    return ai_dict

def main():
    args = parse_args()
    model_name = os.environ.get("MODEL_NAME", 'deepseek-chat')
    language = os.environ.get("LANGUAGE", 'Chinese')

    data = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    # 去重
    seen_ids = set()
    unique_data = []
    for item in data:
        if 'id' not in item or 'summary' not in item:
            logging.warning(f"Missing id or summary in item: {item}")
            continue
        if item['id'] not in seen_ids:
            seen_ids.add(item['id'])
            unique_data.append(item)
    data = unique_data

    print('Open:', args.data, file=sys.stderr)

    llm = ChatOpenAI(model=model_name).with_structured_output(Structure, method="function_calling")
    print('Connect to:', model_name, file=sys.stderr)
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system),
        HumanMessagePromptTemplate.from_template(template=template)
    ])

    chain = prompt_template | llm

    output_path = args.data.replace('.jsonl', f'_AI_enhanced_{language}.jsonl')
    # 清空输出文件
    open(output_path, "w", encoding="utf-8").close()

    for idx, d in enumerate(data):
        try:
            response: Structure = chain.invoke({
                "language": language,
                "content": d['summary']
            })
            ai_data = response.model_dump()
        except langchain_core.exceptions.OutputParserException as e:
            print(f"{d['id']} has an error: {e}", file=sys.stderr)
            logging.error(f"{d['id']} OutputParserException: {e}")
            ai_data = {k: "Error" for k in REQUIRED_FIELDS}
        except Exception as e:
            print(f"{d['id']} unexpected error: {e}", file=sys.stderr)
            logging.error(f"{d['id']} Unexpected error: {e}")
            ai_data = {k: "Error" for k in REQUIRED_FIELDS}

        # 补全所有字段
        d['AI'] = fill_missing_fields(ai_data)

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

        print(f"Finished {idx+1}/{len(data)}", file=sys.stderr)

if __name__ == "__main__":
    main()
