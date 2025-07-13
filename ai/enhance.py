#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异步并发版：多密钥、每 Key 10 RPM
"""

import os, sys, json, time, argparse, asyncio
from typing import Dict, Tuple, List, Optional

import dotenv
from google.api_core import exceptions as google_exceptions
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

from structure import Structure   # 你的 Pydantic 输出结构

# ──────────────────────────────────────────────
# 1. CLI & 环境变量
# ──────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Async arXiv AI enhancer")
    p.add_argument("--data", required=True, help="输入 JSONL")
    p.add_argument("--retries", type=int, default=3, help="瞬时错误重试次数")
    p.add_argument("--language", default="Chinese", help="输出语言")
    return p.parse_args()

dotenv.load_dotenv(override=False)
API_KEYS = [k.strip() for k in os.getenv("GOOGLE_API_KEYS", "").split(",") if k.strip()]
MODEL_LIST = [m.strip() for m in os.getenv("MODEL_PRIORITY_LIST", "").split(",") if m.strip()]
if not API_KEYS or not MODEL_LIST:
    sys.exit("❌ 需要环境变量 GOOGLE_API_KEYS 和 MODEL_PRIORITY_LIST")

# RPM_PER_KEY = 10                # 免费 10 请求/分
PER_CALL_INTERVAL = int(os.environ.get("API_CALL_INTERVAL", 6))    # 6 s
RPM_PER_KEY = 60 // PER_CALL_INTERVAL  # 每 Key 的速率限制 (每分钟请求数)

# ──────────────────────────────────────────────
# 2. 速率限制器：每 Key 独立
# ──────────────────────────────────────────────
class KeyLimiter:
    def __init__(self, rpm: int):
        self._interval = 60 / rpm
        self._next_time = 0.0
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        async with self._lock:
            now = time.monotonic()
            wait = self._next_time - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._next_time = max(now, self._next_time) + self._interval

    async def __aexit__(self, exc_type, exc, tb):
        pass  # nothing


# ──────────────────────────────────────────────
# 3. 初始化 Prompt & Chain
# ──────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(script_dir, "template.txt"), encoding="utf-8") as f:
    HUMAN_TMPL = f.read()
with open(os.path.join(script_dir, "system.txt"), encoding="utf-8") as f:
    SYSTEM_TMPL = f.read()

prompt_template = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_TMPL), ("human", HUMAN_TMPL)]
)

# 缓存 (api_key, model) → chain
CHAIN_CACHE: Dict[Tuple[str, str], Optional[object]] = {}
for key in API_KEYS:
    for model in MODEL_LIST:
        try:
            llm = ChatGoogleGenerativeAI(model=model, google_api_key=key)
            chain = prompt_template | llm.with_structured_output(Structure)
            CHAIN_CACHE[(key, model)] = chain
            print(f"✔ 初始化 {model} @ {key[:6]}…")
        except Exception as e:
            CHAIN_CACHE[(key, model)] = None
            print(f"⚠ 无法初始化 {model} @ {key[:6]}…: {e}")

# 为每个 Key 建立限流器
LIMITERS = {key: KeyLimiter(RPM_PER_KEY) for key in API_KEYS}

# ──────────────────────────────────────────────
# 4. 工具函数
# ──────────────────────────────────────────────
def is_response_valid(resp: Structure) -> bool:
    d = resp.model_dump() if resp else {}
    return all((v and str(v).strip()) for v in d.values())

async def try_chain(chain, prompt, limiter, retries):
    """在 limiter 内调用 chain，带重试"""
    for attempt in range(retries):
        try:
            async with limiter:           # 进入限流
                start = time.perf_counter()
                res = await chain.ainvoke(prompt)
                latency = time.perf_counter() - start
            if res and is_response_valid(res):
                return res.model_dump()
        except (google_exceptions.ResourceExhausted,
                google_exceptions.NotFound) as perm:
            raise perm                     # 永久性错误：外层处理
        except Exception as e:
            if attempt < retries - 1:
                await asyncio.sleep(2)     # backoff
    return None

async def process_paper(paper: dict, retries: int, language: str):
    prompt = {
        "title":   paper["title"],
        "content": paper["summary"],
        "language": language,
    }

    # 按模型优先级，再按 Key 顺序
    for model in MODEL_LIST:
        for key in API_KEYS:
            chain = CHAIN_CACHE.get((key, model))
            if not chain:
                continue
            limiter = LIMITERS[key]
            try:
                result = await try_chain(chain, prompt, limiter, retries)
                if result:
                    paper["AI"] = result
                    return paper
            except (google_exceptions.ResourceExhausted,
                    google_exceptions.NotFound):
                # 配额满 / 模型下线：换下一个组合
                continue
            # 其他错误已在内部重试完
    # 所有组合失败
    paper["AI"] = {fld: "ERROR" for fld in Structure.model_fields.keys()}
    return paper

# ──────────────────────────────────────────────
# 5. 主入口
# ──────────────────────────────────────────────
async def main_async():
    args = parse_args()

    # 读取 JSONL，去重 by id
    with open(args.data, encoding="utf-8") as f:
        seen = set()
        data = []
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            if d["id"] not in seen:
                seen.add(d["id"])
                data.append(d)
    print(f"📑 待处理论文数: {len(data)}")

    tasks = [process_paper(p, args.retries, args.language) for p in data]
    results: List[dict] = await asyncio.gather(*tasks, return_exceptions=False)

    # 写文件
    out_file = args.data.replace(".jsonl", f"_AI_enhanced_{args.language}.jsonl")
    with open(out_file, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    ok = sum(1 for r in results if r["AI"]["summary"] != "ERROR")
    print(f"✅ 完成: {ok}/{len(results)} 输出 → {out_file}")

if __name__ == "__main__":
    asyncio.run(main_async())
