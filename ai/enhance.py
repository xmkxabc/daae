#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
enhance_arxiv.py
──────────────────────────────────────────────────────────────
✓ 多 API-Key × 多模型
✓ 免费档限流：模型级别 RPM + RPD
✓ Google SDK 零重试：自定义 ChatGoogleNoRetry
──────────────────────────────────────────────────────────────
用法示例：
  export GOOGLE_API_KEYS="keyA,keyB,keyC"
  export MODEL_PRIORITY_LIST="gemini-2.5-flash,gemini-2.5-pro"
  python enhance_arxiv.py --data papers.jsonl --language Chinese
"""

import os, sys, json, time, asyncio, argparse, re
from typing import Dict, Tuple, Optional, Any

import dotenv
from google.api_core.retry import Retry
from google.api_core import exceptions as gexc
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

from structure import Structure                 # ← 你自己的 Pydantic 输出结构

# ────────────────────────────── 自定义 LLM ──────────────────────────────
# 关闭 Google SDK 的指数退避（Retry → 0 次）
_NO_RETRY = Retry(predicate=lambda *_: False, max_tries=1)

class ChatGoogleNoRetry(ChatGoogleGenerativeAI):
    """调用 Gemini API，但完全不自动重试。"""
    def _get_retry_decorator(
        self,
        run_manager: Optional[Any] = None,
        max_retries: Optional[int] = None,
    ):
        return _NO_RETRY

# ────────────────────── 免费额度表：参考官方文档 ────────────────────────
FREE_LIMITS: Dict[str, Dict[str, int]] = {
    # prefix                   rpm   rpd
    "gemini-2.5-flash":      {"rpm": 10, "rpd": 250},
    "gemini-2.5-pro":        {"rpm": 5,  "rpd": 100},
    "gemini-2.5-flash-l":    {"rpm": 15, "rpd": 1000},
    "gemini-2.0-flash":      {"rpm": 15, "rpd": 200},
    "gemini-2.0-flash-l":    {"rpm": 30, "rpd": 200},
    "gemini-1.5-flash":      {"rpm": 15, "rpd": 50},
    "gemini-1.5-pro":        {"rpm": 2,  "rpd": 50},
}

def quota_of(model: str) -> Tuple[int, int]:
    """返回 (rpm, rpd)，未知模型默认 10/250。"""
    for prefix, lim in FREE_LIMITS.items():
        if model.startswith(prefix):
            return lim["rpm"], lim["rpd"]
    return 10, 250

# ─────────────────────────── Command-line ───────────────────────────────
def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Asynchronously enhance arXiv JSONL with Gemini")
    p.add_argument("--data", required=True, help="输入 JSONL")
    p.add_argument("--language", default="Chinese")
    p.add_argument("--retries", type=int, default=3, help="瞬时错误重试次数")
    return p.parse_args()

dotenv.load_dotenv(override=False)
API_KEYS = [k.strip() for k in os.getenv("GOOGLE_API_KEYS", "").split(",") if k.strip()]
MODELS   = [m.strip() for m in os.getenv("MODEL_PRIORITY_LIST", "").split(",") if m.strip()]
if not API_KEYS or not MODELS:
    sys.exit("❌ 环境变量 GOOGLE_API_KEYS 或 MODEL_PRIORITY_LIST 未设置。")

# ─────────────────────────── 限流器 ──────────────────────────────────────
class ComboLimiter:
    """按 (key, model) 粒度同时控制 RPM 与 RPD。"""
    def __init__(self, rpm: int, rpd: int):
        self.interval = 60 / rpm
        self.rpd = rpd
        self.calls = 0
        self.next_t = 0.0
        self.exhausted = False
        self.lock = asyncio.Lock()

    async def __aenter__(self):
        if self.exhausted:
            raise RuntimeError("daily-limit-exhausted")
        async with self.lock:
            now = time.monotonic()
            wait = self.next_t - now
            if wait > 0:
                await asyncio.sleep(wait)
            self.next_t = max(now, self.next_t) + self.interval
            self.calls += 1
            if self.calls >= self.rpd:
                self.exhausted = True

    async def __aexit__(self, *_):
        return False

# ──────────────────── Prompt 模板与 Chain 构建 ───────────────────────────
root = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(root, "template.txt"), encoding="utf-8") as f:
    HUMAN_TMPL = f.read()
with open(os.path.join(root, "system.txt"), encoding="utf-8") as f:
    SYSTEM_TMPL = f.read()
PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM_TMPL),
                                           ("human",  HUMAN_TMPL)])

CHAINS: Dict[Tuple[str, str], Optional[Any]] = {}
LIMITERS: Dict[Tuple[str, str], ComboLimiter] = {}

for key in API_KEYS:
    for model in MODELS:
        rpm, rpd = quota_of(model)
        LIMITERS[(key, model)] = ComboLimiter(rpm, rpd)
        try:
            llm = ChatGoogleNoRetry(model=model,
                                    google_api_key=key,
                                    max_retries=1)        # 只调用一次
            chain = PROMPT | llm.with_structured_output(Structure)
            CHAINS[(key, model)] = chain
            print(f"✔  {model:<18} @ {key[:6]}…  RPM={rpm}  RPD={rpd}")
        except Exception as e:
            CHAINS[(key, model)] = None
            print(f"⚠️  初始化失败 {model} @ {key[:6]}…：{e}")

def is_valid(res: Structure) -> bool:
    d = res.model_dump()
    return all(v and str(v).strip() for v in d.values())

# ──────────────────── 调用封装 ───────────────────────────────────────────
async def try_invoke(chain, prompt, limiter: ComboLimiter, retries: int):
    for _ in range(retries):
        try:
            async with limiter:
                return await chain.ainvoke(prompt)
        except RuntimeError:                      # day-quota exhausted
            raise
        except gexc.ResourceExhausted as e:
            if "FreeTier" in str(e):
                limiter.exhausted = True
                raise RuntimeError("daily-limit-exhausted")
            await asyncio.sleep(4)                # RPM 超限，稍退避
        except Exception:
            await asyncio.sleep(2)
    raise RuntimeError("all-retries-failed")

async def process_paper(paper: dict, lang: str, retries: int):
    prm = {"title": paper["title"], "content": paper["summary"], "language": lang}

    for model in MODELS:
        for key in API_KEYS:
            combo = (key, model)
            lim   = LIMITERS[combo]
            chain = CHAINS[combo]
            if lim.exhausted or chain is None:
                continue
            try:
                print(f"→ {paper['id']}  via {model} @ {key[:6]}…")
                res = await try_invoke(chain, prm, lim, retries)
                if res and is_valid(res):
                    paper["AI"] = res.model_dump()
                    return paper
            except RuntimeError:
                continue
    paper["AI"] = {f: "ERROR" for f in Structure.model_fields.keys()}
    return paper

# ──────────────────── 主入口 ────────────────────────────────────────────
async def main():
    args = cli()

    # 读取 JSONL & 去重
    with open(args.data, encoding="utf-8") as f:
        seen, data = set(), []
        for ln in f:
            if ln.strip():
                d = json.loads(ln)
                if d["id"] not in seen:
                    seen.add(d["id"])
                    data.append(d)
    print(f"\n📑 代处理论文数: {len(data)}\n")

    tasks = [process_paper(p, args.language, args.retries) for p in data]
    results = await asyncio.gather(*tasks)

    out_file = args.data.replace(".jsonl", f"_AI_enhanced_{args.language}.jsonl")
    with open(out_file, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    ok = sum(r["AI"][next(iter(r["AI"]))] != "ERROR" for r in results)
    print(f"\n✅ 成功 {ok}/{len(results)}  ➜  {out_file}")

if __name__ == "__main__":
    asyncio.run(main())
