#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
enhance_arxiv.py  —  多 Key × 多模型 × 免费限流（兼容旧版 SDK，无自动重试）
"""

import os, sys, json, time, asyncio, argparse
from typing import Dict, Tuple, List, Optional, Any

import dotenv
from google.api_core import exceptions as gexc
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from google.api_core.retry import Retry

from structure import Structure

# ─────────────────── 1 · 自定义 LLM：彻底禁掉 SDK 重试 ───────────────────
_ID = lambda f: f
class ChatGoogleNoRetry(ChatGoogleGenerativeAI):
    def _get_retry_decorator(self, *_, **__):       # 同步
        return _ID
    def _get_retry_decorator_async(self, *_, **__): # 异步
        return _ID

# ─────────────────── 2 · 免费额度表（官方 2025-07） ───────────────────
FREE_LIMITS = {
    "gemini-2.5-flash":   {"rpm": 10, "rpd": 250},
    "gemini-2.5-pro":     {"rpm": 5,  "rpd": 100},
    "gemini-2.5-flash-l": {"rpm": 15, "rpd": 1000},
    "gemini-2.0-flash":   {"rpm": 15, "rpd": 200},
    "gemini-2.0-flash-l": {"rpm": 30, "rpd": 200},
    "gemini-1.5-flash":   {"rpm": 15, "rpd": 50},
    "gemini-1.5-pro":     {"rpm": 2,  "rpd": 50},
}
def quota_of(model: str) -> Tuple[int, int]:
    for p, q in FREE_LIMITS.items():
        if model.startswith(p):
            return q["rpm"], q["rpd"]
    return 10, 250

# ─────────────────── 3 · CLI & 环境变量 ───────────────────
def cli():
    ap = argparse.ArgumentParser(description="Enhance arXiv JSONL with Gemini")
    ap.add_argument("--data", required=True)
    ap.add_argument("--language", default="Chinese")
    ap.add_argument("--retries", type=int, default=3)
    return ap.parse_args()

dotenv.load_dotenv()
API_KEYS = [k.strip() for k in os.getenv("GOOGLE_API_KEYS", "").split(",") if k.strip()]
MODELS   = [m.strip() for m in os.getenv("MODEL_PRIORITY_LIST", "").split(",") if m.strip()]
if not API_KEYS or not MODELS:
    sys.exit("❌ 请设置 GOOGLE_API_KEYS 与 MODEL_PRIORITY_LIST")

# ─────────────────── 4 · ComboLimiter ───────────────────
class ComboLimiter:
    def __init__(self, rpm: int, rpd: int):
        self.interval = 60 / rpm
        self.rpd = rpd
        self.calls = 0
        self.next = 0.0
        self.exhaust = False
        self.lock = asyncio.Lock()
    async def __aenter__(self):
        if self.exhaust:
            raise RuntimeError("day-quota-exhausted")
        async with self.lock:
            now = time.monotonic()
            wait = self.next - now
            if wait > 0:
                await asyncio.sleep(wait)
            self.next = max(now, self.next) + self.interval
            self.calls += 1
            if self.calls >= self.rpd:
                self.exhaust = True
    async def __aexit__(self, *_): ...

# ─────────────────── 5 · Prompt & Chain 初始化 ───────────────────
root = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(root, "system.txt"), encoding="utf-8") as f:
    SYS = f.read()
with open(os.path.join(root, "template.txt"), encoding="utf-8") as f:
    HUMAN = f.read()
PROMPT = ChatPromptTemplate.from_messages([("system", SYS), ("human", HUMAN)])

CHAINS:  Dict[Tuple[str, str], Optional[Any]] = {}
LIMITER: Dict[Tuple[str, str], ComboLimiter] = {}

for key in API_KEYS:
    for model in MODELS:
        rpm, rpd = quota_of(model)
        LIMITER[(key, model)] = ComboLimiter(rpm, rpd)

        # —— 兼容两套参数名 —— #
        try:
            llm = ChatGoogleNoRetry(model=model,
                                    google_api_key=key,  # 新版
                                    max_retries=1)
        except TypeError as e:
            if "google_api_key" in str(e): # 检查是否是旧版本导致的不兼容错误
                try:
                    # 回退到旧版的 'api_key' 参数
                    llm = ChatGoogleNoRetry(model=model,
                                            api_key=key,      # 旧版
                                            max_retries=1)
                except Exception: # 如果回退尝试也失败，则将此组合视为不可用
                    llm = None
            else:
                llm = None
        if llm is None:
            CHAIN = None
        else:
            CHAIN = PROMPT | llm.with_structured_output(Structure)

        CHAINS[(key, model)] = CHAIN
        stat = "✔" if CHAIN else "⚠️"
        print(f"{stat} {model:<18} @ {key[:6]}… RPM={rpm}, RPD={rpd}")

def ok(resp: Structure) -> bool:
    d = resp.model_dump(); return all(v and str(v).strip() for v in d.values())

# ─────────────────── 6 · 封装调用 ───────────────────
async def invoke(chain, prompt, lim: ComboLimiter, retries: int):
    for _ in range(retries):
        try:
            async with lim:
                return await chain.ainvoke(prompt)
        except RuntimeError:
            raise
        except gexc.ResourceExhausted as e:
            if "FreeTier" in str(e):
                lim.exhaust = True
                raise RuntimeError("day-quota-exhausted")
            await asyncio.sleep(4)           # RPM 超, 退避
        except Exception:
            await asyncio.sleep(2)
    raise RuntimeError("retry-failed")

async def process(paper: dict, lang: str, retries: int):
    prm = {"title": paper["title"], "content": paper["summary"], "language": lang}
    for m in MODELS:
        for k in API_KEYS:
            lim = LIMITER[(k, m)]; chain = CHAINS[(k, m)]
            if lim.exhaust or chain is None: continue
            try:
                print(f"→ {paper['id']} via {m} @ {k[:6]}…")
                r = await invoke(chain, prm, lim, retries)
                if r and ok(r):
                    paper["AI"] = r.model_dump(); return paper
            except RuntimeError:
                continue
    paper["AI"] = {f: "ERROR" for f in Structure.model_fields.keys()}
    return paper

# ─────────────────── 7 · 主函数 ───────────────────
async def main():
    args = cli()
    with open(args.data, encoding="utf-8") as f:
        seen, data = set(), []
        for ln in f:
            if ln.strip():
                d = json.loads(ln)
                if d["id"] not in seen:
                    seen.add(d["id"]); data.append(d)
    print(f"\n📑 待处理：{len(data)} 篇\n")

    res = await asyncio.gather(*(process(p, args.language, args.retries) for p in data))

    outp = args.data.replace(".jsonl", f"_AI_enhanced_{args.language}.jsonl")
    with open(outp, "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok_cnt = sum(r["AI"][next(iter(r["AI"]))] != "ERROR" for r in res)
    print(f"\n✅ 成功 {ok_cnt}/{len(res)} ➜ {outp}")

if __name__ == "__main__":
    asyncio.run(main())
