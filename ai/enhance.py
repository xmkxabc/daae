#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
enhance_arxiv.py  —  AI-增强 arXiv JSONL（并行 + 进度条）
"""

import os, sys, json, time, argparse, asyncio
from typing import Dict, Tuple, List, Optional, Any

import dotenv
from tqdm import tqdm                        # 进度条
from google.api_core import exceptions as gexc
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from structure import Structure              # 你的 Pydantic 输出结构

# ───────── 1 · 自定义 LLM：彻底禁用所有自动重试 ──────────
def _no_retry_deco(func):
    return func                              # 恒等装饰器

class ChatGoogleNoRetry(ChatGoogleGenerativeAI):
    def __init__(self, *args, **kwargs):
        # ① 让 Google-SDK 不重试
        req_opts = dict(kwargs.pop("request_options", {}) or {})
        req_opts["retry"] = False
        super().__init__(*args, request_options=req_opts, **kwargs)

        # ② LangChain 重试装饰器 → 恒等
        if hasattr(self, "_retry_decorator"):
            self._retry_decorator = _no_retry_deco
        if hasattr(self, "_async_retry_decorator"):
            self._async_retry_decorator = _no_retry_deco

# ───────── 2 · 免费额度表 ────────────────────────────────
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

# ───────── 3 · CLI & ENV ────────────────────────────────
def cli():
    ap = argparse.ArgumentParser(description="Enhance arXiv JSONL with Gemini")
    ap.add_argument("--data", required=True)
    ap.add_argument("--language", default="Chinese")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=10,
                    help="同时并发的论文数（默认10）")
    return ap.parse_args()

dotenv.load_dotenv()
API_KEYS = [k.strip() for k in os.getenv("GOOGLE_API_KEYS", "").split(",") if k.strip()]
MODELS   = [m.strip() for m in os.getenv("MODEL_PRIORITY_LIST", "").split(",") if m.strip()]
if not API_KEYS or not MODELS:
    sys.exit("❌ 缺少 GOOGLE_API_KEYS 或 MODEL_PRIORITY_LIST 环境变量")

# ───────── 4 · ComboLimiter：按 (key, model) 控制 RPM & RPD ─────────
class ComboLimiter:
    def __init__(self, rpm: int, rpd: int):
        self.interval = 60 / rpm
        self.rpd = rpd
        self.calls = 0
        self.next_t = 0.0
        self.exhaust = False
        self.lock = asyncio.Lock()
    async def __aenter__(self):
        if self.exhaust:
            raise RuntimeError("day-quota-exhausted")
        async with self.lock:
            now = time.monotonic()
            wait = self.next_t - now
            if wait > 0:
                await asyncio.sleep(wait)
            self.next_t = max(now, self.next_t) + self.interval
            self.calls += 1
            if self.calls >= self.rpd:
                self.exhaust = True
    async def __aexit__(self, *_):
        return False

# ───────── 5 · Prompt 与 Chain / Limiter 初始化 ──────────
ROOT = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(ROOT, "system.txt"), encoding="utf-8") as f:
    SYS = f.read()
with open(os.path.join(ROOT, "template.txt"), encoding="utf-8") as f:
    HUMAN = f.read()
PROMPT = ChatPromptTemplate.from_messages([("system", SYS), ("human", HUMAN)])

CHAINS: Dict[Tuple[str, str], Optional[Any]] = {}
LIMITER: Dict[Tuple[str, str], ComboLimiter] = {}

for key in API_KEYS:
    for model in MODELS:
        rpm, rpd = quota_of(model)
        LIMITER[(key, model)] = ComboLimiter(rpm, rpd)
        try:
            llm = ChatGoogleNoRetry(model=model, google_api_key=key)
        except TypeError:               # 适配旧版 SDK
            llm = ChatGoogleNoRetry(model=model, api_key=key)
        CHAINS[(key, model)] = PROMPT | llm.with_structured_output(Structure)
        print(f"✔ {model:<18} @ {key[:6]}… RPM={rpm} RPD={rpd}")

def good(res: Structure) -> bool:
    d = res.model_dump()
    return all(v and str(v).strip() for v in d.values())

# ───────── 6 · 封装调用 ─────────────────────────────────
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
            await asyncio.sleep(4)      # RPM 超
        except Exception:
            await asyncio.sleep(2)
    raise RuntimeError("invoke-retries-exhausted")

# ───────── 7 · 处理单篇论文 ────────────────────────────
async def process_paper(paper: dict, lang: str, retries: int):
    prm = {"title": paper["title"], "content": paper["summary"], "language": lang}
    for model in MODELS:
        for key in API_KEYS:
            lim = LIMITER[(key, model)]
            chain = CHAINS[(key, model)]
            if lim.exhaust or chain is None:
                continue
            try:
                res = await invoke(chain, prm, lim, retries)
                if res and good(res):
                    paper["AI"] = res.model_dump()
                    return paper
            except RuntimeError:
                continue
    paper["AI"] = {f: "ERROR" for f in Structure.model_fields.keys()}
    return paper

# ───────── 8 · 主函数（并行 + 进度条） ─────────────────
async def main():
    args = cli()

    # 读文件 & 去重
    with open(args.data, encoding="utf-8") as f:
        seen, papers = set(), []
        for ln in f:
            if ln.strip():
                d = json.loads(ln)
                if d["id"] not in seen:
                    seen.add(d["id"]); papers.append(d)
    total = len(papers)
    print(f"\n📑 待处理论文：{total} 篇  |  并发 {args.concurrency}\n")

    # 并发控制 Semaphore
    sem = asyncio.Semaphore(args.concurrency)
    async def wrapped(p):
        async with sem:
            return await process_paper(p, args.language, args.retries)

    tasks = [wrapped(p) for p in papers]
    processed: List[dict] = []
    with tqdm(total=total, desc="Processing", unit="paper") as bar:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            processed.append(result)
            bar.update()

    # 写结果
    outp = args.data.replace(".jsonl", f"_AI_enhanced_{args.language}.jsonl")
    with open(outp, "w", encoding="utf-8") as f:
        for r in processed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ok_cnt = sum(r["AI"][next(iter(r["AI"]))] != "ERROR" for r in processed)
    print(f"\n✅ 完成 {ok_cnt}/{total} ➜ {outp}")

if __name__ == "__main__":
    asyncio.run(main())
