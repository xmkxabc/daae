#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
enhance_arxiv.py
──────────────────────────────────────────────────────────────
✓ 多 API-Key × 多模型
✓ 免费档限流：模型级 (RPM + RPD)
✓ 完全禁用自动重试：Google SDK + LangChain
──────────────────────────────────────────────────────────────
示例：
  export GOOGLE_API_KEYS="keyA,keyB"
  export MODEL_PRIORITY_LIST="gemini-2.5-flash,gemini-2.5-pro"
  python enhance_arxiv.py --data papers.jsonl --language Chinese
"""

import os, sys, json, time, asyncio, argparse
from typing import Dict, Tuple, List, Optional, Any

import dotenv
from google.api_core import exceptions as gexc
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import _RetryDecorator, _AsyncRetryDecorator
from langchain.prompts import ChatPromptTemplate

from structure import Structure   # ← 你的 Pydantic 输出结构

# ───────────────── 1. 自定义 LLM：彻底禁用所有自动重试 ────────────────
def _identity(f):  # 恒等装饰器
    return f

class ChatGoogleNoRetry(ChatGoogleGenerativeAI):
    """
    1) 给 Google SDK 传 request_options={"retry": False} → 不指数退避
    2) 把 LangChain 内部 _retry_decorator/_async_retry_decorator 改为恒等
    """
    def __init__(self, *args, **kwargs):
        # 1. 强制 request_options.retry=False
        req_opts = dict(kwargs.pop("request_options", {}) or {})
        req_opts["retry"] = False
        super().__init__(*args, request_options=req_opts, **kwargs)

        # 2. 清空 LangChain 层的同步 & 异步 retry 装饰器
        self._retry_decorator = _RetryDecorator(_identity, reraise=True)
        self._async_retry_decorator = _AsyncRetryDecorator(_identity, reraise=True)

# ───────────────── 2. 免费额度表（官方 2025-07） ─────────────────────────
FREE_LIMITS: Dict[str, Dict[str, int]] = {
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
    return 10, 250   # 默认

# ───────────────── 3. CLI & 环境变量 ────────────────────────────────────
def cli():
    ap = argparse.ArgumentParser(description="Enhance arXiv JSONL with Gemini")
    ap.add_argument("--data", required=True, help="输入 JSONL 文件")
    ap.add_argument("--language", default="Chinese")
    ap.add_argument("--retries", type=int, default=3, help="瞬时错误重试次数")
    return ap.parse_args()

dotenv.load_dotenv()
API_KEYS = [k.strip() for k in os.getenv("GOOGLE_API_KEYS", "").split(",") if k.strip()]
MODELS   = [m.strip() for m in os.getenv("MODEL_PRIORITY_LIST", "").split(",") if m.strip()]
if not API_KEYS or not MODELS:
    sys.exit("❌ 请设置环境变量 GOOGLE_API_KEYS 与 MODEL_PRIORITY_LIST")

# ───────────────── 4. ComboLimiter：按 (key, model) 控制 RPM & RPD ──────
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

# ───────────────── 5. Prompt & Chain 初始化 ─────────────────────────────
ROOT = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(ROOT, "system.txt"), encoding="utf-8") as f:
    SYS = f.read()
with open(os.path.join(ROOT, "template.txt"), encoding="utf-8") as f:
    HUMAN = f.read()
PROMPT = ChatPromptTemplate.from_messages([("system", SYS), ("human", HUMAN)])

CHAINS: Dict[Tuple[str, str], Optional[Any]] = {}
LIMITERS: Dict[Tuple[str, str], ComboLimiter] = {}

for key in API_KEYS:
    for model in MODELS:
        rpm, rpd = quota_of(model)
        LIMITERS[(key, model)] = ComboLimiter(rpm, rpd)
        try:
            # 新版 SDK 用 google_api_key=；若提示未知参数就换成 api_key=
            llm = ChatGoogleNoRetry(model=model, google_api_key=key, max_retries=1)
        except TypeError:
            llm = ChatGoogleNoRetry(model=model, api_key=key, max_retries=1)

        chain = PROMPT | llm.with_structured_output(Structure)
        CHAINS[(key, model)] = chain
        print(f"✔ {model:<18} @ {key[:6]}…  RPM={rpm}  RPD={rpd}")

def good(resp: Structure) -> bool:
    d = resp.model_dump(); return all(v and str(v).strip() for v in d.values())

# ───────────────── 6. 调用包装 ──────────────────────────────────────────
async def invoke(chain, prompt, limiter: ComboLimiter, retries: int):
    for _ in range(retries):
        try:
            async with limiter:
                return await chain.ainvoke(prompt)
        except RuntimeError:
            raise                              # 当天额度用光 → 外层封禁
        except gexc.ResourceExhausted as e:
            if "FreeTier" in str(e):
                limiter.exhaust = True
                raise RuntimeError("day-quota-exhausted")
            await asyncio.sleep(4)            # 瞬时超 RPM
        except Exception:
            await asyncio.sleep(2)
    raise RuntimeError("invoke-retries-exhausted")

async def process(paper: dict, lang: str, retries: int):
    prm = {"title": paper["title"], "content": paper["summary"], "language": lang}
    for model in MODELS:
        for key in API_KEYS:
            combo = (key, model)
            lim   = LIMITERS[combo]
            chain = CHAINS[combo]
            if lim.exhaust or chain is None:
                continue
            try:
                print(f"→ {paper['id']} via {model} @ {key[:6]}…")
                res = await invoke(chain, prm, lim, retries)
                if res and good(res):
                    paper["AI"] = res.model_dump()
                    return paper
            except RuntimeError:
                continue
    paper["AI"] = {f: "ERROR" for f in Structure.model_fields.keys()}
    return paper

# ───────────────── 7. 主入口 ───────────────────────────────────────────
async def main():
    args = cli()

    # 读 & 去重
    with open(args.data, encoding="utf-8") as f:
        seen, data = set(), []
        for ln in f:
            if ln.strip():
                d = json.loads(ln)
                if d["id"] not in seen:
                    seen.add(d["id"]); data.append(d)
    print(f"\n📑 待处理：{len(data)} 篇\n")

    results = await asyncio.gather(*(process(p, args.language, args.retries) for p in data))

    outp = args.data.replace(".jsonl", f"_AI_enhanced_{args.language}.jsonl")
    with open(outp, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ok_cnt = sum(r["AI"][next(iter(r["AI"]))] != "ERROR" for r in results)
    print(f"\n✅ 成功 {ok_cnt}/{len(results)} ➜ {outp}")

if __name__ == "__main__":
    asyncio.run(main())
