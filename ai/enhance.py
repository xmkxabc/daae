#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Async arXiv enhancer —— 多 API-Key × 多模型；自动按官方免费额度限流
"""

import os, sys, json, time, argparse, asyncio, re
from typing import Dict, Tuple, Optional

import dotenv
from google.api_core import exceptions as gexc
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

from structure import Structure

# ───────────────────────── 免费额度表 ──────────────────────────
# 来源：Gemini API docs › Rate limits › Free tier table.
FREE_LIMITS: Dict[str, Dict[str, int]] = {
    # model-prefix        rpm   rpd
    "gemini-2.5-pro":    {"rpm": 5,  "rpd": 100},
    "gemini-2.5-flash":  {"rpm": 10, "rpd": 250},
    "gemini-2.5-flash-l":{"rpm": 15, "rpd": 1000},  # 2.5 Flash-Lite Preview
    "gemini-2.0-flash":  {"rpm": 15, "rpd": 200},
    "gemini-2.0-flash-l":{"rpm": 30, "rpd": 200},   # 2.0 Flash-Lite
    "gemini-1.5-flash":  {"rpm": 15, "rpd": 50},    # deprecated
    "gemini-1.5-pro":    {"rpm": 2,  "rpd": 50},    # deprecated
}

def limit_of(model: str):
    """返回 (rpm, rpd)；未知模型默认 (10, 250)"""
    for prefix, lim in FREE_LIMITS.items():
        if model.startswith(prefix):
            return lim["rpm"], lim["rpd"]
    return 10, 250

# ───────────────────────── CLI & 环境 ─────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Enhance arXiv JSONL with Gemini")
    p.add_argument("--data", required=True, help="输入 JSONL 文件")
    p.add_argument("--language", default="Chinese")
    p.add_argument("--retries", type=int, default=3)
    return p.parse_args()

dotenv.load_dotenv(override=False)
API_KEYS = [k.strip() for k in os.getenv("GOOGLE_API_KEYS", "").split(",") if k.strip()]
MODELS   = [m.strip() for m in os.getenv("MODEL_PRIORITY_LIST", "").split(",") if m.strip()]
if not API_KEYS or not MODELS:
    sys.exit("❌ 请配置 GOOGLE_API_KEYS、MODEL_PRIORITY_LIST")

# ───────────────────────── 限流器 ─────────────────────────────
class ComboLimiter:
    """单 Key+Model 粒度：RPM + 今日 RPD"""
    def __init__(self, rpm: int, rpd: int):
        self.interval = 60 / rpm
        self.rpd      = rpd
        self.calls    = 0
        self.next_t   = 0.0
        self.lock     = asyncio.Lock()
        self.exhaust  = False

    async def __aenter__(self):
        if self.exhaust:
            raise RuntimeError("daily-quota-exhausted")
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

# ───────────────────────── Prompt & Chain 缓存 ────────────────
root = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(root, "template.txt"), encoding="utf-8") as f:
    HUMAN = f.read()
with open(os.path.join(root, "system.txt"), encoding="utf-8") as f:
    SYS   = f.read()
PROMPT = ChatPromptTemplate.from_messages([("system", SYS), ("human", HUMAN)])

CHAINS: Dict[Tuple[str, str], Optional[object]] = {}
LIMITS: Dict[Tuple[str, str], ComboLimiter] = {}

for key in API_KEYS:
    for model in MODELS:
        rpml, rpdl = limit_of(model)
        LIMITS[(key, model)] = ComboLimiter(rpml, rpdl)
        try:
            llm = ChatGoogleGenerativeAI(model=model,
                                         google_api_key=key,
                                         max_retries=0)   # 禁止 SDK 重试
            CHAINS[(key, model)] = PROMPT | llm.with_structured_output(Structure)
            print(f"✔ {model:<18} @ {key[:6]}… RPM={rpml}, RPD={rpdl}")
        except Exception as e:
            CHAINS[(key, model)] = None
            print(f"⚠ 初始化失败 {model} @ {key[:6]}… {e}")

def valid(res: Structure):
    d = res.model_dump()
    return all(v and str(v).strip() for v in d.values())

# ───────────────────────── 调用函数 ───────────────────────────
async def invoke(chain, prompt, limiter: ComboLimiter, retries=3):
    for _ in range(retries):
        try:
            async with limiter:
                return await chain.ainvoke(prompt)
        except RuntimeError:            # daily 用光
            raise
        except gexc.ResourceExhausted as e:
            if "FreeTier" in str(e):
                limiter.exhaust = True
                raise RuntimeError("daily-quota-exhausted")
            await asyncio.sleep(4)      # RPM 超限，退避
        except Exception:
            await asyncio.sleep(2)
    raise RuntimeError("all-retries-failed")

async def process(paper: dict, lang: str, retries: int):
    prm = {"title": paper["title"], "content": paper["summary"], "language": lang}
    for model in MODELS:
        for key in API_KEYS:
            combo = (key, model)
            lim   = LIMITS[combo]
            if lim.exhaust or not CHAINS[combo]:
                continue
            try:
                print(f"→ {paper['id']} using {model} @ {key[:6]}…")
                res = await invoke(CHAINS[combo], prm, lim, retries)
                if res and valid(res):
                    paper["AI"] = res.model_dump()
                    return paper
            except RuntimeError:
                continue
    paper["AI"] = {f: "ERROR" for f in Structure.model_fields.keys()}
    return paper

# ───────────────────────── 主程 ───────────────────────────────
async def main():
    args = parse_args()

    # 读文件 & 去重
    with open(args.data, encoding="utf-8") as f:
        seen, data = set(), []
        for ln in f:
            if ln.strip():
                d = json.loads(ln)
                if d["id"] not in seen:
                    seen.add(d["id"])
                    data.append(d)
    print(f"📑 总待处理: {len(data)} 篇")

    tasks = [process(p, args.language, args.retries) for p in data]
    done  = await asyncio.gather(*tasks)

    outp = args.data.replace(".jsonl", f"_AI_enhanced_{args.language}.jsonl")
    with open(outp, "w", encoding="utf-8") as f:
        for row in done:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    ok = sum(r["AI"][next(iter(r["AI"]))] != "ERROR" for r in done)
    print(f"✅ 完成 {ok}/{len(done)} → {outp}")

if __name__ == "__main__":
    asyncio.run(main())
