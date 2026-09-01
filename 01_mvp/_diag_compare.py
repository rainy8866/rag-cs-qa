# -*- coding: utf-8 -*-
"""对比诊断：口语化 vs 难例 prompt，找难例空响应的根因。"""
import json
from core.gold_set_enhance import _read_doc_texts, COLLOQUIAL_PROMPT, HARD_CASE_PROMPT
from core.gold_set_enhance import _strip_json_fence
from core import llm as llm_mod

SIMPLIFIED_HARD_PROMPT = """你是客服问答数据工程师。请根据下面的【原文】生成 {n} 条<b>高难度/边界</b>客服问答样例（泛查询、易混淆、假设性、否定排除、长尾实体、跨章节、边界条件等类型）。

【原文】
{docs}

【要求】
1. 每条含 question / expected_answer / required_snippet 三个字段。
2. required_snippet 必须<b>从原文逐字复制</b>，不得改写。
3. expected_answer 基于原文；若原文不足以完整回答，写明"根据给定信息无法完全确认"。
4. 不要两两重复。

只输出一个 JSON 数组，不要输出任何其他文字。格式：
[{{"question": "...", "expected_answer": "...", "required_snippet": "..."}}, ...]
"""

if __name__ == "__main__":
    doc_texts = _read_doc_texts()
    all_text = "\n".join(doc_texts.values())
    docs = all_text[:24000]
    tests = [
        ("COLLOQUIAL", COLLOQUIAL_PROMPT.format(n=3, docs=docs)),
        ("HARD_FULL", HARD_CASE_PROMPT.format(n=3, docs=docs)),
        ("HARD_SIMPLIFIED", SIMPLIFIED_HARD_PROMPT.format(n=3, docs=docs)),
    ]
    for name, prompt in tests:
        print(f"\n===== {name} (prompt_len={len(prompt)}) =====")
        try:
            resp = llm_mod.chat([{"role": "user", "content": prompt}],
                                temperature=0.7, max_tokens=3000, timeout=120)
            print("resp_len:", len(resp or ""))
            print("head:", (resp or "")[:150].replace("\n", " "))
            try:
                obj = json.loads(_strip_json_fence(resp))
                print("parse ok, n =", len(obj) if isinstance(obj, list) else "dict")
            except Exception as e:
                print("parse fail:", e)
        except Exception as e:
            print("exception:", type(e).__name__, str(e)[:200])
