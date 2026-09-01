# -*- coding: utf-8 -*-
"""召回预检：全 125 条，rerank 开 vs 关（无 LLM，纯检索）。只读，不写文件。"""
from core import config, gold_set
from core.gold_set import snippet_to_chunk
from core.retrieval import retrieve

gs = gold_set.load_gold_set()
items = []
for it in gs["items"]:
    cid = snippet_to_chunk(it.get("required_snippet", ""))
    if cid is not None:
        items.append((it["question"], cid))
print(f"有效条数: {len(items)}")


def run(use_rerank):
    hits, miss = 0, []
    for q, cid in items:
        ids = [c.id for c in retrieve(q, use_rerank=use_rerank)]
        if cid in ids:
            hits += 1
        else:
            miss.append((q, cid, ids))
    return hits, miss


if __name__ == "__main__":
    h0, m0 = run(False)
    h1, m1 = run(True)
    print(f"\n[关] 召回 {h0}/{len(items)} ({h0/len(items):.2%})  坏例 {len(m0)}")
    print(f"[开] 召回 {h1}/{len(items)} ({h1/len(items):.2%})  坏例 {len(m1)}")

    m1q = {q for q, _, _ in m1}
    m0q = {q for q, _, _ in m0}
    fixed = sorted(m0q - m1q)
    regress = sorted(m1q - m0q)
    print(f"\nRerank 修复坏例 ({len(fixed)}):")
    for q in fixed:
        print(f"  - {q}")
    print(f"\nRerank 新增/回归坏例 ({len(regress)}):")
    for q in regress:
        print(f"  - {q}")
    print("\n[开] 仍坏明细:")
    for q, cid, ids in m1:
        print(f"  exp={cid}  {q}  -> {ids}")
