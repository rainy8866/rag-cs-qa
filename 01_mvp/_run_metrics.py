# -*- coding: utf-8 -*-
"""双口径全量指标重跑（并发 4 线程 + 改写结果缓存 + 进度日志）。

- 关口径：retrieve + answer(use_rewrite=False, use_post_check=False, 无 Rerank)
- 开口径：rewrite→retrieve_multi（改写结果缓存复用，不再重复调 LLM）+ Rerank 精排 + answer(use_rewrite=True, use_post_check=True, use_rerank=True)
坏例分类复用已算好的答案（不二次调 LLM）；LLM 调用带重试。
报告落盘 02_优化/compare/（md + json 双口径存档，保留可比性；对照上一轮"开=改写+校验(无 Rerank)"存档数据）。
"""
import json
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from core import config, gold_set
from core import qa as qa_mod
from core.vectorstore import collection_count
from core.retrieval import retrieve, retrieve_multi
from core.query_rewrite import rewrite

WORKERS = 4


def log(msg):
    print(msg, flush=True)


def _retry(fn, tries=3, base=2.0):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            if i == tries - 1:
                raise
            log(f"    ! 重试 {i+1}/{tries}: {type(e).__name__}")
            time.sleep(base * (i + 1))


def valid_items(items):
    """过滤 snippet 映射不到 chunk 的条目，回填 _chunk_id。"""
    from core.gold_set import snippet_to_chunk
    valid = []
    for it in items:
        cid = snippet_to_chunk(it.get("required_snippet", ""))
        if cid is None:
            continue
        it = dict(it)
        it["_chunk_id"] = cid
        valid.append(it)
    return valid


def run_recall(valid, rewrites, use_rerank=False):
    """rewrites=None → 单路 retrieve；否则多路 retrieve_multi（用已缓存改写）。返回 (hit, missed)。"""
    hit, missed = 0, []
    for it in valid:
        res = retrieve_multi(rewrites[it["question"]], use_rerank=use_rerank) if rewrites is not None \
            else retrieve(it["question"], use_rerank=use_rerank)
        ids = {c.id for c in res}
        if it["_chunk_id"] in ids:
            hit += 1
        else:
            missed.append({"question": it["question"], "snippet": it["required_snippet"],
                           "expected_chunk": it["_chunk_id"], "got_chunks": [c.id for c in res]})
    return hit, missed


def run_citation(valid, use_rewrite, use_post_check, use_rerank=False, cached=None):
    """并发跑 answer。返回 {question: res}。"""
    config.REWRITE_ENABLED = use_rewrite

    def work(it):
        q = it["question"]
        if use_rewrite and config.REWRITE_ENABLED:
            queries = cached.get(q) if cached is not None else None
            if queries is None:
                queries = _retry(lambda: rewrite(q))
                if cached is not None:
                    cached[q] = queries
            return q, _retry(lambda: qa_mod.answer(q, use_rerank=use_rerank, use_rewrite=True,
                                                   use_post_check=use_post_check, queries=queries))
        return q, _retry(lambda: qa_mod.answer(q, use_rerank=use_rerank, use_post_check=use_post_check))

    answers, done = {}, 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(work, it) for it in valid]
        for fut in as_completed(futs):
            q, res = fut.result()
            answers[q] = res
            done += 1
            if done % 20 == 0 or done == len(valid):
                log(f"    {done}/{len(valid)}")
    return answers


def build_cite(valid, answers):
    """由已算好的 answers 统计引用准确率，返回与 metrics.citation_accuracy 同构的 dict。"""
    correct, total_refs, details = 0, 0, []
    for it in valid:
        res = answers[it["question"]]
        refs = [c["chunk_id"] for c in res["citations"]]
        total_refs += len(refs)
        ok = it["_chunk_id"] in refs
        if ok:
            correct += 1
        details.append({"question": it["question"], "expected_chunk": it["_chunk_id"],
                        "cited_chunks": refs, "correct": ok, "answer": res["answer"][:180]})
    return {"accuracy": round(correct / len(valid), 4), "correct": correct,
            "total": len(valid), "details": details, "citation_count": total_refs}


def classify(valid, missed_qs, answers):
    """坏例分类：复用已算好的答案，不二次调 LLM。"""
    cats = ["recall_miss", "citation_error", "hallucination", "empty_answer"]
    counts = {c: 0 for c in cats}
    items = {c: [] for c in cats}
    for it in valid:
        q = it["question"]
        if q in missed_qs:
            cat = "recall_miss"
        else:
            res = answers[q]
            refs = [c["chunk_id"] for c in res["citations"]]
            if it["_chunk_id"] not in refs:
                cat = "citation_error"
            elif not res.get("success") or not res.get("answer"):
                cat = "empty_answer"
            elif not res.get("citations") and "资料不足" not in res.get("answer", ""):
                cat = "hallucination"
            else:
                continue
        counts[cat] += 1
        items[cat].append({"question": q, "expected_answer": it.get("expected_answer", ""),
                           "required_snippet": it.get("required_snippet", "")})
    return {"counts": counts, "items": items, "total": sum(counts.values())}


def main():
    config.ensure_dirs()
    gs = gold_set.load_gold_set()
    items = gs["items"]
    ver = gs["version"]
    valid = valid_items(items)
    total = len(valid)
    n_chunks = collection_count()
    log(f"评估集 {ver} · {len(items)} 条 · 有效 {total} · chunks {n_chunks} · 模型 {config.LLM_MODEL}")

    # ---- 关口径 ----
    log("[关口径] 召回（无 LLM）...")
    hit_off, missed_off = run_recall(valid, None)
    log(f"[关口径] 召回 {hit_off}/{total}")
    log(f"[关口径] 引用生成（并发 {WORKERS}）...")
    ans_off = run_citation(valid, False, False)
    cite_off = build_cite(valid, ans_off)
    bc_off = classify(valid, {m["question"] for m in missed_off}, ans_off)
    log(f"[关口径] 引用 {cite_off['correct']}/{cite_off['total']} · 坏例 {bc_off['total']}")

    # ---- 开口径（改写 + Rerank + 引用后校验）----
    log(f"[开口径] 查询改写（并发 {WORKERS}）...")
    config.REWRITE_ENABLED = True
    rewrites, done = {}, 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(_retry, lambda q=q: rewrite(q)): q for q in [it["question"] for it in valid]}
        for fut in as_completed(futs):
            rewrites[futs[fut]] = fut.result()
            done += 1
            if done % 20 == 0 or done == total:
                log(f"    改写 {done}/{total}")
    log("[开口径] 召回（多路 + Rerank，无 LLM）...")
    hit_on, missed_on = run_recall(valid, rewrites, use_rerank=True)
    log(f"[开口径] 召回 {hit_on}/{total}")
    log(f"[开口径] 引用生成（并发 {WORKERS}，改写走缓存 + Rerank）...")
    ans_on = run_citation(valid, True, True, use_rerank=True, cached=rewrites)
    cite_on = build_cite(valid, ans_on)
    bc_on = classify(valid, {m["question"] for m in missed_on}, ans_on)
    log(f"[开口径] 引用 {cite_on['correct']}/{cite_on['total']} · 坏例 {bc_on['total']}")

    def fmt(r, c, b):
        return (f"{r[0]:.2%} ({r[0]*total:.0f}/{total})",
                f"{c['accuracy']:.2%} ({c['correct']}/{c['total']})",
                f"{b['total']} {json.dumps(b['counts'], ensure_ascii=False)}")

    ro, co, bo = fmt((hit_off / total,), cite_off, bc_off)
    rn, cn, bn = fmt((hit_on / total,), cite_on, bc_on)

    log("=== 双口径全量指标（开含 Rerank） ===")
    log(f"| 指标 | 关(基线) | 开(改写+Rerank+校验) |")
    log(f"| 召回率 | {ro} | {rn} |")
    log(f"| 引用准确率 | {co} | {cn} |")
    log(f"| 坏例 | {bo} | {bn} |")

    log(f"\n开口径仍召回未命中：{len(missed_on)}")
    for m in missed_on:
        log(f"  - {m['question'][:30]} | exp={m['expected_chunk']} got={m['got_chunks']}")

    # ---- 落盘双口径报告 ----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 加载上一轮"开=改写+校验(无 Rerank)"存档，做历史可比对照
    prev = {}
    prev_json = config.COMPARE_DIR / "compare_rewrite_20260829_065149.json"
    if prev_json.exists():
        try:
            prev = json.loads(prev_json.read_text(encoding="utf-8"))
        except Exception:
            prev = {}

    prev_line = ""
    if prev:
        pv = prev.get("on", {})
        pb = pv.get("badcase", {})
        prev_line = (f"上一轮开(改写+校验,无Rerank): 召回 {pv.get('recall', 0):.2%} · "
                     f"引用 {pv.get('citation', 0):.2%} · 坏例 {sum(pb.values())} "
                     f"{json.dumps(pb, ensure_ascii=False)}")

    lines = [
        "# 优化双口径指标报告（查询改写 + Rerank + 引用后校验）",
        f"- 生成时间：{datetime.now().isoformat()}",
        f"- 评估集版本：{ver} · {len(items)} 条 · 有效 {total} · chunks：{n_chunks}",
        f"- 生成模型：{config.LLM_MODEL}（2026-08-29 由 deepseek-v4-flash 切至 deepseek-chat，非推理、省 token）",
        f"- 口径说明：关=原检索+原生成（无改写/无Rerank）；开=查询改写多路召回 + Rerank 精排 + 引用后校验（阈值 {config.CITATION_CHECK_THRESHOLD}）",
        f"- Rerank 模型：BAAI/bge-reranker-v2-m3（transformers 直调，交叉编码器精排）",
    ]
    if prev_line:
        lines.append(f"- 历史对照：{prev_line}")
    lines += [
        "",
        "| 指标 | 关(基线) | 开(改写+Rerank+校验) | 差值(开-关) |",
        "|---|---|---|---|",
        f"| 召回率 | {ro} | {rn} | {hit_on/total - hit_off/total:+.4f} |",
        f"| 引用准确率 | {co} | {cn} | {cite_on['accuracy'] - cite_off['accuracy']:+.4f} |",
        f"| 坏例 | {bo} | {bn} | - |",
        "",
        "## 开口径仍坏明细（真实排名短板）",
    ]
    if missed_on:
        for m in missed_on:
            lines.append(f"- 问：{m['question']}")
            lines.append(f"  - 应命中：{m['expected_chunk']} | 实际：{m['got_chunks']}")
    else:
        lines.append("- 无")
    lines.append("")
    lines.append("## 开口径仍坏问题清单")
    for it in bc_on["items"].get("recall_miss", []):
        lines.append(f"- {it['question']}")
    for it in bc_on["items"].get("citation_error", []):
        lines.append(f"- {it['question']}")
    lines.append("")
    lines.append("## 关口径坏例问题清单（对照）")
    for it in bc_off["items"].get("recall_miss", []):
        lines.append(f"- {it['question']}")
    for it in bc_off["items"].get("citation_error", []):
        lines.append(f"- {it['question']}")

    md_path = config.COMPARE_DIR / f"compare_rewrite_{ts}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    payload = {
        "ts": datetime.now().isoformat(), "version": ver, "items": len(items),
        "valid": total, "n_chunks": n_chunks,
        "params": {"llm_model": config.LLM_MODEL, "rewrite_enabled": True,
                   "use_rewrite": True, "use_post_check": True,
                   "use_rerank": True, "rerank_model": "BAAI/bge-reranker-v2-m3",
                   "rerank_candidates": config.RERANK_CANDIDATES,
                   "citation_check_threshold": config.CITATION_CHECK_THRESHOLD,
                   "rewrite_model": config.REWRITE_MODEL, "workers": WORKERS},
        "prev_round_on_no_rerank": prev.get("on", {}) if prev else None,
        "off": {"recall": round(hit_off / total, 4), "citation": cite_off["accuracy"],
                "badcase": bc_off["counts"]},
        "on": {"recall": round(hit_on / total, 4), "citation": cite_on["accuracy"],
               "badcase": bc_on["counts"]},
        "on_remaining_miss": [{"q": m["question"], "exp": m["expected_chunk"],
                               "got": m["got_chunks"]} for m in missed_on],
        "off_badcase_questions": [it["question"] for c in bc_off["items"].values() for it in c],
        "on_badcase_questions": [it["question"] for c in bc_on["items"].values() for it in c],
    }
    json_path = config.COMPARE_DIR / f"compare_rewrite_{ts}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"\n报告落盘：{md_path}\n{json_path}")


if __name__ == "__main__":
    main()
