# -*- coding: utf-8 -*-
"""评估集自动生成：用 DeepSeek 根据真实原文生成 20~30 条(问题/答案/应命中片段)，
并做文档源版本快照。用户无需手写、无需填 chunk_id。"""
import json
import shutil
from datetime import datetime

from core import config, llm as llm_mod

GEN_PROMPT = """你是评估数据工程师。请严格根据下面给定的【原文】生成 {n} 条客服问答评估样例。
不得使用原文之外的信息。

【原文】
{docs}

【生成要求】
1. 每条样例包含三个字段：
   - question: 一个真实用户会提出的问题。
   - expected_answer: 基于原文的答案(若原文不足以完整回答，请在其中写明"根据给定信息无法完全确认")。
   - required_snippet: 一段<b>必须从原文逐字复制</b>的句子/片段，不得改写。用于检验该知识点是否被检索命中。
2. 问题要覆盖多种问法，确保多样性：
   - 直接询问（什么是…？）
   - 请求确认（是否可以说…？）
   - 寻求解释（请解释…）
   - 假设性提问（如果…会怎样？）
   - 求例子（能否举个例子…）
3. required_snippet 必须能直接在原文中找到原文。

只输出一个 JSON 数组，不要输出任何其他文字。格式：
[{{"question": "...", "expected_answer": "...", "required_snippet": "..."}}, ...]
"""


def _doc_fingerprint() -> str:
    files = sorted(config.DOC_DIR.glob("*.md"))
    sig = ",".join(f"{f.name}:{f.stat().st_size}" for f in files)
    import hashlib
    return hashlib.md5(sig.encode("utf-8", errors="ignore")).hexdigest()[:8]


def _read_docs_text() -> str:
    parts = []
    for f in sorted(config.DOC_DIR.glob("*.md")):
        parts.append(f"===== {f.name} =====\n" + f.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(parts)


def _save_snapshot(version: str):
    snap = config.SNAPSHOT_DIR / version
    doc_snap = snap / "docs_snapshot"
    doc_snap.mkdir(parents=True, exist_ok=True)
    for f in config.DOC_DIR.glob("*.md"):
        shutil.copy(f, doc_snap / f.name)
    (snap / "doc_fingerprint.txt").write_text(
        f"fingerprint={_doc_fingerprint()}\ncreated_at={datetime.now().isoformat()}",
        encoding="utf-8")


def _strip_json_fence(text: str) -> str:
    """提取模型返回里的 JSON 主体，兼容 ```json fence、语言标记、前后解释文字。"""
    text = (text or "").strip()
    # 去掉 ``` fence 行和语言标记（如 ```json）
    if "```" in text:
        kept = []
        for ln in text.splitlines():
            if ln.strip().startswith("```"):
                continue
            kept.append(ln)
        text = "\n".join(kept).strip()
    # 找第一个 [ 或 {，按括号配对提取完整 JSON 主体（处理字符串内转义/括号）
    start = -1
    for i, ch in enumerate(text):
        if ch in "[{":
            start = i
            break
    if start < 0:
        return text
    open_ch = text[start]
    close_ch = "]" if open_ch == "[" else "}"
    depth = 0
    in_str = False
    esc = False
    for j in range(start, len(text)):
        ch = text[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return text[start:j + 1]
    return text[start:]


def generate_gold_set(num: int = 30) -> dict:
    config.ensure_dirs()
    docs = _read_docs_text()
    if not docs.strip():
        return {"ok": False, "reason": "04_文档源 无文档"}

    # 分批生成（每批 BATCH 条）：绕过 reasoning 模型长输出把 max_tokens 用在推理上、
    # 导致正式 content 为空被截断的问题。每批量小，推理能完成、content 有输出。
    BATCH = 5
    items = []
    failed = 0
    last_resp = ""
    batches = (num + BATCH - 1) // BATCH
    for i in range(batches):
        want = min(BATCH, num - i * BATCH)
        prompt = (f"（第 {i + 1}/{batches} 批，请生成与其它批不同的问题，避免重复）\n"
                  + GEN_PROMPT.format(n=want, docs=docs[:12000]))
        try:
            resp = llm_mod.chat([{"role": "user", "content": prompt}],
                                temperature=0.2, max_tokens=4000)
            batch_items = json.loads(_strip_json_fence(resp))
            if isinstance(batch_items, list):
                items.extend(batch_items)
            last_resp = resp
        except Exception:
            failed += 1
            continue

    if not items:
        raise RuntimeError(f"评估集生成返回无法解析（所有批次失败）：{last_resp[:300]}")

    # 批间按 question 文本去重
    seen = set()
    uniq = []
    for it in items:
        q = (it.get("question") or "").strip()
        if q and q not in seen:
            seen.add(q)
            uniq.append(it)
    items = uniq
    version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    record = {
        "version": version,
        "created_at": datetime.now().isoformat(),
        "doc_set": _doc_fingerprint(),
        "params": {
            "CHUNK_SIZE": config.CHUNK_SIZE,
            "CHUNK_OVERLAP": config.CHUNK_OVERLAP,
            "EMBED_MODEL": config.EMBED_MODEL,
            "MIN_SCORE": config.MIN_SCORE,
            "SIMILARITY_METRIC": config.SIMILARITY_METRIC,
            "LLM_MODEL": config.LLM_MODEL,
        },
        "items": items,
    }
    config.gold_set_path().write_text(json.dumps(record, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    _save_snapshot(version)
    return {"ok": True, "n": len(items), "version": version}


def load_gold_set(force: bool = False) -> dict:
    """读取评估集(自动 + 可选手工合并)。返回 {'version','items':[...],'params':{...}}。"""
    config.ensure_dirs()
    auto = {"version": "none", "items": []}
    if config.gold_set_path().exists():
        auto = json.loads(config.gold_set_path().read_text(encoding="utf-8"))
    manual = {"items": []}
    if config.gold_set_manual_path().exists():
        manual = json.loads(config.gold_set_manual_path().read_text(encoding="utf-8"))
        if not isinstance(manual, dict):
            manual = {"items": manual if isinstance(manual, list) else []}
    merged_items = list(auto.get("items", [])) + list(manual.get("items", []))
    return {"version": auto.get("version", "none"),
            "items": merged_items,
            "params": auto.get("params", {})}


def enhance_gold_set(n_colloquial: int = 12) -> dict:
    """评估集增强入口：抽检现有评估集 + 补充真实口语化问题 + 合并更新为新版本。

    具体逻辑在新增模块 core/gold_set_enhance.py；本函数只做入口转发，
    不改动 generate_gold_set() / load_gold_set() 的既有行为。
    """
    from core.gold_set_enhance import enhance
    return enhance(n_colloquial=n_colloquial)


def snippet_to_chunk(snippet: str):
    """在入库 chunks 里找包含 snippet 的 chunk，返回其 id；找不到返回 None。

    若多个 chunk 都包含 snippet（如相邻块重叠区），取与 snippet 内容最相似（嵌入余弦）的那个，
    避免重叠机制导致映射到语义混杂的低分块、造成指标误判（双口径依据见 PRD §6.5）。"""
    from core.vectorstore import get_collection
    col = get_collection()
    n = col.count()
    if n == 0:
        return None
    # 取全部(文档量不大)并比对子串
    res = col.get(include=["documents", "metadatas"])
    ids = res["ids"]
    docs = res["documents"]
    cands = [i for i, doc in enumerate(docs) if doc and snippet and snippet in doc]
    if not cands:
        return None
    if len(cands) == 1:
        return ids[cands[0]]
    # 多个候选块都包含 snippet：选内容最相似的块，消除重叠歧义
    from core.embeddings import embed_query
    svec = embed_query(snippet)
    best_i, best_sim = cands[0], -1.0
    for i in cands:
        dvec = embed_query(docs[i])
        sim = sum(x * y for x, y in zip(svec, dvec))  # normalize 后点积即余弦
        if sim > best_sim:
            best_sim, best_i = sim, i
    return ids[best_i]


def rebuild_index_for_mapping() -> int:
    """确保向量库已存在；返回 chunk 数量。"""
    from core.ingest import build_index
    from core.vectorstore import collection_count
    if collection_count() == 0:
        build_index()
    return collection_count()


if __name__ == "__main__":
    config.ensure_dirs()
    r = generate_gold_set()
    print(r)