# -*- coding: utf-8 -*-
"""问答生成：检索 top-k -> 拼 prompt -> DeepSeek 生成带引用答案 -> 解析 citations。"""
import re
from core import config, llm as llm_mod

FALLBACK = "抱歉，知识库中没有找到相关资料，建议改述或联系人工。"

PROMPT_TMPL = """你是客服问答助手。只依据下面提供的资料回答，严禁编造或引入资料外的信息。
若资料不足以确认，请直接回答"资料不足无法确认"。

资料：
{context}

问题：{question}

要求：
1. 先给结论，再补充要点。
2. 用到哪条资料，就在对应句末标注 [编号]（例如 [1]）。不得引用未提供的编号。
3. 最后一行用"参考资料："列出用到的编号与来源文件名。"""


def _build_context(chunks):
    lines = []
    for i, c in enumerate(chunks, start=1):
        src = f"{c.source}" + (f"/{c.heading}" if c.heading else "")
        lines.append(f"[{i}] (来源: {src}) {c.text}")
    return "\n".join(lines)


def _extract_refnums(answer: str) -> list:
    return [int(x) for x in re.findall(r"\[(\d+)\]", answer)]


def _parse_citations(answer: str, chunks: list):
    """把答案里的 [n] 编号映射回 chunk，去掉超出范围的编号。

    同一条资料（编号 n）在答案里被标注多次也只保留一条，避免参考资料重复展示。
    """
    num2chunk = {i: c for i, c in enumerate(chunks, start=1)}
    refs = _extract_refnums(answer)
    citations = []
    seen = set()
    for n in refs:
        if n in num2chunk and n not in seen:
            seen.add(n)
            c = num2chunk[n]
            citations.append({"chunk_id": c.id, "source": c.source,
                              "heading": c.heading, "text": c.text[:400],
                              "score": getattr(c, "score", 0.0)})
    return citations


def _post_check_citations(answer: str, chunks: list) -> str:
    """引用后校验：逐句检查 [n] 引用的块是否真的支撑该句（PRD §7 优化）。

    - 句-最优块相似度 < CITATION_CHECK_THRESHOLD → 没有块能支撑该句，移除引用；
    - 有块支撑但指向的不是最优块 → 改为最相似块的编号；
    - 无效编号 → 移除。
    预计算各块向量，仅对带引用的句子做一次 embedding，避免逐句×逐块重复编码。
    """
    if not chunks:
        return answer  # 无块可校验，原样返回（此处 answer 为入参字符串）
    from core.embeddings import embed_query

    chunk_embs = [embed_query(c.text) for c in chunks]
    num2idx = {i + 1: i for i in range(len(chunks))}  # [n] -> chunk index

    # 中文客服答案按句末标点切句（保留标点），带引用的句子单独处理
    parts = re.split(r"(?<=[。！？!?；;])", answer)
    new_parts = []
    for part in parts:
        if not re.search(r"\[\d+\]", part):
            new_parts.append(part)
            continue
        body = re.sub(r"\[\d+\]", "", part).strip()
        if not body:
            new_parts.append(part)
            continue
        body_emb = embed_query(body)
        sims = [sum(x * y for x, y in zip(body_emb, e)) for e in chunk_embs]
        best_sim = max(sims)
        best_num = sims.index(best_sim) + 1

        def _repl(m):
            ref_num = int(m.group(0)[1:-1])
            if ref_num not in num2idx:
                return ""                       # 无效编号
            if best_sim < config.CITATION_CHECK_THRESHOLD:
                return ""                       # 无块能支撑该句
            return f"[{best_num}]" if best_num != ref_num else m.group(0)

        new_parts.append(re.sub(r"\[\d+\]", _repl, part))
    return "".join(new_parts)


def answer(question: str, use_rerank: bool = False, return_chunks: bool = False,
           use_rewrite: bool = False, use_post_check: bool = False, queries: list = None):
    """返回 dict。调用方负责异常处理与空检索。

    use_rewrite=True 且 config.REWRITE_ENABLED 时：查询改写拆分子查询多路召回。
    传入 queries（已算好的子查询）可复用改写结果，避免重复调 LLM。
    use_post_check=True 时：生成后对引用做语义后校验（修正/移除错误引用）。
    """
    from core.retrieval import retrieve, retrieve_multi
    if use_rewrite and config.REWRITE_ENABLED:
        if queries is None:
            from core.query_rewrite import rewrite
            queries = rewrite(question)
        chunks = retrieve_multi(queries, use_rerank=use_rerank)
    else:
        chunks = retrieve(question, use_rerank=use_rerank)
    if not chunks:
        return {"answer": FALLBACK, "citations": [], "success": False,
                "chunks": chunks}

    context = _build_context(chunks)
    prompt = PROMPT_TMPL.format(context=context, question=question)
    raw = llm_mod.chat([{"role": "user", "content": prompt}])

    # 引用后校验
    if use_post_check:
        raw = _post_check_citations(raw, chunks)

    citations = _parse_citations(raw, chunks)
    result = {"answer": raw.strip(), "citations": citations, "success": True,
              "chunks": chunks}
    if not citations:
        # 模型没标引用：看是不是资料不足兜底；否则标记为无引用
        if "资料不足无法确认" in raw:
            result["success"] = True
    if return_chunks:
        result["chunks"] = chunks
    return result