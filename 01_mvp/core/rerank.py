# -*- coding: utf-8 -*-
"""Rerank 精排：bge-reranker-v2-m3。

直接用 transformers 新版 API（AutoModel + AutoTokenizer），
规避 FlagEmbedding 内部旧 API `tokenizer.prepare_for_model`（transformers>=5 已移除，会导致
compute_score 崩溃并被 except 吞掉、rerank 静默失效）。错误时降级为原顺序(不阻塞)。
"""
_re = None


def _get_reranker():
    """加载交叉编码器。返回 (model, tokenizer)。"""
    global _re
    if _re is None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        model_id = "BAAI/bge-reranker-v2-m3"
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        model.eval()
        _re = (model, tok)
    return _re


def rerank(query: str, chunks: list, top_k: int) -> list:
    """对候选 chunks 打分重排，返回前 top_k 个。分数仅用于排序，不参与相似度语义。"""
    if not chunks:
        return chunks
    try:
        import torch
        model, tok = _get_reranker()
        pairs = [[query, c.text] for c in chunks]
        inputs = tok(pairs, padding=True, truncation="only_second",
                     max_length=512, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits.view(-1).float()
        scores = logits.tolist()
        if isinstance(scores, float):
            scores = [scores]
        for c, s in zip(chunks, scores):
            c.rerank_score = s
        ordered = sorted(chunks, key=lambda c: getattr(c, "rerank_score", 0.0),
                         reverse=True)
        return ordered[:top_k]
    except Exception:
        return chunks[:top_k]
