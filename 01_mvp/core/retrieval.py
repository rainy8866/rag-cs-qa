# -*- coding: utf-8 -*-
"""检索：向量相似度召回（bge-m3 + cosine）。注意 distance->similarity 换算。"""
from core import config
from core.embeddings import embed_query

SENT = '为这个句子生成表示以用于检索相关文章：'


class Chunk:
    def __init__(self, chunk_id, text, source, heading, score):
        self.id = chunk_id
        self.text = text
        self.source = source
        self.heading = heading
        self.score = score

    @property
    def index(self) -> int:
        try:
            return int(self.id)
        except (TypeError, ValueError):
            return -1

    def to_dict(self):
        return {"id": self.id, "text": self.text, "source": self.source,
                "heading": self.heading, "score": round(self.score, 4)}


def retrieve(query: str, top_k: int = None, use_rerank: bool = False, min_score: float = None) -> list:
    """向量相似度召回。返回按相似度降序的 Chunk 列表(过滤掉低于阈值的)。"""
    from core.vectorstore import query_collection
    top_k = top_k or config.TOP_K
    min_score = config.MIN_SCORE if min_score is None else min_score

    # Rerank 时要放宽候选到 RERANK_CANDIDATES，先召回候选再做精排
    candidate_k = config.RERANK_CANDIDATES if use_rerank else top_k
    qvec = embed_query(config.QUERY_PREFIX + query)

    raw = query_collection(qvec, top_k=candidate_k)

    chunks = []
    for item in raw:
        # ★ 关键：distance -> similarity —— cosine distance 约等于 1 - 相似度
        sim = round(1.0 - item["distance"], 4)
        chunks.append(Chunk(
            chunk_id=item["id"],
            text=item["text"],
            source=item["metadata"].get("source", ""),
            heading=item["metadata"].get("heading", ""),
            score=sim,
        ))

    # 低于阈值闸门的过滤掉（score >= min_score 才保留）
    passed = [c for c in chunks if c.score >= min_score]

    # 去重：内容相同的块只保留相似度最高的一条（按归一化文本判断，避免重复召回）
    dedup = {}
    for c in passed:
        key = "".join((c.text or "").split()) or str(c.id)
        if key not in dedup or c.score > dedup[key].score:
            dedup[key] = c
    passed = list(dedup.values())

    if use_rerank:
        from core.rerank import rerank
        # rerank 已按交叉编码器分降序取 top_k，直出精排顺序，不再按 bi-encoder 重排
        return rerank(query, passed, top_k)

    # 排序取 top_k
    passed.sort(key=lambda c: c.score, reverse=True)
    return passed[:top_k]


def retrieve_multi(queries: list, top_k: int = None, use_rerank: bool = False, min_score: float = None) -> list:
    """多查询召回（查询改写用）：每个子查询各召回候选，按 chunk_id 聚合取最高分，再走阈值/去重/精排取 top_k。"""
    from core.vectorstore import query_collection
    top_k = top_k or config.TOP_K
    min_score = config.MIN_SCORE if min_score is None else min_score
    candidate_k = config.RERANK_CANDIDATES if use_rerank else top_k

    # 各子查询独立召回
    all_chunks = []
    for q in queries:
        qvec = embed_query(config.QUERY_PREFIX + q)
        raw = query_collection(qvec, top_k=candidate_k)
        for item in raw:
            sim = round(1.0 - item["distance"], 4)
            all_chunks.append(Chunk(
                chunk_id=item["id"],
                text=item["text"],
                source=item["metadata"].get("source", ""),
                heading=item["metadata"].get("heading", ""),
                score=sim,
            ))

    # 按 chunk_id 聚合取最高分
    chunk_dict = {}
    for c in all_chunks:
        if c.id not in chunk_dict or c.score > chunk_dict[c.id].score:
            chunk_dict[c.id] = c
    passed = list(chunk_dict.values())

    # 阈值闸门
    passed = [c for c in passed if c.score >= min_score]

    # 内容去重（同文本保留高分）
    dedup = {}
    for c in passed:
        key = "".join((c.text or "").split()) or str(c.id)
        if key not in dedup or c.score > dedup[key].score:
            dedup[key] = c
    passed = list(dedup.values())

    if use_rerank:
        from core.rerank import rerank
        # rerank 已按交叉编码器分降序取 top_k，直出精排顺序，不再按 bi-encoder 重排
        return rerank(queries[0], passed, top_k)

    passed.sort(key=lambda c: c.score, reverse=True)
    return passed[:top_k]