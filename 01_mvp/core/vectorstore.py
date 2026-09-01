# -*- coding: utf-8 -*-
"""ChromaDB 封装。注意统一用 cosine，并做 distance->similarity 换算。"""
import chromadb
from core import config

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(config.VECTOR_DIR))
    return _client


def get_collection():
    return _get_client().get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection():
    client = _get_client()
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    return get_collection()


def collection_count() -> int:
    col = get_collection()
    try:
        return col.count()
    except Exception:
        return 0


def query_collection(query_embedding, top_k):
    """返回 raw 记录列表，每条含 distance。调用方负责 score=1-distance 换算。"""
    col = get_collection()
    res = col.query(query_embeddings=[query_embedding], n_results=top_k,
                    include=["documents", "metadatas", "distances"])
    items = []
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for i in range(len(ids)):
        items.append({
            "id": ids[i],
            "text": docs[i],
            "metadata": metas[i] or {},
            "distance": dists[i],
        })
    return items