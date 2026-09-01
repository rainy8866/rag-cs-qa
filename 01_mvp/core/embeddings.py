# -*- coding: utf-8 -*-
"""Embedding：本地 bge-m3。缓存单例避免重复加载。"""
from core import config

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(config.EMBED_MODEL)
    return _embedder


def embed_query(text: str):
    return get_embedder().encode([text], normalize_embeddings=True)[0]