# -*- coding: utf-8 -*-
"""独立下载 bge-reranker-v2-m3（不 import core.config，避免被 .env 强制离线）。
只下载到 HF 缓存，不实例化模型。"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"  # hf-mirror 不支持 Xet 后端(401)，强制经典 LFS 下载

from huggingface_hub import snapshot_download

p = snapshot_download("BAAI/bge-reranker-v2-m3")
print("DOWNLOAD_OK:", p)
