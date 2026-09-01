# -*- coding: utf-8 -*-
"""RAG 客服问答平台 — Streamlit 入口。"""
import streamlit as st

from core import config
from ui import chat, eval as eval_page, style

config.ensure_dirs()
style.set_page()

# 启动自检：模型名 / API key（结果缓存，避免每次交互都调 API）
if not config.DEEPSEEK_API_KEY:
    st.sidebar.warning("未配置 DEEPSEEK_API_KEY，请在 01_mvp/.env 中填写后再问答/评测。")
else:
    if "llm_ok" not in st.session_state:
        st.session_state["llm_ok"] = config.verify_llm()
    if st.session_state["llm_ok"] is False:
        st.sidebar.error("当前 LLM_MODEL 可能不可用，请在 .env 改为 deepseek-chat 后重启。")

with st.sidebar:
    st.markdown("## 💬 RAG 客服问答")
    page = st.radio("导航", ["智能问答", "评测与优化"], label_visibility="collapsed")

if page == "智能问答":
    chat.render_chat()
else:
    eval_page.render_eval()