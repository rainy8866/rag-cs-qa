# -*- coding: utf-8 -*-
"""「智能问答」页：聊天气泡 + 引用折叠。"""
import streamlit as st

from core import config, ingest, qa as qa_mod
from core.vectorstore import collection_count
from ui import style


def render_chat():

    if collection_count() == 0:
        style.section_card("知识库为空", "🗂️")
        st.warning("还没有建立知识库。请先在左侧把文档源建好，或点击下方按钮重建。")
        if st.button("🛠️ 重建知识库", use_container_width=True):
            with st.spinner("正在切块并向量化入库..."):
                r = ingest.build_index()
            if r.get("ok"):
                st.success(f"知识库已建立，共 {r['n_chunks']} 个文本块。")
            else:
                st.error(r.get("reason", "建档失败"))
            st.rerun()
        return

    from core.optimize import read_summary
    default_rr = read_summary().get("use_rerank", False)
    use_rerank = st.sidebar.toggle("启用 Rerank（二次优化）",
                                   value=default_rr)

    style.section_card("智能问答", "💬")
    st.caption("答案依据知识库检索生成，带引用出处，点击可查看原文。")

    if "msgs" not in st.session_state:
        st.session_state.msgs = [{
            "role": "assistant",
            "content": "你好！我是你的知识库客服助手。有什么想问的？",
        }]

    for m in st.session_state.msgs:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            for c in m.get("citations", []):
                with st.expander(f"📎 参考资料 · {c.get('source','')}"):
                    st.markdown(c.get("text", ""))

    q = st.chat_input("输入你的问题…")
    if q:
        st.session_state.msgs.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("检索并生成中..."):
                try:
                    res = qa_mod.answer(q, use_rerank=use_rerank)
                    answer = res["answer"]
                    citations = res.get("citations", [])
                except Exception as e:
                    answer = f"⚠️ 出错了：{e}"
                    citations = []
                st.markdown(answer)
                for c in citations:
                    with st.expander(
                            f"📎 参考资料 · {c.get('source','')}"):
                        st.markdown(c.get("text", ""))
        st.session_state.msgs.append(
            {"role": "assistant", "content": answer, "citations": citations})
        st.rerun()