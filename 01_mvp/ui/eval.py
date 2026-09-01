# -*- coding: utf-8 -*-
"""「评测与优化」页：评估集生成 + 三大指标检测 + 标定阈值 + 一键优化 + 对比图。"""
import json

import pandas as pd
import streamlit as st

from core import config, ingest, report as report_mod
from core import badcase as badcase_mod
from core import convert_docs as convert_docs_mod
from core import gold_set as gold_set_mod
from core import metrics as metrics_mod
from core import optimize as opt_mod
from ui import style


def _have_gold() -> bool:
    return config.gold_set_path().exists() and config.gold_set_path().stat().st_size > 2


def render_eval():
    style.section_card("评测与优化", "📊")
    st.caption("用同一份评估集量化效果并迭代。按钮独立触发，结果实时展示。")

    use_rerank = opt_mod.read_summary().get("use_rerank", False)

    # ---------- A. 评估集区 ----------
    style.section_card("评估集", "🔄")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🔄 生成评估集", use_container_width=True, type="primary"):
            with st.spinner("正在让大模型根据原文生成评估集..."):
                try:
                    r = gold_set_mod.generate_gold_set()
                    if r.get("ok"):
                        st.success(f"已生成 {r['n']} 条，版本 {r['version']}")
                    else:
                        st.error(r.get("reason", "生成失败"))
                except Exception as e:
                    st.error(f"生成失败：{e}")
    with c2:
        if st.button("🔄 转码文档源", use_container_width=True,
                     help="把 04_文档源 下的 docx/pdf/xlsx/pptx 转成同名 .md"):
            with st.spinner("正在转码办公文档为 Markdown..."):
                r = convert_docs_mod.convert_docs(config.DOC_DIR)
            if r["errors"]:
                st.error(
                    f"转换 {r['converted']} 个，跳过同名 {r['skipped_existing']} 个，"
                    f"失败 {r['failed']} 个"
                )
                for e in r["errors"][:5]:
                    st.caption(f"- {e}")
            else:
                st.success(
                    f"转换 {r['converted']} 个，跳过同名 {r['skipped_existing']} 个，"
                    f"失败 {r['failed']} 个"
                )
            st.caption("💡 转换出的 .md 需点击「🗂️ 重建知识库」才会进入向量库")
    with c3:
        if st.button("🗂️ 重建知识库", use_container_width=True,
                     help="更换文档源/切块方式后需重建，并应新建版本快照"):
            with st.spinner("重建知识库..."):
                prog = st.progress(0.0)
                def _cb(done, total):
                    prog.progress(min(1.0, done / total if total else 1.0))
                r = ingest.build_index(progress=_cb)
            if r.get("ok"):
                st.success(f"重建完成，共 {r['n_chunks']} 块")
            else:
                st.error(r.get("reason", "重建失败"))

    # ---------- B. 指标检测区 ----------
    style.section_card("指标检测", "🎯")
    gold = gold_set_mod.load_gold_set()
    if not gold["items"]:
        st.info("评估集为空，请先在上方「生成评估集」。")
        if _have_gold():
            # 可能文件损坏或为空
            gold_set_mod.config.gold_set_path().unlink(missing_ok=True)
        return

    if "metrics" not in st.session_state:
        st.session_state.metrics = None

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("🎯 检测召回率", use_container_width=True):
            with st.spinner("计算召回率..."):
                r = metrics_mod.recall_at_k(gold["items"], use_rerank=use_rerank)
                st.session_state.metrics = r
            st.metric(f"Recall@{config.TOP_K}", f"{r['recall']:.2%}",
                      f"{r['hit']}/{r['total']} 命中")
            if r["missed"]:
                st.markdown("**未命中明细**")
                st.dataframe(pd.DataFrame(r["missed"]),
                             use_container_width=True)
    with b2:
        if st.button("🎯 检测引用准确率", use_container_width=True):
            with st.spinner("计算引用准确率..."):
                r = metrics_mod.citation_accuracy(gold["items"],
                                                  use_rerank=use_rerank)
                st.session_state.metrics = r
            st.metric("引用准确率", f"{r['accuracy']:.2%}",
                      f"{r['correct']}/{r['total']}")
            st.dataframe(pd.DataFrame(r["details"]).drop(columns=["answer"]),
                         use_container_width=True)
    with b3:
        if st.button("🧪 检测坏例", use_container_width=True):
            with st.spinner("归类坏例..."):
                _rec = metrics_mod.recall_at_k(gold["items"], use_rerank=use_rerank)
                _cite = metrics_mod.citation_accuracy(gold["items"],
                                                      use_rerank=use_rerank)
                bc = badcase_mod.classify_badcases(gold["items"], _rec, _cite,
                                                   use_rerank=use_rerank)
            counts = bc["counts"]
            st.bar_chart(pd.DataFrame({"坏例数": list(counts.values())},
                                      index=list(counts.keys())))
            for cat, lst in bc["items"].items():
                if lst:
                    with st.expander(f"{cat}（{len(lst)}）"):
                        for it in lst[:5]:
                            st.markdown(f"**Q:** {it['question']}")
                            st.caption(it.get("required_snippet", ""))

    # 标定阈值
    style.section_card("阈值标定", "⚖️")
    if st.button("⚖️ 标定相似度阈值", use_container_width=True):
        with st.spinner("分析命中/未命中分数分布..."):
            cal = opt_mod.calibrate_threshold(gold["items"])
        st.session_state.calib = cal
        if cal["hit_scores"]:
            st.metric("推荐 MIN_SCORE", cal["recommended"],
                      f"当前 {cal['current']}")
            st.write("**命中分数**", pd.Series(cal["hit_scores"]).describe())
            st.write("**未命中分数**", pd.Series(cal["miss_scores"]).describe())
            if cal["hit_scores"]:
                st.write("命中分数分布")
                st.bar_chart(pd.Series(cal["hit_scores"], name="命中").value_counts().sort_index())
            if cal["miss_scores"]:
                st.write("未命中分数分布")
                st.bar_chart(pd.Series(cal["miss_scores"], name="未命中").value_counts().sort_index())
    if "calib" in st.session_state and st.session_state["calib"]:
        if st.button("✅ 应用到推荐阈值", use_container_width=True):
            cal = st.session_state.get("calib", {})
            if cal:
                opt_mod.apply_threshold(cal["recommended"])
                st.success(f"已将 MIN_SCORE 设为 {cal['recommended']}")

    # ---------- C. 二次优化区 ----------
    style.section_card("二次优化", "🚀")
    if st.button("🚀 一键优化", use_container_width=True, type="primary"):
        with st.spinner("搜索参数组合（TOP_K/阈值/Rerank），用同一份评估集评估..."):
            r = opt_mod.optimize_auto(gold["items"])
        if r.get("ok"):
            st.session_state.opt_result = r
            best = r["best"]
            st.success(f"最优: TOP_K={best['top_k']} MIN_SCORE={best['min_score']} "
                       f"Rerank={'开' if best['use_rerank'] else '关'}  Recall={best['recall']:.2%}")
        else:
            st.error(r.get("reason", "优化失败"))

    base = st.session_state.get("base_metrics")
    optres = st.session_state.get("opt_result")
    if optres and optres.get("ok") and base:
        best = optres["best"]
        df = pd.DataFrame({
            "Recall": [base.get("recall", 0), best["recall"]],
            "坏例数": [base.get("bad", 0), best.get("bad", 0)],
        }, index=["优化前", "优化后"])
        st.bar_chart(df)

    # 导出报告
    style.section_card("导出", "📄")
    if st.button("📄 导出指标报告", use_container_width=True):
        # 组合完整指标
        with st.spinner("生成指标报告..."):
            _rec = metrics_mod.recall_at_k(gold["items"], use_rerank=use_rerank)
            _cite = metrics_mod.citation_accuracy(gold["items"], use_rerank=use_rerank)
            _bc = badcase_mod.classify_badcases(gold["items"], _rec, _cite,
                                                use_rerank=use_rerank)
            p = report_mod.write_metrics_report(
                {"recall": _rec, "citation": _cite, "badcase": _bc},
                version=gold["version"])
        st.success(f"已导出：{p}")