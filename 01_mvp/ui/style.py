# -*- coding: utf-8 -*-
"""页面美化：统一配色、卡片、按钮、气泡。"""
import streamlit as st

BRAND = "#4F6DF5"
BRAND2 = "#7A5BF5"
BRAND_SOFT = "#EAF0FE"
BG = "#F7F8FC"
CARD = "#FFFFFF"
TEXT = "#1B2233"
MUTED = "#6B7280"


def apply_style():
    st.markdown(f"""
    <style>
    :root {{
      --brand:{BRAND}; --brand-soft:{BRAND_SOFT}; --bg:{BG};
      --card:{CARD}; --text:{TEXT}; --muted:{MUTED};
    }}
    .stApp {{ background:linear-gradient(160deg,#F7F8FC 0%,#EEF1FB 100%); }}
    .block-container {{ padding-top:1.6rem; }}
    h1,h2,h3 {{ letter-spacing:.2px; }}
    div[data-testid="stSidebar"] {{
      background:{CARD};
      border-right:1px solid #E7EAF3;
    }}
    .metric-badge {{
      background:linear-gradient(135deg,{BRAND},#7A5BF5);
      color:#fff; font-weight:600;
      padding:14px 18px; border-radius:16px;
    }}
    div[data-testid="stMetric"] {{
      background:{CARD}; border:1px solid #E9ECF5; border-radius:14px;
      padding:14px 16px;
      box-shadow:0 6px 18px rgba(79,109,245,.06);
    }}
    div[data-testid="stMetricLabel"] {{ color:{MUTED}; }}
    div[data-testid="stMetricValue"] {{ color:{BRAND}; font-weight:700; }}
    .stButton>button, .stDownloadButton>button {{
      background:{CARD}; color:{TEXT};
      border:1px solid #DFE4F2; border-radius:10px;
      transition:all .15s ease;
    }}
    .stButton>button:hover, .stDownloadButton>button:hover {{
      border-color:{BRAND}; color:{BRAND};
      transform:translateY(-1px);
      box-shadow:0 6px 18px rgba(79,109,245,.15);
    }}
    section[data-testid="stSidebarUserContent"] .stButton>button {{
      background:linear-gradient(135deg,{BRAND},{BRAND2});
      color:#fff; border:none;
    }}
    </style>
    """, unsafe_allow_html=True)


def set_page():
    st.set_page_config(page_title="RAG 客服问答", page_icon="💬", layout="wide")
    apply_style()


def section_card(title: str, icon: str = "📌"):
    st.markdown(
        f"<div style='background:{CARD};border:1px solid #E9ECF5;border-radius:16px;"
        f"padding:14px 18px;margin:6px 0 12px'>"
        f"<span style='font-weight:700;color:{TEXT};font-size:15px'>{icon} {title}</span></div>",
        unsafe_allow_html=True,
    )
    # 让 st.container 顶掉上面的方形 div 不影响布局体验
    st.markdown("<br>", unsafe_allow_html=False)