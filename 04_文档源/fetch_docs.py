# -*- coding: utf-8 -*-
"""下载 LangChain 官方文档作为示例知识库(示例用途，非整站爬取)。
用法: 在 04_文档源 目录下运行  python fetch_docs.py
"""
import re
import sys
import time
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify as md2md

BASE = "https://python.langchain.com"
OUT = Path(__file__).resolve().parent
MAX_PAGES = 30


def get_sitemap_urls() -> list:
    r = httpx.get(f"{BASE}/sitemap.xml", timeout=20)
    r.raise_for_status()
    root = BeautifulSoup(r.text, "xml")
    urls = []
    for loc in root.find_all("loc"):
        u = loc.get_text(strip=True)
        if "/docs/" in u:
            urls.append(u)
    # 去重并取前 MAX_PAGES
    seen, out = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
        if len(out) >= MAX_PAGES:
            break
    return out


def page_to_md(url: str) -> str:
    r = httpx.get(url, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    main = soup.find("main") or soup.find("article") or soup.body
    return md2md(str(main), heading_style="ATX", strip=["script", "style", "nav"])


def main():
    print("抓取 sitemap...")
    urls = get_sitemap_urls()
    print(f"共 {len(urls)} 页待下载")
    ok = 0
    for i, u in enumerate(urls):
        try:
            mk = page_to_md(u)
            # 文件用 URL 里末段取名
            name = re.sub(r"[^A-Za-z0-9_-]", "_", u.rstrip("/").split("/")[-1]) or f"doc_{i}"
            (OUT / f"doc_{name}.md").write_text(f"# {u}\n\n{mk}", encoding="utf-8")
            ok += 1
            time.sleep(0.3)
        except Exception as e:
            print(f"      跳过 {u}: {e} | 使用 offline 兜底")
    print(f"完成: {ok}/{len(urls)} 下载成功。若为 0，请改用离线 offline_*.md。")


if __name__ == "__main__":
    main()