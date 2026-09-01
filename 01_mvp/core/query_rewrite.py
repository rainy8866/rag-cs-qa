# -*- coding: utf-8 -*-
"""查询改写：把复杂问题拆成 1~3 个子查询，多路召回提升 recall（PRD §7 优化）。

独立模块、不 import gold_set_enhance（避免拉入评估集生成的重量级依赖）。
统一走 config.REWRITE_MODEL（非推理模型，避免 reasoning 占满 max_tokens）。
"""
import json
import re

from core import config

REWRITE_PROMPT = """你是查询改写专家。把用户问题拆分成 1~3 个更具体的子查询，帮助检索系统找到更相关的资料。
要求：
1. 子查询必须是完整的问题，保留原问题核心意图。
2. 覆盖原问题的不同方面（如适用：条件/实体/规则/例子）。
3. 原问题本身必须作为第一个子查询。
4. 只返回 JSON 数组，如：["原问题", "子查询1", "子查询2"]
5. 不要解释，不要多余内容，确保 JSON 可解析。

用户问题：{question}"""


def _gen(messages, temperature=0.3, max_tokens=500, timeout=60):
    from openai import OpenAI
    client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.LLM_BASE_URL)
    resp = client.chat.completions.create(
        model=config.REWRITE_MODEL, messages=messages,
        temperature=temperature, max_tokens=max_tokens, timeout=timeout)
    return resp.choices[0].message.content


def _parse(text):
    """从 LLM 输出里抠出 JSON 数组；失败返回 None。"""
    text = (text or "").strip()
    m = re.search(r"\[.*\]", text, re.S)
    if m:
        text = m.group(0)
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(x) for x in data if str(x).strip()]
    except Exception:
        pass
    return None


def rewrite(question: str) -> list:
    """返回子查询列表（原问题在首位），长度 ≤ REWRITE_MAX_QUERIES。失败降级为 [原问题]。"""
    if not question or not config.REWRITE_ENABLED:
        return [question]
    try:
        prompt = REWRITE_PROMPT.format(question=question)
        resp = _gen([{"role": "user", "content": prompt}])
        queries = _parse(resp)
        if not queries:
            return [question]
        if queries[0] != question:
            queries.insert(0, question)
        return queries[:config.REWRITE_MAX_QUERIES]
    except Exception:
        return [question]
