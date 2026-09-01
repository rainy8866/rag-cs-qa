# -*- coding: utf-8 -*-
"""LLM 封装：调 DeepSeek(OpenAI 兼容接口)。"""
from core import config


def build_client():
    if not config.DEEPSEEK_API_KEY:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY，请在 01_mvp/.env 中填写。")
    from openai import OpenAI
    return OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.LLM_BASE_URL)


def chat(messages, temperature=0.0, max_tokens=None, timeout=40):
    client = build_client()
    kwargs = {"temperature": temperature}
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    try:
        resp = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=messages,
            **kwargs,
            timeout=timeout,
        )
    except Exception as e:
        raise RuntimeError(f"调用大模型失败：{e}") from e
    return resp.choices[0].message.content