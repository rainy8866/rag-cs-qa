# -*- coding: utf-8 -*-
"""评估集增强：抽检现有评估集 + 补充真实口语化问题 + 合并更新为新版本（沿用版本快照机制）。

新增模块，不改动 gold_set.generate_gold_set / load_gold_set 的既有行为；
复用 gold_set 的 _read_docs_text / _save_snapshot / _strip_json_fence / load_gold_set。
"""
import json
from datetime import datetime

from core import config
from core.gold_set import (_read_docs_text, _save_snapshot, _strip_json_fence,
                           _doc_fingerprint, load_gold_set)

# 生成类任务统一走非推理模型（deepseek-chat）：
# deepseek-v4-flash 是推理模型，复杂生成任务会把 max_tokens 全耗在 reasoning_content 上、
# 正式 content 为空（实测 2026-08-29），导致难例/问法/口语化批量空批、欠产。
# deepseek-chat 不产生隐藏推理，直接输出 JSON，产出稳定且更省 token。
GEN_MODEL = "deepseek-chat"


def _gen_chat(messages, temperature=0.7, max_tokens=2000, timeout=150):
    from openai import OpenAI
    client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.LLM_BASE_URL)
    resp = client.chat.completions.create(
        model=GEN_MODEL, messages=messages,
        temperature=temperature, max_tokens=max_tokens, timeout=timeout)
    return resp.choices[0].message.content

# 问法 5 类（用于多样性统计，按列表顺序做首个匹配）
QUESTION_TYPES = [
    ("直接询问", ["什么是", "是什么", "怎么", "如何", "多少", "哪些", "哪个", "有哪", "有没有"]),
    ("请求确认", ["是否", "能不能", "可不可以", "可以吗", "对吗", "是不是"]),
    ("寻求解释", ["请解释", "为什么", "为啥", "解释一下"]),
    ("假设性提问", ["如果", "假如", "万一", "假设", "若"]),
    ("求例子", ["举例", "例子", "比如", "举个例子", "例如"]),
]

COLLOQUIAL_PROMPT = """你是客服问答数据标注员。请根据下面的【原文】生成 {n} 条<b>真实口语化</b>的客服问答样例，模拟真实用户打字/说话时会问的短句，允许带语气词和模糊表述。

【原文】
{docs}

【口语化风格示例（仅作风格参考，不要照抄）】
- "我买的东西坏了咋办？"
- "你们退货麻烦不？"
- "这运费谁出啊？"
- "会员能免运费不？"
- "优惠券退款了还能用吗？"

【生成要求】
1. 每条样例包含三个字段：question / expected_answer / required_snippet。
2. question 必须口语化、短句、贴近真实用户，可带语气词（啊/呀/咋/嘛/不/吗/吧）。
3. required_snippet 必须<b>从原文逐字复制</b>，不得改写。
4. expected_answer 基于原文；若原文不足以完整回答，写明"根据给定信息无法完全确认"。
5. 不要与示例重复，也不要两两重复。

只输出一个 JSON 数组，不要输出任何其他文字。格式：
[{{"question": "...", "expected_answer": "...", "required_snippet": "..."}}, ...]
"""

FIX_PROMPT = """你是客服问答数据修复员。下面评估条目里的 required_snippet 必须逐字来自【原文】。
请根据 question 在【原文】中定位对应知识点：
- 若能定位：重写 required_snippet（<b>必须从原文逐字复制</b>），并在需要时修正 expected_answer（也须基于原文）。
- 若原文中确实没有该知识点：输出 null。

【原文】
{docs}

【待修复条目】
question: {q}
expected_answer: {ans}
required_snippet: {snip}

只输出一个 JSON 对象（找不到则输出 null），不要输出任何其他文字：
{{"question": "...", "expected_answer": "...", "required_snippet": "..."}}
"""


# 难例/边界例生成（PRD §4.2；3 字段版：5 字段 JSON 会让 reasoning 模型把 max_tokens
# 耗尽在推理上导致正式 content 截断，故 tag/difficulty 改为本地规则回填）
HARD_CASE_PROMPT = """你是客服问答数据工程师。请根据下面的【原文】生成 {n} 条<b>高难度/边界</b>客服问答样例，刻意制造真实检索难点。

【原文】
{docs}

【难例类型】（每条必须且只属于以下某一类，自然融入问题，无需输出类型字段）
1. 泛查询：笼统问一个章节/主题（如「你们的退货政策是啥」），答案分散在多个块。
2. 易混淆：两个相近政策放一起（如「7 天无理由退货」vs「质量问题换货」）。
3. 假设性：偏离标准条件（如「我超过 7 天才申请，还能退吗」）。
4. 否定/排除式：问「哪种情况不能退款」「哪些商品不支持」。
5. 长尾实体：涉及 SKU/政策编号/专有名词等精确词。
6. 跨章节组合：答案分散在两个标题下。
7. 边界条件：满减/优惠门槛恰好等于或差一点（如「满 300 减 30，差 1 元算吗」）。

【生成要求】
1. 每条含三个字段：question / expected_answer / required_snippet。
2. required_snippet 必须<b>从原文逐字复制</b>，不得改写。
3. expected_answer 基于原文；若原文不足以完整回答，写明"根据给定信息无法完全确认"。
4. 不要与示例重复，也不要两两重复。

只输出一个 JSON 数组，不要输出任何其他文字。格式：
[{{"question": "...", "expected_answer": "...", "required_snippet": "..."}}, ...]
"""

# 按问法类型补齐（PRD §4.4）
BY_TYPE_PROMPT = """你是客服问答数据工程师。请根据下面的【原文】生成 {n} 条客服问答样例，<b>优先使用「{focus}」问法</b>，帮助补齐该问法类型的覆盖。

【原文】
{docs}

【问法类型说明】
- 直接询问：什么是…？怎么…？如何…？多少…？
- 请求确认：是否可以说…？能不能…？可以…吗？
- 寻求解释：为什么…？请解释…
- 假设性提问：如果…会怎样？假如…？
- 求例子：能否举个例子…？比如…？

【生成要求】
1. 每条样例包含三个字段：question / expected_answer / required_snippet。
2. 每条必须能<b>从原文逐字复制</b> required_snippet。
3. expected_answer 基于原文；若原文不足以完整回答，写明"根据给定信息无法完全确认"。
4. 不要与示例重复，也不要两两重复。

只输出一个 JSON 数组，不要输出任何其他文字。格式：
[{{"question": "...", "expected_answer": "...", "required_snippet": "..."}}, ...]
"""


def _read_doc_texts():
    """返回 {文件名: 纯文本}，用于 snippet 子串校验（不含 === 头）。"""
    return {f.name: f.read_text(encoding="utf-8", errors="ignore")
            for f in sorted(config.DOC_DIR.glob("*.md"))}


def _classify_question(q):
    for name, kws in QUESTION_TYPES:
        if any(k in q for k in kws):
            return name
    return "其它"


def _snippet_valid(snippet, all_text):
    return bool(snippet) and snippet in all_text


def _is_answer_weak(ans):
    return bool(ans) and ("无法确认" in ans or "无法完全确认" in ans)


def audit_items(items, all_text):
    """抽检：去重 + snippet 有效性 + 答案完整性 + 问法统计。
    返回 dict：{deduped, removed_dup, flags, qtype, rows}"""
    # 1) 去重：question 完全相同只保留第一条
    seen, deduped, removed_dup = set(), [], []
    for it in items:
        q = (it.get("question") or "").strip()
        if not q or q in seen:
            removed_dup.append(it)
        else:
            seen.add(q)
            deduped.append(it)
    # 2) 逐条检查 + 问法统计
    flags, qtype, rows = {}, {}, []
    for i, it in enumerate(deduped):
        reasons = []
        if not _snippet_valid(it.get("required_snippet", ""), all_text):
            reasons.append("snippet_invalid")
        if _is_answer_weak(it.get("expected_answer", "")):
            reasons.append("answer_weak")
        if reasons:
            flags[i] = reasons
        typ = _classify_question(it.get("question", ""))
        qtype[typ] = qtype.get(typ, 0) + 1
        rows.append({"index": i, "question": it.get("question", ""), "reasons": reasons})
    return {"deduped": deduped, "removed_dup": removed_dup,
            "flags": flags, "qtype": qtype, "rows": rows}


def _fix_items(deduped, flags, all_text):
    """对带问题条目尝试 LLM 修复。返回 (fixed: {index: new_item}, failed: [index])。"""
    fixed, failed = {}, []
    for i, _reasons in flags.items():
        it = deduped[i]
        prompt = FIX_PROMPT.format(docs=all_text[:24000],
                                   q=it.get("question", ""),
                                   ans=it.get("expected_answer", ""),
                                   snip=it.get("required_snippet", ""))
        try:
            resp = _gen_chat([{"role": "user", "content": prompt}],
                             temperature=0.1, max_tokens=1200)
            cleaned = _strip_json_fence(resp).strip()
            if not cleaned or cleaned.lower() == "null":
                failed.append(i)
                continue
            obj = json.loads(cleaned)
            new_snip = obj.get("required_snippet", "")
            if not _snippet_valid(new_snip, all_text):
                failed.append(i)
                continue
            fixed[i] = {"question": obj.get("question") or it.get("question", ""),
                        "expected_answer": obj.get("expected_answer") or it.get("expected_answer", ""),
                        "required_snippet": new_snip}
        except Exception:
            failed.append(i)
    return fixed, failed


def _generate_colloquial(n, all_text, existing_q):
    """分批用 LLM 生成口语化问题，过滤掉 snippet 无效 / 与现有重复的条目。"""
    BATCH = 4
    added, seen = [], set(existing_q)
    batches = (n + BATCH - 1) // BATCH
    for i in range(batches):
        want = min(BATCH, n - i * BATCH)
        prompt = (f"（第 {i + 1}/{batches} 批，请生成与其它批不同的口语化问题，避免重复）\n"
                  + COLLOQUIAL_PROMPT.format(n=want, docs=all_text[:24000]))
        try:
            resp = _gen_chat([{"role": "user", "content": prompt}],
                             temperature=0.7, max_tokens=2000)
            batch = json.loads(_strip_json_fence(resp))
            if not isinstance(batch, list):
                continue
            for it in batch:
                q = (it.get("question") or "").strip()
                if not q or q in seen:
                    continue
                if not _snippet_valid(it.get("required_snippet", ""), all_text):
                    continue
                seen.add(q)
                added.append(it)
        except Exception:
            continue
    return added


def _generate_by_type(n, items, all_text):
    """按问法类型补齐：统计现有条目的问法分布，对覆盖最少的类型定向生成补足（PRD §4.4）。"""
    dist = {}
    for it in items:
        t = _classify_question(it.get("question", ""))
        dist[t] = dist.get(t, 0) + 1
    # 按现有分布从少到多排序，优先补最缺的类型
    focus_order = sorted(dist.items(), key=lambda x: x[1])
    focus_types = [t for t, _ in focus_order]
    focus_types += [t for t, _ in QUESTION_TYPES if t not in dist]
    focus = focus_types[0] if focus_types else "直接询问"
    BATCH = 4
    added, seen = [], set(it["question"] for it in items)
    batches = (n + BATCH - 1) // BATCH
    for i in range(batches):
        want = min(BATCH, n - i * BATCH)
        prompt = (f"（第 {i + 1}/{batches} 批，请生成与其它批不同的问题，避免重复）\n"
                  + BY_TYPE_PROMPT.format(n=want, focus=focus, docs=all_text[:24000]))
        try:
            resp = _gen_chat([{"role": "user", "content": prompt}],
                             temperature=0.5, max_tokens=2000)
            batch = json.loads(_strip_json_fence(resp))
            if not isinstance(batch, list):
                continue
            for it in batch:
                q = (it.get("question") or "").strip()
                if not q or q in seen:
                    continue
                if not _snippet_valid(it.get("required_snippet", ""), all_text):
                    continue
                seen.add(q)
                added.append(it)
        except Exception:
            continue
    return added


def _generate_hard_cases(n, all_text, existing_q):
    """分批用 LLM 生成难例/边界例（覆盖 7 类难例），过滤 snippet 无效 / 重复条目（PRD §4.2）。

    难例 prompt 用 3 字段输出（5 字段会让 reasoning 模型截断 content），
    tag 由 _backfill_tags 回填，difficulty 按边界词规则本地判定。
    """
    BATCH = 5
    added, seen = [], set(existing_q)
    batches = (n + BATCH - 1) // BATCH
    for i in range(batches):
        want = min(BATCH, n - i * BATCH)
        prompt = (f"（第 {i + 1}/{batches} 批，请生成与其它批不同的难例，覆盖不同难例类型，避免重复）\n"
                  + HARD_CASE_PROMPT.format(n=want, docs=all_text[:24000]))
        try:
            resp = _gen_chat([{"role": "user", "content": prompt}],
                             temperature=0.7, max_tokens=3000)
            batch = json.loads(_strip_json_fence(resp))
            if not isinstance(batch, list):
                continue
            for it in batch:
                q = (it.get("question") or "").strip()
                if not q or q in seen:
                    continue
                if not _snippet_valid(it.get("required_snippet", ""), all_text):
                    continue
                seen.add(q)
                it["difficulty"] = _boundary_flag(q)
                added.append(it)
        except Exception:
            continue
    return added


def _boundary_flag(q):
    """难例里含边界/临界关键词的问题标记为 boundary，否则 hard（PRD §4.3）。"""
    for kw in ("满", "刚好", "恰好", "差", "等于", "门槛", "临界", "满减", "正好"):
        if kw in q:
            return "boundary"
    return "hard"


# 标签池（PRD §4.3）
TAG_POOL = ["退货", "退款", "物流", "发货", "支付", "发票", "会员",
            "优惠券", "账号", "安全", "售后", "换货", "跨境", "其他"]


def _backfill_tags(items):
    """对缺 tag/difficulty 的旧条目做规则回填：按 question 关键词命中标签池关键词即回填（PRD §4.3）。"""
    for it in items:
        if not it.get("tag"):
            q = it.get("question", "")
            tag = "其他"
            for t in TAG_POOL:
                if t in q:
                    tag = t
                    break
            it["tag"] = tag
        if not it.get("difficulty"):
            it["difficulty"] = "normal"
    return items


def _semantic_dedup(items, threshold=None):
    """语义去重：question 两两余弦相似度 ≥ threshold 视为重复（PRD §5.2）。

    保留优先级：expected_answer 更长 → difficulty 为 hard/boundary 优先 → 靠前保留。
    贪婪消解：按相似度从高到低处理，移除后的条目不参与后续配对。
    返回 (deduped, removed_pairs, before, after)。
    """
    from core.embeddings import embed_query
    threshold = threshold if threshold is not None else config.SEMANTIC_DEDUP_THRESHOLD
    if len(items) < 2:
        return items, [], len(items), len(items)

    vecs = [embed_query(it.get("question", "")) for it in items]

    def _sim(a, b):
        return sum(x * y for x, y in zip(a, b))  # normalize 后点积即余弦

    # 构建重复对（相似度从高到低）
    pairs = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            s = _sim(vecs[i], vecs[j])
            if s >= threshold:
                pairs.append((s, i, j))
    pairs.sort(key=lambda x: -x[0])

    removed = set()
    removed_pairs = []
    for s, i, j in pairs:
        if i in removed or j in removed:
            continue  # 已移除，跳过
        kept_i, removed_i = _pick_keep(items, i, j)
        removed.add(removed_i)
        removed_pairs.append({"q1": items[i]["question"], "q2": items[j]["question"],
                              "sim": float(round(s, 4)), "kept_index": kept_i,
                              "removed_index": removed_i,
                              "kept_question": items[kept_i]["question"],
                              "removed_question": items[removed_i]["question"]})
    deduped = [it for idx, it in enumerate(items) if idx not in removed]
    return deduped, removed_pairs, len(items), len(deduped)


def _pick_keep(items, i, j):
    """按信息量优先级决定保留哪条，返回 (kept_index, removed_index)。"""
    a, b = items[i], items[j]

    def _score(it):
        sc = 0
        sc += min(len(it.get("expected_answer", "")), 500) / 50.0   # 答案越长分越高
        if it.get("difficulty") in ("hard", "boundary"):
            sc += 10
        return sc
    if _score(a) != _score(b):
        return (i, j) if _score(a) > _score(b) else (j, i)
    return (i, j)  # 分数相同，靠前保留


def _write_audit_report(audit, fixed, failed, added, version,
                        tag_dist=None, dedup_pairs=None, added_by_type=None,
                        added_hard=None):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = config.REPORT_DIR / f"goldset_audit_{ts}.md"
    lines = [
        f"# 评估集抽检报告（{version}）",
        "",
        f"- 生成时间：{datetime.now().isoformat()}",
        f"- 原条数（去重前）：{len(audit['deduped']) + len(audit['removed_dup'])}",
        f"- 去重后条数：{len(audit['deduped'])}",
        f"- 剔除（dup）：{len(audit['removed_dup'])}",
        f"- 待修（snippet_invalid / answer_weak）：{len(audit['flags'])}，已修复 {len(fixed)}，修复失败 {len(failed)}",
        f"- 新增口语化：{len(added)}",
    ]
    if added_by_type is not None:
        lines.append(f"- 按问法类型补齐：{len(added_by_type)}")
    if added_hard is not None:
        lines.append(f"- 新增难例/边界例：{len(added_hard)}")
    lines += [
        "",
        "## 抽检明细",
        "",
        "| 序号 | question | 检查结果 | 处置 |",
        "|---|---|---|---|",
    ]
    for r in audit["rows"]:
        res = "OK" if not r["reasons"] else ";".join(r["reasons"])
        if r["index"] in fixed:
            disp = "已修复"
        elif r["index"] in audit["flags"]:
            disp = "保留(修复失败)" if r["index"] in failed else "保留"
        else:
            disp = "保留"
        q = (r["question"] or "").replace("|", "\\|")
        lines.append(f"| {r['index'] + 1} | {q} | {res} | {disp} |")
    lines += ["", "## 问法分布", ""]
    for k, v in sorted(audit["qtype"].items(), key=lambda x: -x[1]):
        lines.append(f"- {k}：{v}")
    # 语义去重明细（PRD §5.3）
    if dedup_pairs:
        lines += ["", "## 语义去重明细", "",
                  f"- 去重前条数：{dedup_pairs.get('before', '')}，去重后条数：{dedup_pairs.get('after', '')}，移除对数：{len(dedup_pairs.get('pairs', []))}",
                  "",
                  "| q1 | q2 | 相似度 | 保留 | 移除 |",
                  "|---|---|---|---|---|"]
        for p in dedup_pairs.get("pairs", []):
            lines.append(f"| {p['q1']} | {p['q2']} | {p['sim']} | {p['kept_question']} | {p['removed_question']} |")
    # tag 分布（PRD §4.3）
    if tag_dist is not None:
        lines += ["", "## Tag 分布", ""]
        for k, v in sorted(tag_dist.items(), key=lambda x: -x[1]):
            lines.append(f"- {k}：{v}")
    path.write_text("\n".join(lines), encoding="utf-8")


def enhance(n_auto: int = 30, n_colloquial: int = 10, n_hard: int = 25,
            do_semantic_dedup: bool = True) -> dict:
    """评估集扩容主入口（增量模式，PRD §4.4）。

    流程：读现有评估集 → 抽检(去重/snippet校验/答案校验/问法统计) → LLM 修复问题条目
    → 按问法类型补齐自动基线 → 口语化补充 → 难例/边界例补充 → 文本级去重
    → tag 回填 → 语义去重(可选) → 合并写回 gold_set.json(新版本) → 文档快照 → 抽检报告。

    兼容：返回结构在既有字段基础上扩展，不删除旧字段。
    """
    config.ensure_dirs()
    gs = load_gold_set()
    items = list(gs.get("items", []))
    if not items:
        return {"ok": False, "reason": "评估集为空，无法增强"}

    doc_texts = _read_doc_texts()
    all_text = "\n".join(doc_texts.values())
    version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # 1) 抽检现有条目 + 修复问题条目
    audit = audit_items(items, all_text)
    fixed, failed = _fix_items(audit["deduped"], audit["flags"], all_text)
    kept = []
    for i, it in enumerate(audit["deduped"]):
        kept.append(fixed[i] if i in fixed else it)
    existing_q = {it["question"] for it in kept}

    # 2) 按问法类型补齐自动基线
    added_by_type = _generate_by_type(n_auto, kept, all_text) if n_auto > 0 else []
    seen_after_type = existing_q | {it["question"] for it in added_by_type}
    # 3) 口语化补充
    added_coll = _generate_colloquial(n_colloquial, all_text, seen_after_type) if n_colloquial > 0 else []
    # 4) 难例/边界例补充
    seen_all_new = seen_after_type | {it["question"] for it in added_coll}
    added_hard = _generate_hard_cases(n_hard, all_text, seen_all_new) if n_hard > 0 else []

    # 5) 合并 + tag 回填 + 文本级去重
    merged = kept + added_by_type + added_coll + added_hard
    merged = _backfill_tags(merged)
    seen, merged_u = set(), []
    for it in merged:
        q = (it.get("question") or "").strip()
        if q and q not in seen:
            seen.add(q)
            merged_u.append(it)
    merged = merged_u

    # 6) 语义去重（PRD §5）
    dedup_pairs = None
    if do_semantic_dedup:
        merged, pairs, before, after = _semantic_dedup(merged)
        dedup_pairs = {"pairs": pairs, "before": before, "after": after}

    # 7) tag 分布统计
    tag_dist = {}
    for it in merged:
        t = it.get("tag", "其他") or "其他"
        tag_dist[t] = tag_dist.get(t, 0) + 1

    # 8) 写回 + 快照
    record = {
        "version": version,
        "created_at": datetime.now().isoformat(),
        "doc_set": _doc_fingerprint(),
        "params": gs.get("params", {}),
        "items": merged,
    }
    config.gold_set_path().write_text(json.dumps(record, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    _save_snapshot(version)

    # 9) 抽检报告（含 tag 分布 + 语义去重明细）
    _write_audit_report(audit, fixed, failed, added_coll, version,
                        tag_dist=tag_dist, dedup_pairs=dedup_pairs,
                        added_by_type=added_by_type, added_hard=added_hard)

    return {"ok": True, "total": len(merged), "kept": len(kept),
            "removed_dup": len(audit["removed_dup"]),
            "added_by_type": len(added_by_type),
            "added_colloquial": len(added_coll),
            "added_hard": len(added_hard),
            "semantic_dedup": dedup_pairs,
            "fixed": len(fixed), "fix_failed": len(failed),
            "version": version}


if __name__ == "__main__":
    config.ensure_dirs()
    r = enhance()
    print(json.dumps(r, ensure_ascii=False, indent=2))
