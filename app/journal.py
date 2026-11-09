"""日记：追问、结构化整理、渲染 Markdown、安全写入（只追加，绝不覆盖）。"""
from __future__ import annotations

import calendar
import os
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from .prompts import load_prompt

DATE_OK = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def add_months(d: date, months: int) -> date:
    y, m = divmod(d.month - 1 + months, 12)
    y, m = d.year + y, m + 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def review_dates(d: date) -> list[tuple[str, date]]:
    return [("一周后回看", d + timedelta(days=7)),
            ("一个月后回看", add_months(d, 1)),
            ("三个月后回看", add_months(d, 3))]


def _str_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v.strip()] if v.strip() else []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    return [str(v)]


def _s(v: Any) -> str:
    return "" if v is None else str(v).strip()


async def make_followups(llm, settings, text: str, n: int = 3) -> list[str]:
    system = load_prompt(settings, "followups", max_questions=n)
    data = await llm.chat_json([{"role": "system", "content": system},
                                {"role": "user", "content": text}])
    qs = data.get("questions", []) if isinstance(data, dict) else data
    qs = [q for q in _str_list(qs) if q]
    return qs[:n]


def _format_answers(answers: Any) -> str:
    if not answers:
        return ""
    if isinstance(answers, str):
        return answers
    lines = []
    for a in answers:
        if isinstance(a, dict):
            q, ans = _s(a.get("question")), _s(a.get("answer"))
            if ans:
                lines.append(f"问：{q}\n答：{ans}")
        elif _s(a):
            lines.append(_s(a))
    return "\n\n".join(lines)


async def structure_entry(llm, settings, text: str, answers: Any, day: date) -> dict:
    system = load_prompt(settings, "compose", date=day.isoformat())
    user = f"【原始语音/文字记录】\n{text.strip()}"
    ans = _format_answers(answers)
    if ans:
        user += f"\n\n【追问和回答】\n{ans}"
    data = await llm.chat_json([{"role": "system", "content": system},
                                {"role": "user", "content": user}])
    if not isinstance(data, dict):
        raise ValueError("compose: model returned non-object JSON")
    return data


# ---------- 渲染 ----------

def _ai_callout(items: list[str]) -> list[str]:
    if not items:
        return []
    out = ["", "> [!ai] AI 补充（不是我说的，仅供参考）"]
    out += [f"> - {x}" for x in items]
    return out


def _fear(v: Any) -> int | None:
    try:
        f = int(round(float(v)))
        return max(1, min(10, f))
    except (TypeError, ValueError):
        return None


def render_sections(data: dict, day: date, h: str = "##", raw: str = "", answers: Any = None) -> list[str]:
    lines: list[str] = []
    sub = h + "#"

    learn = data.get("learnings") or {}
    items = _str_list(learn.get("items") if isinstance(learn, dict) else learn)
    ai = _str_list(learn.get("ai_additions")) if isinstance(learn, dict) else []
    if items or ai:
        lines += [f"{h} 今天学到的", ""] + [f"- {x}" for x in items] + _ai_callout(ai) + [""]

    inv = data.get("investing")
    if isinstance(inv, dict) and any(_s(inv.get(k)) for k in ("event", "decision", "fear_level", "market_mood")):
        fear = _fear(inv.get("fear_level"))
        lines += [f"{h} 投资记录", ""]
        lines.append(f"- **发生了什么**：{_s(inv.get('event'))}")
        lines.append(f"- **恐慌程度**：{fear}/10" if fear else "- **恐慌程度**：（未说明）")
        lines.append(f"- **当时的情绪**：{_s(inv.get('emotions'))}")
        lines.append(f"- **决策**：{_s(inv.get('decision'))}")
        reasons = _str_list(inv.get("reasons"))
        lines.append("- **理由**：" + ("" if reasons else "（未说明）"))
        lines += [f"  - {r}" for r in reasons]
        lines.append(f"- **市场气氛**：{_s(inv.get('market_mood'))}")
        if _s(inv.get("plan")):
            lines.append(f"- **后续计划**：{_s(inv.get('plan'))}")
        if fear:
            lines.append(f"- fear_level:: {fear}")
        lines += _ai_callout(_str_list(inv.get("ai_additions")))
        lines += ["", f"{sub} 回看", ""]
        lines += [f"- [ ] {label} 📅 {d.isoformat()}：" for label, d in review_dates(day)]
        lines.append("")

    kid = data.get("kid")
    if isinstance(kid, dict) and any(_s(kid.get(k)) or _str_list(kid.get(k)) for k in ("what_happened", "quotes", "growth")):
        lines += [f"{h} 和孩子", ""]
        lines.append(f"- **发生了什么**：{_s(kid.get('what_happened'))}")
        quotes = _str_list(kid.get("quotes"))
        if quotes:
            lines.append("- **孩子的原话**：")
            lines += ["  - “" + q.strip('“”"') + "”" for q in quotes]
        growth = _str_list(kid.get("growth"))
        if growth:
            lines.append("- **成长和第一次**：")
            lines += [f"  - {g}" for g in growth]
        if _s(kid.get("feelings")):
            lines.append(f"- **我的感受**：{_s(kid.get('feelings'))}")
        lines += _ai_callout(_str_list(kid.get("ai_additions"))) + [""]

    other = data.get("other") or {}
    oitems = _str_list(other.get("items") if isinstance(other, dict) else other)
    oai = _str_list(other.get("ai_additions")) if isinstance(other, dict) else []
    if oitems or oai:
        lines += [f"{h} 其他", ""] + [f"- {x}" for x in oitems] + _ai_callout(oai) + [""]

    open_q = _str_list(data.get("open_questions"))
    if open_q:
        lines += [f"{h} 还没想清楚的", ""] + [f"- {x}" for x in open_q] + [""]

    if raw.strip():
        lines += ["> [!quote]- 原始记录（语音转写，未修改）"]
        lines += ["> " + ln if ln else ">" for ln in raw.strip().splitlines()]
        ans = _format_answers(answers)
        if ans:
            lines += [">", "> **追问和回答**"] + ["> " + ln if ln else ">" for ln in ans.splitlines()]
        lines.append("")
    return lines


def themes_of(data: dict) -> list[str]:
    t = []
    if (data.get("learnings") or {}) and _str_list((data.get("learnings") or {}).get("items")):
        t.append("learnings")
    if isinstance(data.get("investing"), dict) and _s(data["investing"].get("event") or data["investing"].get("decision")):
        t.append("investing")
    if isinstance(data.get("kid"), dict) and _s(data["kid"].get("what_happened") or _str_list(data["kid"].get("quotes"))):
        t.append("kid")
    return t


def _yaml_list(xs: list[str]) -> str:
    return "[" + ", ".join(re.sub(r"[\s,\[\]#]+", "-", x).strip("-") for x in xs if x) + "]"


def render_new_note(data: dict, day: date, raw: str = "", answers: Any = None) -> str:
    themes = themes_of(data)
    tags = ["journal"] + [f"journal/{t}" for t in themes] + _str_list(data.get("tags"))
    tags = list(dict.fromkeys(tags))
    fm = ["---", f"date: {day.isoformat()}", "type: journal", f"mood: \"{_s(data.get('mood')).replace(chr(34), '')}\"",
          f"tags: {_yaml_list(tags)}", f"themes: {_yaml_list(themes)}"]
    inv = data.get("investing")
    if isinstance(inv, dict) and _fear(inv.get("fear_level")):
        fm.append(f"fear_level: {_fear(inv.get('fear_level'))}")
    fm += ["created_by: journal-backend", "---", ""]
    title = _s(data.get("title"))
    body = [f"# {day.isoformat()} {title}".rstrip(), ""]
    if _s(data.get("summary")):
        body += [f"> [!summary] {_s(data.get('summary'))}", ""]
    return "\n".join(fm + body + render_sections(data, day, "##", raw if raw else "", answers)).rstrip() + "\n"


def render_append_block(data: dict, day: date, now: datetime, raw: str = "", answers: Any = None) -> str:
    themes = themes_of(data)
    tags = " ".join("#" + t for t in (["journal/" + x for x in themes]))
    head = [ "", "---", "", f"## 📝 {now.strftime('%H:%M')} 补充记录 {_s(data.get('title'))}".rstrip(), ""]
    if tags:
        head += [tags, ""]
    if _s(data.get("mood")):
        head += [f"mood:: {_s(data.get('mood'))}", ""]
    if _s(data.get("summary")):
        head += [f"> [!summary] {_s(data.get('summary'))}", ""]
    return "\n".join(head + render_sections(data, day, "###", raw, answers)).rstrip() + "\n"


def entry_path(settings, day: date) -> Path:
    iso = day.isoformat()
    if not DATE_OK.match(iso):
        raise ValueError("bad date")
    return settings.journal_path / f"{iso}.md"


def write_entry(path: Path, new_content: str, append_block: str) -> str:
    """新文件用 O_EXCL 创建；已存在就只在末尾追加。绝不改动已有内容。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            ends_nl = True
            if f.tell() > 0:
                f.seek(-1, os.SEEK_END)
                ends_nl = f.read(1) == b"\n"
        with open(path, "a", encoding="utf-8", newline="\n") as f:
            f.write(("" if ends_nl else "\n") + append_block)
        return "appended"
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
        f.write(new_content)
    return "created"


async def compose_entry(llm, settings, text: str, answers: Any = None, day: date | None = None,
                        dry_run: bool = False, now: datetime | None = None) -> dict:
    now = now or datetime.now()
    day = day or now.date()
    data = await structure_entry(llm, settings, text, answers, day)
    raw = text if settings.keep_raw else ""
    path = entry_path(settings, day)
    new_note = render_new_note(data, day, raw, answers)
    block = render_append_block(data, day, now, raw, answers)
    rel = path.relative_to(settings.vault_path).with_suffix("").as_posix()
    if dry_run:
        return {"action": "dry_run", "path": rel, "link": f"[[{rel}]]",
                "markdown": block if path.exists() else new_note, "data": data}
    action = write_entry(path, new_note, block)
    return {"action": action, "path": rel, "link": f"[[{rel}]]",
            "markdown": new_note if action == "created" else block, "data": data}
