"""投资回看：找出到期（+1周 / +1月 / +3月）还没勾选的回看项。"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

REVIEW_RE = re.compile(r"^\s*- \[( |x|X)\] (一周后回看|一个月后回看|三个月后回看) 📅 (\d{4}-\d{2}-\d{2})[：:]?(.*)$")
DIGEST_NAME = "投资回看清单.md"


def scan_reviews(settings, as_of: date, include_upcoming_days: int = 0) -> list[dict]:
    root = settings.journal_path
    out = []
    if not root.exists():
        return out
    inbox = settings.inbox_path
    for p in sorted(root.rglob("*.md")):
        if inbox in p.parents or p.name == DIGEST_NAME:
            continue
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for i, ln in enumerate(lines, 1):
            m = REVIEW_RE.match(ln)
            if not m or m.group(1).strip():
                continue
            due = date.fromisoformat(m.group(3))
            days = (due - as_of).days
            if days <= include_upcoming_days:
                rel = p.relative_to(settings.vault_path).with_suffix("").as_posix()
                out.append({"note": rel, "link": f"[[{rel}#投资记录]]", "line": i,
                            "slot": m.group(2), "due": due.isoformat(),
                            "status": "overdue" if days < 0 else ("due" if days == 0 else "upcoming"),
                            "days_until_due": days})
    out.sort(key=lambda r: r["due"])
    return out


def render_digest(items: list[dict], as_of: date) -> str:
    lines = ["---", "type: journal-review-digest", "created_by: journal-backend", "---", "",
             "# 投资回看清单", "",
             f"> [!info] 自动生成于 {as_of.isoformat()}，每次都会重写这个文件，请不要在这里写内容。",
             "> 回看请写在原日记的「回看」那一行，写完勾选即可。", ""]
    if not items:
        lines.append("目前没有到期的回看。")
    for r in items:
        flag = {"overdue": "⏰ 已过期", "due": "📌 今天到期", "upcoming": "🗓 即将到期"}[r["status"]]
        lines.append(f"- {flag} {r['due']} · {r['slot']} · {r['link']}")
    return "\n".join(lines) + "\n"


def write_digest(settings, as_of: date, upcoming_days: int = 3) -> Path:
    items = scan_reviews(settings, as_of, upcoming_days)
    path = settings.journal_path / DIGEST_NAME
    content = render_digest(items, as_of)
    old = path.read_text(encoding="utf-8") if path.exists() else None
    # 只在内容变化时写，避免 Syncthing 无谓同步（忽略生成日期行）
    strip = lambda t: re.sub(r"自动生成于 \S+", "", t or "")
    if old is None or strip(old) != strip(content):
        path.parent.mkdir(parents=True, exist_ok=True)
        if old is not None and "created_by: journal-backend" not in old:
            raise RuntimeError(f"{path} 不是本程序生成的文件，拒绝覆盖")
        path.write_text(content, encoding="utf-8")
    return path
