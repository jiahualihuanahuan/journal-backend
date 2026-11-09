"""Inbox 监听：手机把录音或文字丢进 Journal/Inbox，自动转写、追问、生成日记。

流程（幂等，可反复运行）：
1. 新的音频 / .md / .txt -> 转写 -> 生成草稿 Inbox/<时间>-草稿.md（带追问），原文件移到 Inbox/processed/
2. 你在草稿里回答问题，勾选「我已回答完」或把 status 改成 answered
3. 下一轮 -> 生成正式日记（追加到 Journal/YYYY-MM-DD.md），草稿 status 变 done 并移到 processed/
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import time
from datetime import date, datetime
from pathlib import Path

from .journal import compose_entry, make_followups
from .transcribe import AUDIO_EXT

log = logging.getLogger("journal.inbox")

DRAFT_KEY = "journal-backend: draft"
FM_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.S)
DONE_BOX_RE = re.compile(r"^\s*- \[[xX]\] 我已回答完", re.M)
TEXT_EXT = {".md", ".txt"}


def _frontmatter(text: str) -> dict:
    m = FM_RE.match(text)
    out = {}
    if m:
        for ln in m.group(1).splitlines():
            if ":" in ln:
                k, v = ln.split(":", 1)
                out[k.strip()] = v.strip().strip('"\'')
    return out


def _set_fm(text: str, updates: dict) -> str:
    m = FM_RE.match(text)
    if not m:
        return text
    lines = m.group(1).splitlines()
    keys = {ln.split(":", 1)[0].strip(): i for i, ln in enumerate(lines) if ":" in ln}
    for k, v in updates.items():
        if k in keys:
            lines[keys[k]] = f"{k}: {v}"
        else:
            lines.append(f"{k}: {v}")
    return "---\n" + "\n".join(lines) + "\n---\n" + text[m.end():]


def _section(text: str, title: str) -> str:
    m = re.search(rf"^## {re.escape(title)}\s*\n(.*?)(?=^## |^---\s*$|\Z)", text, re.S | re.M)
    return m.group(1).strip() if m else ""


def parse_draft(text: str) -> tuple[str, list[dict], str]:
    raw = _section(text, "原始记录")
    qa_text = _section(text, "追问")
    answers = []
    for m in re.finditer(r"^### \d+\.\s*(.+?)\n(.*?)(?=^### \d+\.|\Z)", qa_text, re.S | re.M):
        ans = re.sub(r"^\s*回答[：:]\s*", "", m.group(2).strip()).strip()
        answers.append({"question": m.group(1).strip(), "answer": ans})
    extra = _section(text, "想再补充的")
    return raw, answers, extra


def render_draft(raw: str, questions: list[str], day: date, source: str) -> str:
    lines = ["---", DRAFT_KEY, "status: needs-answers", f"date: {day.isoformat()}",
             f"source: \"{source}\"", "---", "", f"# 日记草稿 {day.isoformat()}", "",
             "> [!tip] 直接在每个问题下面写回答（不想答的可以留空），写完勾选最下面的框。", "",
             "## 原始记录", "", raw.strip(), "", "## 追问", ""]
    for i, q in enumerate(questions, 1):
        lines += [f"### {i}. {q}", "回答：", ""]
    lines += ["## 想再补充的", "", "", "---", "- [ ] 我已回答完，生成日记", ""]
    return "\n".join(lines)


class InboxProcessor:
    def __init__(self, settings, llm, transcriber, index=None):
        self.s = settings
        self.llm = llm
        self.tr = transcriber
        self.index = index
        self.state_file = settings.data_dir / "inbox_state.json"
        self.state = self._load_state()

    def _load_state(self) -> dict:
        try:
            return json.loads(self.state_file.read_text("utf-8"))
        except Exception:
            return {"sources": {}}

    def _save_state(self):
        self.s.data_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=1), "utf-8")
        tmp.replace(self.state_file)

    @property
    def processed_dir(self) -> Path:
        return self.s.inbox_path / "processed"

    def _settled(self, p: Path) -> bool:
        return time.time() - p.stat().st_mtime >= self.s.inbox_settle_seconds

    def _move_processed(self, p: Path) -> Path:
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        dest = self.processed_dir / p.name
        n = 1
        while dest.exists():
            dest = self.processed_dir / f"{p.stem}-{n}{p.suffix}"
            n += 1
        shutil.move(str(p), dest)
        return dest

    async def process_once(self) -> dict:
        inbox = self.s.inbox_path
        result = {"drafts": [], "composed": [], "skipped": [], "errors": []}
        if not inbox.exists():
            inbox.mkdir(parents=True, exist_ok=True)
            return result
        for p in sorted(x for x in inbox.iterdir() if x.is_file()):
            if p.name.startswith(".") or p.suffix.lower() not in AUDIO_EXT | TEXT_EXT:
                continue
            try:
                if not self._settled(p):
                    result["skipped"].append(p.name)
                    continue
                if p.suffix.lower() in TEXT_EXT:
                    text = p.read_text(encoding="utf-8")
                    if DRAFT_KEY in text[:300]:
                        r = await self._maybe_compose(p, text)
                        if r:
                            result["composed"].append(r)
                        continue
                r = await self._new_source(p)
                if r:
                    result["drafts"].append(r)
            except Exception as e:  # 单个文件出错不影响其他文件
                log.exception("inbox error on %s", p)
                result["errors"].append({"file": p.name, "error": str(e)})
        return result

    async def _new_source(self, p: Path) -> str | None:
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest in self.state["sources"]:
            self._move_processed(p)  # 重复的（例如同步重新出现），直接归档
            return None
        day = datetime.fromtimestamp(p.stat().st_mtime).date()
        if p.suffix.lower() in AUDIO_EXT:
            if not self.tr.enabled:
                raise RuntimeError("收到音频但 WHISPER_BACKEND=off")
            raw = await self.tr.transcribe(p)
        else:
            raw = p.read_text(encoding="utf-8")
            raw = FM_RE.sub("", raw, count=1).strip()
        if not raw.strip():
            raise RuntimeError("转写结果为空")
        moved = self._move_processed(p)
        source_rel = moved.relative_to(self.s.vault_path).as_posix()
        if self.s.inbox_skip_followups:
            r = await compose_entry(self.llm, self.s, raw, None, day)
            self.state["sources"][digest] = {"file": p.name, "journal": r["path"], "at": datetime.now().isoformat()}
            self._save_state()
            await self._reindex()
            return r["path"]
        questions = await make_followups(self.llm, self.s, raw)
        stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        draft = self.s.inbox_path / f"{stamp}-草稿.md"
        draft.write_text(render_draft(raw, questions, day, source_rel), encoding="utf-8")
        self.state["sources"][digest] = {"file": p.name, "draft": draft.name, "at": datetime.now().isoformat()}
        self._save_state()
        return draft.name

    async def _maybe_compose(self, p: Path, text: str) -> str | None:
        fm = _frontmatter(text)
        status = fm.get("status", "")
        if status == "done":
            self._move_processed(p)
            return None
        if status != "answered" and not DONE_BOX_RE.search(text):
            return None
        raw, answers, extra = parse_draft(text)
        if extra:
            raw = raw + "\n\n" + extra
        try:
            day = date.fromisoformat(fm.get("date", ""))
        except ValueError:
            day = None
        r = await compose_entry(self.llm, self.s, raw, answers, day)
        # 先标记 done 再移动：中途崩溃也不会重复生成
        p.write_text(_set_fm(text, {"status": "done", "journal": f"\"[[{r['path']}]]\""}), encoding="utf-8")
        self._move_processed(p)
        await self._reindex()
        return r["path"]

    async def _reindex(self):
        if self.index is not None:
            try:
                await self.index.rebuild()
            except Exception:
                log.exception("reindex failed")
