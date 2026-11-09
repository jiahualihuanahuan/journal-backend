"""对 vault 里的 Markdown 做检索：BM25（中英文都支持，零显存）+ 可选向量检索。"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CJK = r"\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
TOKEN_RE = re.compile(rf"[{CJK}]+|[a-zA-Z0-9][a-zA-Z0-9_\-\.]*")
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
FM_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.M)


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for m in TOKEN_RE.finditer(text.lower()):
        t = m.group(0)
        if re.match(rf"[{CJK}]", t):
            out.extend(t)                                  # 单字
            out.extend(t[i:i + 2] for i in range(len(t) - 1))  # 双字
        else:
            out.append(t.strip(".-"))
    return [t for t in out if t]


@dataclass
class Chunk:
    path: str          # vault 内相对路径，不含 .md
    heading: str
    text: str
    date: str = ""
    tokens: list[str] = field(default_factory=list, repr=False)

    @property
    def link(self) -> str:
        return f"[[{self.path}#{self.heading}]]" if self.heading else f"[[{self.path}]]"

    @property
    def sha(self) -> str:
        return hashlib.sha256(f"{self.path}|{self.heading}|{self.text}".encode()).hexdigest()


def split_note(rel: str, content: str, max_chars: int = 1200) -> list[Chunk]:
    date = ""
    m = FM_RE.match(content)
    if m:
        dm = re.search(r"^date:\s*['\"]?(\d{4}-\d{2}-\d{2})", m.group(1), re.M)
        if dm:
            date = dm.group(1)
        content = content[m.end():]
    if not date:
        dm = DATE_RE.search(Path(rel).name)
        date = dm.group(1) if dm else ""
    chunks: list[Chunk] = []
    positions = [(mm.start(), mm.group(2).strip()) for mm in HEADING_RE.finditer(content)]
    sections = []
    if not positions or positions[0][0] > 0:
        sections.append(("", content[: positions[0][0]] if positions else content))
    for i, (pos, title) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(content)
        body = content[pos:end]
        sections.append((re.sub(r"[#\[\]|^]", "", title).strip(), body))
    for heading, body in sections:
        body = body.strip()
        if not body or len(body) < 2:
            continue
        buf = ""
        for para in re.split(r"\n\s*\n", body):
            if buf and len(buf) + len(para) > max_chars:
                chunks.append(Chunk(rel, heading, buf.strip(), date))
                buf = ""
            buf += para + "\n\n"
        if buf.strip():
            chunks.append(Chunk(rel, heading, buf.strip(), date))
    return chunks


class Embedder:
    """向量化：openai（任意 OpenAI 兼容 /embeddings）或 local（sentence-transformers，CPU）。"""

    def __init__(self, settings, llm=None):
        self.s = settings
        self.llm = llm
        self._local = None
        self.cache_file = settings.data_dir / "embeddings_cache.json"
        self.cache: dict[str, list[float]] = {}
        if self.cache_file.exists():
            try:
                self.cache = json.loads(self.cache_file.read_text("utf-8"))
            except Exception:
                self.cache = {}

    @property
    def enabled(self) -> bool:
        return self.s.embed_backend in {"openai", "local"}

    async def _embed_raw(self, texts: list[str]) -> list[list[float]]:
        if self.s.embed_backend == "openai":
            return await self.llm.embed(texts)
        if self._local is None:
            from sentence_transformers import SentenceTransformer  # 可选依赖
            self._local = SentenceTransformer(self.s.embed_model, device=self.s.embed_device)
        vecs = await asyncio.to_thread(self._local.encode, texts, normalize_embeddings=True)
        return [v.tolist() for v in vecs]

    async def embed_chunks(self, chunks: list[Chunk], batch: int = 32) -> np.ndarray:
        missing = [c for c in chunks if c.sha not in self.cache]
        for i in range(0, len(missing), batch):
            part = missing[i:i + batch]
            vecs = await self._embed_raw([f"{c.path} {c.heading}\n{c.text}" for c in part])
            for c, v in zip(part, vecs):
                self.cache[c.sha] = v
        if missing:
            self.s.data_dir.mkdir(parents=True, exist_ok=True)
            keep = {c.sha for c in chunks}
            self.cache = {k: v for k, v in self.cache.items() if k in keep}
            self.cache_file.write_text(json.dumps(self.cache), "utf-8")
        if not chunks:
            return np.zeros((0, 1))
        return _normalize(np.array([self.cache[c.sha] for c in chunks], dtype=np.float32))

    async def embed_query(self, q: str) -> np.ndarray:
        return _normalize(np.array(await self._embed_raw([q]), dtype=np.float32))[0]


def _normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=-1, keepdims=True)
    n[n == 0] = 1
    return m / n


class VaultIndex:
    def __init__(self, settings, embedder: Embedder | None = None):
        self.s = settings
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.df: Counter = Counter()
        self.avgdl = 1.0
        self.vectors: np.ndarray | None = None
        self._lock = asyncio.Lock()

    def _excluded(self, rel: Path) -> bool:
        parts = rel.as_posix()
        excl = list(self.s.index_exclude) + [
            (Path(self.s.journal_dir) / self.s.inbox_subdir).as_posix()]
        return any(parts == e or parts.startswith(e.rstrip("/") + "/") or e in rel.parts for e in excl)

    def _scan(self) -> list[Chunk]:
        root = self.s.vault_path
        chunks: list[Chunk] = []
        if not root.exists():
            return chunks
        for p in sorted(root.rglob("*.md")):
            rel = p.relative_to(root)
            if self._excluded(rel):
                continue
            try:
                content = p.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            chunks.extend(split_note(rel.with_suffix("").as_posix(), content))
        return chunks

    async def rebuild(self) -> dict:
        async with self._lock:
            chunks = await asyncio.to_thread(self._scan)
            df: Counter = Counter()
            total = 0
            for c in chunks:
                c.tokens = tokenize(f"{c.heading} {c.text}")
                df.update(set(c.tokens))
                total += len(c.tokens)
            self.chunks, self.df = chunks, df
            self.avgdl = (total / len(chunks)) if chunks else 1.0
            self.vectors = None
            emb_err = None
            if self.embedder and self.embedder.enabled and chunks:
                try:
                    self.vectors = await self.embedder.embed_chunks(chunks)
                except Exception as e:  # 向量服务挂了就退回 BM25
                    emb_err = str(e)
            return {"chunks": len(chunks), "notes": len({c.path for c in chunks}),
                    "vectors": self.vectors is not None, "embed_error": emb_err}

    def _bm25(self, q_tokens: list[str], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
        n = len(self.chunks)
        scores = np.zeros(n, dtype=np.float32)
        q = Counter(q_tokens)
        for i, c in enumerate(self.chunks):
            tf = Counter(c.tokens)
            dl = len(c.tokens)
            s = 0.0
            for t in q:
                if t not in tf:
                    continue
                idf = math.log(1 + (n - self.df[t] + 0.5) / (self.df[t] + 0.5))
                f = tf[t]
                s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / self.avgdl))
            scores[i] = s
        return scores

    async def search(self, query: str, top_k: int | None = None) -> list[dict]:
        top_k = top_k or self.s.search_top_k
        if not self.chunks:
            return []
        bm = self._bm25(tokenize(query))
        score = bm / bm.max() if bm.max() > 0 else bm
        mode = "bm25"
        if self.vectors is not None and self.embedder:
            try:
                qv = await self.embedder.embed_query(query)
                cos = self.vectors @ qv
                score = 0.4 * score + 0.6 * np.clip(cos, 0, 1)
                mode = "hybrid"
            except Exception:
                pass
        order = np.argsort(-score)[:top_k]
        out = []
        for i in order:
            if score[i] <= 0:
                continue
            c = self.chunks[int(i)]
            out.append({"link": c.link, "path": c.path, "heading": c.heading, "date": c.date,
                        "score": round(float(score[i]), 4), "text": c.text, "mode": mode})
        return out


def build_context(hits: list[dict], max_chars: int) -> str:
    parts, used = [], 0
    for h in hits:
        block = f"来源 {h['link']}（日期 {h['date'] or '未知'}）\n{h['text']}"
        if used + len(block) > max_chars:
            block = block[: max(0, max_chars - used)]
        if not block:
            break
        parts.append(block)
        used += len(block)
    return "\n\n-----\n\n".join(parts)
