"""FastAPI 入口。"""
from __future__ import annotations

import asyncio
import hmac
import logging
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
import datetime as dt
from datetime import date
from pathlib import Path
from typing import Any

import httpx
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import __version__
from .config import Settings
from .inbox import InboxProcessor
from .journal import compose_entry, make_followups
from .llm import LLMClient, LLMError
from .prompts import load_prompt
from .review import render_digest, scan_reviews, write_digest
from .search import Embedder, VaultIndex, build_context
from .transcribe import AUDIO_EXT, TranscribeDisabled, Transcriber

log = logging.getLogger("journal")
MODEL_RAG = "journal"
MODEL_RAW = "journal-raw"


class FollowupsIn(BaseModel):
    text: str
    max_questions: int = Field(3, ge=1, le=5)


class ComposeIn(BaseModel):
    text: str
    answers: list[dict] | str | None = None
    date: dt.date | None = None
    dry_run: bool = False


class SearchIn(BaseModel):
    query: str
    top_k: int | None = Field(None, ge=1, le=50)


class AskIn(BaseModel):
    question: str
    top_k: int | None = Field(None, ge=1, le=50)


def create_app(settings: Settings | None = None, transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        llm = LLMClient(settings, transport=transport)
        index = VaultIndex(settings, Embedder(settings, llm))
        tr = Transcriber(settings, llm.client)
        app.state.llm, app.state.index, app.state.tr = llm, index, tr
        app.state.inbox = InboxProcessor(settings, llm, tr, index)
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        try:
            info = await index.rebuild()
            log.info("index built: %s", info)
        except Exception:
            log.exception("initial index failed")
        task = None
        if settings.inbox_enabled or settings.review_digest_enabled:
            task = asyncio.create_task(_background(app))
        yield
        if task:
            task.cancel()
        await llm.aclose()

    async def _background(app: FastAPI):
        last_digest = 0.0
        while True:
            try:
                if settings.inbox_enabled:
                    r = await app.state.inbox.process_once()
                    if r["drafts"] or r["composed"] or r["errors"]:
                        log.info("inbox: %s", r)
                if settings.review_digest_enabled and time.time() - last_digest > 3600:
                    write_digest(settings, date.today())
                    last_digest = time.time()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("background loop error")
            await asyncio.sleep(settings.inbox_poll_seconds)

    app = FastAPI(title="journal-backend", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=False,
                       allow_methods=["*"], allow_headers=["*"])

    async def auth(request: Request):
        key = settings.backend_api_key
        if not key:
            return
        got = request.headers.get("authorization", "")
        got = got[7:] if got.lower().startswith("bearer ") else request.headers.get("x-api-key", "")
        if not hmac.compare_digest(got.encode(), key.encode()):
            raise HTTPException(401, "invalid or missing API key")

    @app.exception_handler(LLMError)
    async def _llm_err(_req, exc: LLMError):
        return JSONResponse({"error": "vLLM 调用失败", "status": exc.status, "detail": exc.body[:1000]}, status_code=502)

    @app.exception_handler(httpx.HTTPError)
    async def _http_err(_req, exc: httpx.HTTPError):
        return JSONResponse({"error": "连接上游失败", "detail": str(exc)}, status_code=502)

    @app.exception_handler(ValueError)
    async def _val_err(_req, exc: ValueError):
        return JSONResponse({"error": str(exc)}, status_code=422)

    @app.get("/health")
    async def health():
        return {"ok": True, "version": __version__}

    deps = [Depends(auth)]

    @app.get("/status", dependencies=deps)
    async def status(request: Request, check_upstream: bool = False):
        idx: VaultIndex = request.app.state.index
        upstream = None
        if check_upstream:
            try:
                llm: LLMClient = request.app.state.llm
                r = await llm.client.get(f"{settings.vllm_base_url}/models", headers=llm._headers(), timeout=10)
                upstream = {"ok": r.status_code == 200, "status": r.status_code,
                            "models": [m.get("id") for m in r.json().get("data", [])] if r.status_code == 200 else None}
            except Exception as e:
                upstream = {"ok": False, "error": str(e)}
        return {"upstream": upstream, "vault": str(settings.vault_path), "journal_dir": settings.journal_dir,
                "vault_exists": settings.vault_path.exists(), "model": settings.vllm_model,
                "chunks": len(idx.chunks), "vectors": idx.vectors is not None,
                "whisper": settings.whisper_backend, "inbox_enabled": settings.inbox_enabled}

    # ---------- OpenAI 兼容接口（给 Obsidian Copilot / Text Generator 用） ----------

    @app.get("/v1/models", dependencies=deps)
    async def models():
        now = int(time.time())
        return {"object": "list", "data": [
            {"id": MODEL_RAG, "object": "model", "created": now, "owned_by": "journal-backend"},
            {"id": MODEL_RAW, "object": "model", "created": now, "owned_by": "journal-backend"}]}

    @app.post("/v1/chat/completions", dependencies=deps)
    async def chat_completions(request: Request):
        body: dict[str, Any] = await request.json()
        messages = list(body.get("messages") or [])
        if not messages:
            raise HTTPException(400, "messages is required")
        requested = str(body.get("model") or MODEL_RAG)
        sources: list[dict] = []
        if requested != MODEL_RAW:
            query = _last_user_text(messages)
            hits = await request.app.state.index.search(query) if query else []
            sources = hits
            ctx = build_context(hits, settings.context_max_chars) or "（没有检索到相关日记）"
            system = load_prompt(settings, "chat_system", today=date.today().isoformat(), context=ctx)
            messages = [{"role": "system", "content": system}] + messages
        payload = {k: v for k, v in body.items() if k not in {"model", "messages"}}
        payload["messages"] = messages
        llm: LLMClient = request.app.state.llm
        if body.get("stream"):
            resp = await llm.proxy_stream(payload)
            if resp.status_code >= 400:
                text = (await resp.aread()).decode("utf-8", "replace")
                await resp.aclose()
                raise LLMError(resp.status_code, text)
            return StreamingResponse(llm.iter_stream(resp), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        r = await llm.proxy(payload)
        if r.status_code >= 400:
            raise LLMError(r.status_code, r.text)
        data = r.json()
        data["model"] = requested
        if sources:
            data["journal_sources"] = [s["link"] for s in sources]
        return data

    # ---------- 日记流程 ----------

    @app.post("/transcribe", dependencies=deps)
    async def transcribe(request: Request, file: UploadFile = File(...)):
        suffix = Path(file.filename or "audio.m4a").suffix.lower() or ".m4a"
        if suffix not in AUDIO_EXT:
            raise HTTPException(415, f"unsupported audio type {suffix}")
        tmpdir = Path(tempfile.mkdtemp(prefix="jb-"))
        tmp = tmpdir / f"{uuid.uuid4().hex}{suffix}"
        try:
            with open(tmp, "wb") as f:
                while chunk := await file.read(1 << 20):
                    f.write(chunk)
            text = await request.app.state.tr.transcribe(tmp)
        except TranscribeDisabled as e:
            raise HTTPException(503, str(e))
        finally:
            tmp.unlink(missing_ok=True)
            tmpdir.rmdir()
        return {"text": text}

    @app.post("/journal/followups", dependencies=deps)
    async def followups(request: Request, body: FollowupsIn):
        qs = await make_followups(request.app.state.llm, settings, body.text, body.max_questions)
        return {"questions": qs}

    @app.post("/journal/compose", dependencies=deps)
    async def compose(request: Request, body: ComposeIn):
        r = await compose_entry(request.app.state.llm, settings, body.text, body.answers, body.date, body.dry_run)
        if r["action"] != "dry_run":
            await request.app.state.index.rebuild()
        return r

    @app.post("/search", dependencies=deps)
    async def search(request: Request, body: SearchIn):
        hits = await request.app.state.index.search(body.query, body.top_k)
        return {"results": hits}

    @app.post("/ask", dependencies=deps)
    async def ask(request: Request, body: AskIn):
        hits = await request.app.state.index.search(body.question, body.top_k)
        if not hits:
            return {"answer": "日记里没有找到相关的记录。", "sources": []}
        ctx = build_context(hits, settings.context_max_chars)
        system = load_prompt(settings, "ask", today=date.today().isoformat())
        answer = await request.app.state.llm.chat([
            {"role": "system", "content": system},
            {"role": "user", "content": f"【日记片段】\n{ctx}\n\n【问题】\n{body.question}"}])
        return {"answer": answer, "sources": [{"link": h["link"], "date": h["date"], "score": h["score"]} for h in hits]}

    @app.post("/reindex", dependencies=deps)
    async def reindex(request: Request):
        return await request.app.state.index.rebuild()

    @app.post("/inbox/process", dependencies=deps)
    async def inbox_process(request: Request):
        return await request.app.state.inbox.process_once()

    @app.get("/invest/reviews", dependencies=deps)
    async def invest_reviews(as_of: date | None = None, upcoming_days: int = 0, write: bool = False):
        d = as_of or date.today()
        items = scan_reviews(settings, d, upcoming_days)
        out = {"as_of": d.isoformat(), "items": items, "markdown": render_digest(items, d)}
        if write:
            out["written"] = write_digest(settings, d, upcoming_days).relative_to(settings.vault_path).as_posix()
        return out

    return app


def _last_user_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):  # OpenAI 多段格式
                return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return ""


def get_app() -> FastAPI:  # uvicorn --factory app.main:get_app
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return create_app()
