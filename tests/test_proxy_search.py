import json

import httpx
from fastapi.testclient import TestClient

from app.main import create_app


def test_models(client):
    ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
    assert ids == ["journal", "journal-raw"]


def test_chat_proxy_injects_context_and_model(client, fake):
    r = client.post("/v1/chat/completions", json={
        "model": "journal", "messages": [{"role": "user", "content": "上次纳斯达克大跌我怎么想的？"}]})
    assert r.status_code == 200, r.text
    sent = fake.requests[-1]["body"]
    assert sent["model"] == "test-model"
    assert sent["messages"][0]["role"] == "system"
    assert "[[Journal/2026-11-05#投资记录]]" in sent["messages"][0]["content"]
    assert ".obsidian" not in sent["messages"][0]["content"]
    j = r.json()
    assert j["model"] == "journal"
    assert "Journal/2026-11-05" in j["journal_sources"][0]


def test_chat_proxy_raw_passthrough(client, fake):
    r = client.post("/v1/chat/completions", json={
        "model": "journal-raw", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5})
    assert r.status_code == 200
    sent = fake.requests[-1]["body"]
    assert sent["messages"] == [{"role": "user", "content": "hi"}]
    assert sent["max_tokens"] == 5


def test_chat_proxy_streaming(client):
    with client.stream("POST", "/v1/chat/completions", json={
            "model": "journal", "stream": True, "messages": [{"role": "user", "content": "你好"}]}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        body = b"".join(r.iter_raw()).decode()
    assert "你好" in body and "data: [DONE]" in body


def test_upstream_error_is_502(settings):
    app = create_app(settings, transport=httpx.MockTransport(lambda req: httpx.Response(500, text="boom")))
    with TestClient(app) as c:
        r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}]})
        assert r.status_code == 502


def test_search_bm25_chinese(client):
    res = client.post("/search", json={"query": "恐慌 大跌"}).json()["results"]
    assert res and res[0]["path"] == "Journal/2026-11-05"
    assert res[0]["link"] == "[[Journal/2026-11-05#投资记录]]"
    assert res[0]["mode"] == "bm25"
    kid = client.post("/search", json={"query": "宇航员"}).json()["results"]
    assert kid[0]["path"] == "Notes"
    assert all(".obsidian" not in h["path"] for h in res + kid)


def test_search_hybrid_with_embeddings(settings, fake):
    settings.embed_backend = "openai"
    settings.embed_base_url = "http://vllm.test/v1"
    app = create_app(settings, transport=httpx.MockTransport(fake.handler))
    with TestClient(app) as c:
        assert c.get("/status").json()["vectors"] is True
        res = c.post("/search", json={"query": "QQQ 投资"}).json()["results"]
        assert res[0]["mode"] == "hybrid" and res[0]["path"] == "Journal/2026-11-05"
    assert (settings.data_dir / "embeddings_cache.json").exists()


def test_search_falls_back_when_embeddings_down(settings):
    settings.embed_backend = "openai"
    settings.embed_base_url = "http://vllm.test/v1"
    app = create_app(settings, transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    with TestClient(app) as c:
        res = c.post("/search", json={"query": "恐慌"}).json()["results"]
        assert res and res[0]["mode"] == "bm25"


def test_ask_returns_answer_and_citations(client):
    j = client.post("/ask", json={"question": "上次大跌时我怎么想的"}).json()
    assert "[[Journal/2026-11-05" in j["answer"]
    assert j["sources"][0]["link"].startswith("[[Journal/2026-11-05")


def test_reindex_picks_up_new_notes(client, vault):
    (vault / "New.md").write_text("# 新笔记\n\n滑雪 第一次", "utf-8")
    assert client.post("/search", json={"query": "滑雪"}).json()["results"] == []
    assert client.post("/reindex").json()["notes"] >= 3
    assert client.post("/search", json={"query": "滑雪"}).json()["results"][0]["path"] == "New"


def test_api_key_required(settings, fake):
    settings.backend_api_key = "test-key-not-real"
    app = create_app(settings, transport=httpx.MockTransport(fake.handler))
    with TestClient(app) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/v1/models").status_code == 401
        assert c.get("/v1/models", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert c.get("/v1/models", headers={"Authorization": "Bearer test-key-not-real"}).status_code == 200


def test_transcribe_disabled_503(client):
    r = client.post("/transcribe", files={"file": ("a.m4a", b"123", "audio/mp4")})
    assert r.status_code == 503
