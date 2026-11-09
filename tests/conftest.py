"""测试用的假 vLLM：完全离线，不连任何服务器。"""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

COMPOSE_JSON = {
    "title": "大跌时加仓",
    "summary": "纳指大跌，硬着头皮买了一点。",
    "mood": "紧张但坚定",
    "tags": ["投资", "孩子"],
    "learnings": {"items": ["恐慌时按计划分批买，比猜底更重要"],
                  "ai_additions": ["可以提前写好分批加仓规则"]},
    "investing": {"event": "NASDAQ 100 单日跌 6%", "fear_level": 8, "emotions": "怕继续跌",
                  "decision": "买入一小部分 QQQ", "reasons": ["长期看好", "估值回落"],
                  "market_mood": "媒体一片恐慌", "plan": "再跌再买",
                  "ai_additions": ["历史上单日大跌后一年的回报多数为正，但不保证"]},
    "kid": {"what_happened": "一起搭乐高", "quotes": ["妈妈，这是火箭！"],
            "growth": ["第一次自己看图纸"], "feelings": "很骄傲", "ai_additions": []},
    "other": {"items": [], "ai_additions": []},
    "open_questions": [],
}


class ChunkStream(httpx.AsyncByteStream):
    """像真实 vLLM 一样分块返回 SSE。"""

    def __init__(self, chunks):
        self.chunks = chunks

    async def __aiter__(self):
        for c in self.chunks:
            yield c


class FakeVLLM:
    def __init__(self):
        self.requests: list[dict] = []
        self.compose_json = COMPOSE_JSON

    def handler(self, request: httpx.Request) -> httpx.Response:
        assert request.url.host == "vllm.test", f"unexpected host {request.url}"
        body = json.loads(request.content or b"{}")
        self.requests.append({"path": request.url.path, "body": body,
                              "auth": request.headers.get("authorization")})
        if request.url.path.endswith("/embeddings"):
            inputs = body["input"]
            vecs = [[float(("投资" in t) or ("QQQ" in t)), float("孩子" in t or "乐高" in t), 0.1] for t in inputs]
            return httpx.Response(200, json={"data": [{"index": i, "embedding": v} for i, v in enumerate(vecs)]})
        system = body["messages"][0]["content"] if body["messages"][0]["role"] == "system" else ""
        if "日记整理助手" in system:
            content = "好的，结果如下：\n" + json.dumps(self.compose_json, ensure_ascii=False)
        elif "日记教练" in system:
            content = '<think>想一想</think>```json\n{"questions": ["当时恐慌打几分？", "孩子原话是什么？", "学到了什么？", "多余的第四个"]}\n```'
        else:
            content = "根据日记，你当时很紧张 [[Journal/2026-11-05#投资记录]]"
        if body.get("stream"):
            chunks = [
                {"choices": [{"delta": {"content": "你好"}, "index": 0}]},
                {"choices": [{"delta": {"content": "！"}, "index": 0, "finish_reason": "stop"}]},
            ]
            sse = "".join(f"data: {json.dumps(c, ensure_ascii=False)}\n\n" for c in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, stream=ChunkStream([sse.encode()[:20], sse.encode()[20:]]),
                                  headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={
            "id": "x", "object": "chat.completion", "model": body.get("model"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]})


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    v = tmp_path / "vault"
    (v / "Journal").mkdir(parents=True)
    (v / ".obsidian").mkdir()
    (v / ".obsidian" / "secret.md").write_text("投资 不应该被索引", "utf-8")
    (v / "Journal" / "2026-11-05.md").write_text(
        "---\ndate: 2026-11-05\n---\n# 2026-11-05\n\n## 投资记录\n\n- 纳斯达克大跌，我很恐慌，恐慌 8 分，硬着头皮买了 QQQ。\n",
        "utf-8")
    (v / "Notes.md").write_text("# 读书\n\n## 孩子\n\n和孩子一起读绘本，孩子说想当宇航员。\n", "utf-8")
    return v


@pytest.fixture
def settings(vault: Path, tmp_path: Path) -> Settings:
    return Settings(vllm_base_url="http://vllm.test/v1", vllm_model="test-model", vllm_api_key="",
                    vault_path=vault, data_dir=tmp_path / "data", inbox_settle_seconds=0)


@pytest.fixture
def fake() -> FakeVLLM:
    return FakeVLLM()


@pytest.fixture
def client(settings, fake):
    app = create_app(settings, transport=httpx.MockTransport(fake.handler))
    with TestClient(app) as c:
        yield c
