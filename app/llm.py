"""vLLM（OpenAI 兼容接口）客户端。"""
from __future__ import annotations

import json
import re
from typing import Any, AsyncIterator

import httpx

THINK_RE = re.compile(r"<think>.*?</think>", re.S)


class LLMError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"upstream error {status}: {body[:500]}")
        self.status = status
        self.body = body


def strip_think(text: str) -> str:
    text = THINK_RE.sub("", text or "")
    if "</think>" in text:  # 只有结束标签的情况
        text = text.split("</think>", 1)[1]
    return text.strip()


def extract_json(text: str) -> Any:
    """尽量从模型输出里解析出 JSON（处理 ```json 代码块、前后废话等）。"""
    text = strip_think(text)
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    candidates = [fence.group(1)] if fence else []
    candidates.append(text)
    for c in candidates:
        c = c.strip()
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            pass
        for open_, close in (("{", "}"), ("[", "]")):
            s, e = c.find(open_), c.rfind(close)
            if s != -1 and e > s:
                try:
                    return json.loads(c[s:e + 1])
                except json.JSONDecodeError:
                    continue
    raise ValueError("model did not return valid JSON")


class LLMClient:
    def __init__(self, settings, transport: httpx.AsyncBaseTransport | None = None):
        self.s = settings
        self.client = httpx.AsyncClient(timeout=settings.vllm_timeout, transport=transport)

    async def aclose(self):
        await self.client.aclose()

    def _headers(self, key: str = "") -> dict:
        h = {"Content-Type": "application/json"}
        key = key or self.s.vllm_api_key
        if key:
            h["Authorization"] = f"Bearer {key}"
        return h

    def _payload(self, payload: dict) -> dict:
        p = dict(payload)
        p["model"] = self.s.vllm_model
        if self.s.vllm_disable_thinking:
            p.setdefault("chat_template_kwargs", {"enable_thinking": False})
        return p

    async def chat(self, messages: list[dict], json_mode: bool = False, **kw) -> str:
        payload = {"messages": messages, "temperature": kw.pop("temperature", self.s.temperature), **kw}
        if json_mode and self.s.vllm_json_mode:
            payload["response_format"] = {"type": "json_object"}
        r = await self.client.post(f"{self.s.vllm_base_url}/chat/completions",
                                   json=self._payload(payload), headers=self._headers())
        if r.status_code >= 400:
            raise LLMError(r.status_code, r.text)
        data = r.json()
        return strip_think(data["choices"][0]["message"].get("content") or "")

    async def chat_json(self, messages: list[dict], retries: int = 1, **kw) -> Any:
        last_err: Exception | None = None
        msgs = list(messages)
        for _ in range(retries + 1):
            text = await self.chat(msgs, json_mode=True, **kw)
            try:
                return extract_json(text)
            except ValueError as e:
                last_err = e
                msgs = msgs + [{"role": "assistant", "content": text},
                               {"role": "user", "content": "请只输出合法的 JSON，不要有任何其他文字。"}]
        raise last_err  # type: ignore[misc]

    async def proxy(self, payload: dict) -> httpx.Response:
        return await self.client.post(f"{self.s.vllm_base_url}/chat/completions",
                                      json=self._payload(payload), headers=self._headers())

    async def proxy_stream(self, payload: dict) -> httpx.Response:
        req = self.client.build_request("POST", f"{self.s.vllm_base_url}/chat/completions",
                                        json=self._payload(payload), headers=self._headers())
        return await self.client.send(req, stream=True)

    @staticmethod
    async def iter_stream(resp: httpx.Response) -> AsyncIterator[bytes]:
        try:
            async for chunk in resp.aiter_raw():
                yield chunk
        finally:
            await resp.aclose()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        url = f"{self.s.embed_base_url}/embeddings"
        r = await self.client.post(url, json={"model": self.s.embed_model, "input": texts},
                                   headers=self._headers(self.s.embed_api_key or self.s.vllm_api_key))
        if r.status_code >= 400:
            raise LLMError(r.status_code, r.text)
        data = sorted(r.json()["data"], key=lambda d: d.get("index", 0))
        return [d["embedding"] for d in data]
