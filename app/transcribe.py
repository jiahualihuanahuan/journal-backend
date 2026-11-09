"""语音转文字：local（faster-whisper，默认 CPU）或 openai（任意 OpenAI 兼容 /audio/transcriptions）。"""
from __future__ import annotations

import asyncio
from pathlib import Path

AUDIO_EXT = {".m4a", ".mp3", ".wav", ".ogg", ".oga", ".opus", ".webm", ".flac", ".aac", ".amr", ".3gp", ".mp4"}


class TranscribeDisabled(RuntimeError):
    pass


class Transcriber:
    def __init__(self, settings, http_client=None):
        self.s = settings
        self.http = http_client
        self._model = None
        self._lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return self.s.whisper_backend in {"local", "openai"}

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel  # 可选依赖
            self._model = WhisperModel(self.s.whisper_model, device=self.s.whisper_device,
                                       compute_type=self.s.whisper_compute_type)
        return self._model

    def _run_local(self, path: Path) -> str:
        model = self._load()
        lang = self.s.whisper_language or None
        if lang == "auto":
            lang = None
        segments, _info = model.transcribe(str(path), language=lang, vad_filter=True,
                                           initial_prompt=self.s.whisper_initial_prompt or None)
        return "".join(seg.text for seg in segments).strip()

    async def transcribe(self, path: Path) -> str:
        if self.s.whisper_backend == "local":
            async with self._lock:  # 一次只跑一个，CPU 吃不消并发
                return await asyncio.to_thread(self._run_local, path)
        if self.s.whisper_backend == "openai":
            headers = {"Authorization": f"Bearer {self.s.whisper_api_key}"} if self.s.whisper_api_key else {}
            data = {"model": self.s.whisper_model, "response_format": "json"}
            if self.s.whisper_language and self.s.whisper_language != "auto":
                data["language"] = self.s.whisper_language
            if self.s.whisper_initial_prompt:
                data["prompt"] = self.s.whisper_initial_prompt
            with open(path, "rb") as f:
                r = await self.http.post(f"{self.s.whisper_base_url}/audio/transcriptions",
                                         data=data, files={"file": (path.name, f)}, headers=headers)
            r.raise_for_status()
            return (r.json().get("text") or "").strip()
        raise TranscribeDisabled("WHISPER_BACKEND=off，语音转写未开启")
