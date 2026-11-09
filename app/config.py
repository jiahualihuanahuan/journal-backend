"""所有配置都来自环境变量或 .env 文件，代码里没有任何密钥。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


def _bool(v: str | None, default: bool = False) -> bool:
    if v is None or v == "":
        return default
    return v.strip().lower() in {"1", "true", "yes", "on", "y"}


def _int(v: str | None, default: int) -> int:
    try:
        return int(v) if v not in (None, "") else default
    except ValueError:
        return default


def _list(v: str | None, default: list[str]) -> list[str]:
    if v is None or v.strip() == "":
        return default
    return [x.strip() for x in v.split(",") if x.strip()]


@dataclass
class Settings:
    # vLLM (OpenAI-compatible)
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model: str = "Qwen/Qwen2.5-7B-Instruct-AWQ"
    vllm_api_key: str = ""
    vllm_timeout: int = 300
    vllm_json_mode: bool = False          # 发送 response_format=json_object
    vllm_disable_thinking: bool = False   # Qwen3 等：发送 chat_template_kwargs.enable_thinking=false
    temperature: float = 0.4

    # Vault
    vault_path: Path = Path("./vault")
    journal_dir: str = "Journal"
    inbox_subdir: str = "Inbox"
    language: str = "zh"
    keep_raw: bool = True
    index_exclude: list[str] = field(default_factory=lambda: [".obsidian", ".trash", ".git"])

    # Retrieval
    embed_backend: str = "none"   # none | openai | local
    embed_base_url: str = ""
    embed_model: str = "BAAI/bge-m3"
    embed_api_key: str = ""
    embed_device: str = "cpu"
    search_top_k: int = 8
    context_max_chars: int = 6000

    # Whisper
    whisper_backend: str = "off"   # off | local | openai
    whisper_model: str = "small"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_language: str = "zh"
    whisper_base_url: str = ""
    whisper_api_key: str = ""
    whisper_initial_prompt: str = "以下是普通话的日记录音，可能夹杂 English 单词，请使用简体中文和标点。"

    # Inbox watcher / scheduler
    inbox_enabled: bool = False
    inbox_poll_seconds: int = 30
    inbox_settle_seconds: int = 15
    inbox_skip_followups: bool = False
    review_digest_enabled: bool = False

    # Server
    backend_api_key: str = ""
    cors_origins: list[str] = field(default_factory=lambda: [
        "app://obsidian.md", "capacitor://localhost", "http://localhost"])
    data_dir: Path = Path("./data")
    prompts_dir: Path = Path(__file__).resolve().parent.parent / "prompts"

    @property
    def journal_path(self) -> Path:
        return self.vault_path / self.journal_dir

    @property
    def inbox_path(self) -> Path:
        return self.journal_path / self.inbox_subdir

    @classmethod
    def from_env(cls, env_file: str | None = ".env") -> "Settings":
        if load_dotenv and env_file and Path(env_file).exists():
            load_dotenv(env_file, override=False)
        e = os.environ.get
        d = cls()
        return cls(
            vllm_base_url=e("VLLM_BASE_URL", d.vllm_base_url).rstrip("/"),
            vllm_model=e("VLLM_MODEL", d.vllm_model),
            vllm_api_key=e("VLLM_API_KEY", ""),
            vllm_timeout=_int(e("VLLM_TIMEOUT"), d.vllm_timeout),
            vllm_json_mode=_bool(e("VLLM_JSON_MODE"), False),
            vllm_disable_thinking=_bool(e("VLLM_DISABLE_THINKING"), False),
            temperature=float(e("TEMPERATURE", d.temperature)),
            vault_path=Path(e("VAULT_PATH", str(d.vault_path))).expanduser(),
            journal_dir=e("JOURNAL_DIR", d.journal_dir).strip("/\\") or "Journal",
            inbox_subdir=e("INBOX_SUBDIR", d.inbox_subdir).strip("/\\") or "Inbox",
            language=e("LANGUAGE", d.language),
            keep_raw=_bool(e("KEEP_RAW"), True),
            index_exclude=_list(e("INDEX_EXCLUDE"), d.index_exclude),
            embed_backend=e("EMBED_BACKEND", "openai" if e("EMBED_BASE_URL") else "none").lower(),
            embed_base_url=e("EMBED_BASE_URL", "").rstrip("/"),
            embed_model=e("EMBED_MODEL", d.embed_model),
            embed_api_key=e("EMBED_API_KEY", ""),
            embed_device=e("EMBED_DEVICE", "cpu"),
            search_top_k=_int(e("SEARCH_TOP_K"), d.search_top_k),
            context_max_chars=_int(e("CONTEXT_MAX_CHARS"), d.context_max_chars),
            whisper_backend=e("WHISPER_BACKEND", d.whisper_backend).lower(),
            whisper_model=e("WHISPER_MODEL", d.whisper_model),
            whisper_device=e("WHISPER_DEVICE", d.whisper_device),
            whisper_compute_type=e("WHISPER_COMPUTE_TYPE", d.whisper_compute_type),
            whisper_language=e("WHISPER_LANGUAGE", d.whisper_language),
            whisper_base_url=e("WHISPER_BASE_URL", "").rstrip("/"),
            whisper_api_key=e("WHISPER_API_KEY", ""),
            whisper_initial_prompt=e("WHISPER_INITIAL_PROMPT", d.whisper_initial_prompt),
            inbox_enabled=_bool(e("INBOX_ENABLED"), False),
            inbox_poll_seconds=_int(e("INBOX_POLL_SECONDS"), d.inbox_poll_seconds),
            inbox_settle_seconds=_int(e("INBOX_SETTLE_SECONDS"), d.inbox_settle_seconds),
            inbox_skip_followups=_bool(e("INBOX_SKIP_FOLLOWUPS"), False),
            review_digest_enabled=_bool(e("REVIEW_DIGEST_ENABLED"), False),
            backend_api_key=e("BACKEND_API_KEY", ""),
            cors_origins=_list(e("CORS_ORIGINS"), d.cors_origins),
            data_dir=Path(e("DATA_DIR", str(d.data_dir))).expanduser(),
            prompts_dir=Path(e("PROMPTS_DIR", str(d.prompts_dir))).expanduser(),
        )
