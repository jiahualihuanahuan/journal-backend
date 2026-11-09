"""从 prompts/<lang>/ 读取提示词。每次调用都重新读取，改完不用重启。"""
from __future__ import annotations

from pathlib import Path


def load_prompt(settings, name: str, **vars) -> str:
    base: Path = settings.prompts_dir
    for lang in (settings.language, "zh"):
        p = base / lang / f"{name}.md"
        if p.exists():
            text = p.read_text(encoding="utf-8")
            for k, v in vars.items():
                text = text.replace("{{" + k + "}}", str(v))
            return text
    raise FileNotFoundError(f"prompt not found: {name}")
