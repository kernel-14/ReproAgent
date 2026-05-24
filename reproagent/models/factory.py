"""Minimal chat-model factory for the standalone reproduction pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class _LocalResponse:
    content: str


class LocalFallbackChatModel:
    """Deterministic model used when no external LangChain model is configured."""

    def invoke(self, messages: list[Any]) -> _LocalResponse:
        rendered = "\n\n".join(str(getattr(item, "content", item)) for item in messages)
        payload = {
            "summary": "local_fallback_model_used",
            "notes": [
                "No external model factory is configured in this standalone pipeline.",
                "Use PAPERBENCH_REPRO_STRUCTURED_STAGE_BACKEND/codex/claude settings for real LLM execution.",
            ],
            "prompt_preview": rendered[:1200],
        }
        return _LocalResponse(json.dumps(payload, ensure_ascii=False))


def create_chat_model(*_: Any, **__: Any) -> LocalFallbackChatModel:
    """Return a deterministic local fallback chat model."""
    return LocalFallbackChatModel()
