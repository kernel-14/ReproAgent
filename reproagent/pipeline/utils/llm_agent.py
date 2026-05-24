"""OpenAI-compatible LLM executor for reproagent agent workflow stages."""

from __future__ import annotations

import os
import time

from langchain_core.messages import HumanMessage, SystemMessage

from reproagent.pipeline.config import create_node_model

from .agent_executor import AgentExecutor


class LLMWorkflowAgent(AgentExecutor):
    """Thin adapter that executes reproagent prompts through the configured node LLM."""

    def __init__(self, node_name: str = "agent_workflow", system_prompt: str | None = None):
        self.node_name = node_name
        self.system_prompt = system_prompt or (
            "You are the reproagent workflow agent. Follow the prompt exactly and return only the requested content."
        )

    def execute(self, prompt: str, context: dict, output_mode: str = "code") -> str:
        del context
        model = create_node_model(self.node_name)
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt),
        ]
        use_streaming = os.getenv("PAPERBENCH_REPRO_NODE_STREAMING", "1").strip().lower() not in {
            "0",
            "false",
            "no",
        }
        if use_streaming:
            chunks: list[str] = []
            try:
                for chunk in model.stream(messages):
                    chunks.append(self._content_to_text(getattr(chunk, "content", "")))
                return self._normalize_output("".join(chunks), output_mode=output_mode)
            except Exception as exc:
                allow_fallback = os.getenv(
                    "PAPERBENCH_REPRO_STREAM_FALLBACK_TO_INVOKE",
                    "1",
                ).strip().lower() not in {"0", "false", "no"}
                if not allow_fallback:
                    raise
                if not isinstance(exc, NotImplementedError) and os.getenv(
                    "PAPERBENCH_REPRO_LOG_STREAM_FALLBACK",
                    "",
                ).strip():
                    print(
                        f"[reproagent] streaming generation failed; retrying non-streaming invoke: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
        max_attempts = max(1, int(os.getenv("PAPERBENCH_REPRO_NODE_INVOKE_RETRIES", "3") or "3"))
        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = model.invoke(messages)
                return self._normalize_output(getattr(response, "content", str(response)), output_mode=output_mode)
            except Exception as exc:
                last_exc = exc
                if attempt >= max_attempts:
                    break
                delay = min(30.0, 2.0 * attempt)
                if os.getenv("PAPERBENCH_REPRO_LOG_STREAM_FALLBACK", "").strip():
                    print(
                        f"[reproagent] non-streaming generation invoke failed; retrying {attempt}/{max_attempts}: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                time.sleep(delay)
        assert last_exc is not None
        raise last_exc

    def _content_to_text(self, content) -> str:
        """Normalize one streamed chat chunk into text."""
        if isinstance(content, list):
            return "".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict)
            )
        return str(content or "")

    def _normalize_output(self, content, output_mode: str) -> str:
        """Normalize chat-model output into plain text or plain code."""
        text = self._content_to_text(content).strip()

        if output_mode != "code":
            return text

        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3:
                text = "\n".join(lines[1:-1]).strip()
                if text.startswith("python"):
                    text = text[len("python"):].lstrip()
        return text
