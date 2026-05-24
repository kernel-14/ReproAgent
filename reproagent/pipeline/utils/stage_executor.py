"""Structured JSON stage invocation helpers for reproagent."""

import json
import os
import queue
import threading
import time

from reproagent.pipeline.config import (
    get_codegen_config,
    get_structured_stage_config,
    _slot_env,
)
from reproagent.pipeline.schemas import PaperBenchReproState
from reproagent.pipeline.utils.claude_sdk_wrapper import ClaudeSDKWrapper
from reproagent.pipeline.utils.codex_wrapper import CodexWrapper, aggregate_codex_usage
from reproagent.utils.structured_output import parse_structured_output

from . import memory as run_memory
from .artifact_writer import register_existing_file
from .run_context import _build_agent_context, _get_output_dir, _json_default

_STAGE_INVOCATION_USAGES: list[dict] = []
_STRUCTURED_STAGE_PARSE_RETRY_SUFFIX = (
    "\n\nPrevious response was not parseable as the required JSON schema. "
    "Retry once. Return ONLY one strict JSON object matching the schema, with no markdown fences or prose."
)


def _reset_stage_invocation_usages() -> None:
    _STAGE_INVOCATION_USAGES.clear()


def _consume_stage_invocation_usage_summary() -> dict:
    summary = aggregate_codex_usage(_STAGE_INVOCATION_USAGES)
    _STAGE_INVOCATION_USAGES.clear()
    return summary


def _structured_stage_backend(stage_name: str) -> str:
    """Resolve the backend used by one structured JSON stage."""
    config = get_structured_stage_config()
    rendered_stage = str(stage_name or "").strip()
    if rendered_stage.startswith("repair_"):
        return str(config.repair_backend or config.default_backend or "llm").strip()
    return str(config.default_backend or "llm").strip()


def _invoke_with_hard_timeout(callable_obj, *, timeout_seconds: float, stage_name: str):
    """Run a blocking provider call with a process-local wall-clock timeout."""
    if timeout_seconds <= 0:
        return callable_obj()
    result_queue: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=1)

    def _target() -> None:
        try:
            result_queue.put(("result", callable_obj()))
        except BaseException as exc:
            result_queue.put(("error", exc))

    worker = threading.Thread(
        target=_target,
        name=f"reproagent_structured_stage_{stage_name}",
        daemon=True,
    )
    worker.start()
    try:
        kind, payload = result_queue.get(timeout=timeout_seconds)
    except queue.Empty as exc:
        raise TimeoutError(f"{stage_name} exceeded structured stage timeout of {timeout_seconds:.1f}s") from exc
    if kind == "error":
        raise payload
    return payload


def _content_to_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
                continue
            parts.append(str(getattr(item, "text", "") or getattr(item, "content", "") or item))
        return "\n".join(part for part in parts if part)
    return str(content)


def _openai_chat_response_text(response) -> str:
    if isinstance(response, str):
        return response
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices")
    if choices:
        choice = choices[0]
        message = getattr(choice, "message", None)
        if message is None and isinstance(choice, dict):
            message = choice.get("message")
        if message is not None:
            content = getattr(message, "content", None)
            if content is None and isinstance(message, dict):
                content = message.get("content")
            return _content_to_text(content)
        text = getattr(choice, "text", None)
        if text is None and isinstance(choice, dict):
            text = choice.get("text")
        if text is not None:
            return _content_to_text(text)
    return str(response)


def _openai_stream_response_text(stream) -> str:
    if isinstance(stream, str):
        parts: list[str] = []
        for line in stream.splitlines():
            rendered = line.strip()
            if not rendered.startswith("data:"):
                continue
            data = rendered.removeprefix("data:").strip()
            if not data or data == "[DONE]":
                continue
            try:
                payload = json.loads(data)
            except Exception:
                continue
            for choice in list(payload.get("choices", []) or []):
                delta = dict(choice.get("delta", {}) or {})
                content = delta.get("content")
                if content:
                    parts.append(str(content))
        return "".join(parts) or stream

    parts: list[str] = []
    for chunk in stream:
        choices = getattr(chunk, "choices", None)
        if choices is None and isinstance(chunk, dict):
            choices = chunk.get("choices")
        for choice in list(choices or []):
            delta = getattr(choice, "delta", None)
            if delta is None and isinstance(choice, dict):
                delta = choice.get("delta")
            content = None
            if delta is not None:
                content = getattr(delta, "content", None)
                if content is None and isinstance(delta, dict):
                    content = delta.get("content")
            if content:
                parts.append(_content_to_text(content))
    return "".join(parts)


def _invoke_openai_chat_text(system: str, user: str) -> str:
    """Call an OpenAI-compatible chat endpoint and preserve raw-string provider responses."""
    from openai import OpenAI

    config = get_structured_stage_config()
    api_key = (
        config.llm_api_key
        or _slot_env("OPENAI_API_KEY")
        or _slot_env("DF_API_KEY")
        or os.getenv("DF_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    base_url = (
        config.llm_base_url
        or _slot_env("OPENAI_BASE_URL")
        or _slot_env("DF_API_URL")
        or os.getenv("DF_API_URL")
        or os.getenv("OPENAI_BASE_URL")
        or None
    )
    kwargs = {
        "model": config.llm_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": config.llm_temperature,
        "max_tokens": config.llm_max_tokens,
    }
    client_kwargs = {
        "api_key": api_key,
        "base_url": base_url,
        "max_retries": config.llm_max_retries,
    }
    if config.llm_timeout_seconds > 0:
        client_kwargs["timeout"] = config.llm_timeout_seconds
    client = OpenAI(**client_kwargs)
    stream_enabled = os.getenv("PAPERBENCH_REPRO_STRUCTURED_STREAM", "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }
    if not stream_enabled:
        response = client.chat.completions.create(**kwargs, stream=False)
        return _openai_chat_response_text(response)
    try:
        response = client.chat.completions.create(**kwargs, stream=True)
        text = _openai_stream_response_text(response)
        if text:
            return text
    except Exception:
        if os.getenv("PAPERBENCH_REPRO_STRUCTURED_STREAM_FALLBACK_TO_INVOKE", "1").strip().lower() in {"0", "false", "no"}:
            raise
    response = client.chat.completions.create(**kwargs, stream=False)
    return _openai_chat_response_text(response)


def _invoke_openai_chat_text_with_retries(stage_name: str, system: str, user: str) -> str:
    max_attempts = max(1, int(os.getenv("PAPERBENCH_REPRO_STRUCTURED_INVOKE_RETRIES", "3") or "3"))
    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return _invoke_openai_chat_text(system, user)
        except BaseException as exc:
            last_exc = exc
            if attempt >= max_attempts:
                break
            time.sleep(min(30.0, 2.0 * attempt))
    assert last_exc is not None
    raise last_exc


def _parse_structured_stage_response(
    response_text: str,
    *,
    stage_name: str,
    schema_name: str,
    state: PaperBenchReproState | None,
    attempt: int,
) -> dict | list:
    try:
        return parse_structured_output(response_text, schema_name=schema_name)
    except Exception as exc:
        if state is not None:
            run_memory.record_structured_stage_parse_failure(
                state,
                stage_name=stage_name,
                schema_name=schema_name,
                error=exc,
                raw_response=response_text,
                attempt=attempt,
            )
        raise


def _invoke_json_stage(
    stage_name: str,
    schema_name: str,
    system: str,
    user: str,
    state: PaperBenchReproState | None = None,
) -> dict:
    """Invoke one structured planning stage without fallback."""
    if state is not None:
        memory_block = run_memory.get_run_memory_prompt(state, max_chars=4000)
        if memory_block:
            user = f"{memory_block}\n\n{user}"
    backend = _structured_stage_backend(stage_name)
    parse_error: Exception | None = None
    for attempt in (1, 2):
        attempt_user = user if attempt == 1 else f"{user}{_STRUCTURED_STAGE_PARSE_RETRY_SUFFIX}"
        if backend == "codex_cli":
            config = get_codegen_config()
            planner = CodexWrapper(
                config.codex_cli_path,
                model=getattr(config, "codex_model", None),
                model_provider=getattr(config, "codex_model_provider", None),
                base_url=getattr(config, "codex_base_url", None),
                reasoning_effort=getattr(config, "codex_reasoning_effort", None),
            )
            prompt = (
                f"{system}\n\n"
                f"{attempt_user}\n\n"
                "Return ONLY valid JSON."
            )
            response_text = planner.execute(prompt, _build_agent_context(), output_mode="text")
            if isinstance(planner.last_result, dict):
                _STAGE_INVOCATION_USAGES.append(dict(planner.last_result.get("usage") or {}))
        elif backend == "claude_cli":
            config = get_codegen_config()
            planner = ClaudeSDKWrapper(
                config.claude_cli_path,
                model=getattr(config, "claude_model", None),
                effort=getattr(config, "claude_effort", None),
            )
            prompt = (
                f"{system}\n\n"
                f"{attempt_user}\n\n"
                "Return ONLY valid JSON."
            )
            response_text = planner.execute(prompt, _build_agent_context(), output_mode="text")
        else:
            timeout_seconds = float(get_structured_stage_config().llm_timeout_seconds or 0)
            response_text = _invoke_with_hard_timeout(
                lambda: _invoke_openai_chat_text_with_retries(stage_name, system, attempt_user),
                timeout_seconds=timeout_seconds,
                stage_name=stage_name,
            )
        try:
            parsed = _parse_structured_stage_response(
                response_text,
                stage_name=stage_name,
                schema_name=schema_name,
                state=state,
                attempt=attempt,
            )
            break
        except Exception as exc:
            parse_error = exc
            if attempt >= 2:
                raise
            time.sleep(1.0)
    else:
        assert parse_error is not None
        raise parse_error
    if not isinstance(parsed, dict):
        raise ValueError(f"{stage_name} did not return a JSON object")
    return parsed


def _structured_stage_backend_label() -> dict[str, str]:
    """Report the structured-stage backend policy currently in effect."""
    config = get_structured_stage_config()
    return {
        "default_backend": str(config.default_backend or "llm"),
        "repair_backend": str(config.repair_backend or config.default_backend or "llm"),
    }

def _write_stage_output(state: PaperBenchReproState, filename: str, payload: object) -> None:
    """Persist one structured stage artifact under the run output directory."""
    if not state.run_id:
        return
    output_dir = _get_output_dir(state)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_payload = payload.model_dump(mode="json") if hasattr(payload, "model_dump") else payload
    (output_dir / filename).write_text(
        json.dumps(json_payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    register_existing_file(
        output_dir / filename,
        run_dir=output_dir,
        logical_name=filename.rsplit(".", 1)[0],
        kind="output",
        stage=filename.rsplit(".", 1)[0],
        authority="derived",
    )
