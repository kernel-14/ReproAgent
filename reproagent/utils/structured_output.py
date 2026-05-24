"""Structured output parsing utilities."""

from __future__ import annotations

import json
import re
from typing import Any


_FENCED_BLOCK_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)\s*```", re.DOTALL)
_VALID_JSON_ESCAPE_CHARS = {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}


def _escape_invalid_json_backslashes(text: str) -> str:
    r"""Escape backslashes that are not part of a valid JSON escape sequence.

    Gemini-backed stages occasionally emit raw LaTeX snippets such as
    ``\ell_\infty`` inside JSON strings.  Those are invalid JSON escapes, so we
    normalize only the invalid backslashes while leaving legitimate JSON escape
    sequences intact.
    """
    if "\\" not in text:
        return text
    result: list[str] = []
    i = 0
    length = len(text)
    while i < length:
        ch = text[i]
        if ch != "\\":
            result.append(ch)
            i += 1
            continue
        if i + 1 < length:
            nxt = text[i + 1]
            if nxt == "\\":
                result.append("\\\\")
                i += 2
                continue
            if nxt in _VALID_JSON_ESCAPE_CHARS:
                result.append("\\")
                result.append(nxt)
                i += 2
                continue
        result.append("\\\\")
        i += 1
    return "".join(result)


def _expects_list(schema_name: str | type) -> bool:
    return schema_name is list or (isinstance(schema_name, type) and issubclass(schema_name, list))


def _coerce_expected_type(value: Any, expect_list: bool) -> dict | list | None:
    if expect_list:
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            for key in ("seeds", "items", "results", "data", "output"):
                candidate = value.get(key)
                if isinstance(candidate, list):
                    return candidate
        return None
    return value if isinstance(value, dict) else None


def _try_raw_decode(candidate: str, expect_list: bool) -> dict | list | None:
    decoder = json.JSONDecoder()
    stripped = candidate.strip()
    if not stripped:
        return None
    try:
        parsed, _ = decoder.raw_decode(stripped)
    except json.JSONDecodeError:
        try:
            repaired = _escape_invalid_json_backslashes(stripped)
            parsed, _ = decoder.raw_decode(repaired)
        except json.JSONDecodeError:
            return None
    return _coerce_expected_type(parsed, expect_list)


def _iter_json_candidates(text: str, expect_list: bool):
    preferred_start = "[" if expect_list else "{"
    yield text
    for match in _FENCED_BLOCK_RE.finditer(text):
        yield match.group(1)
    start_chars = (preferred_start,) if preferred_start in text else ()
    fallback_chars = tuple(ch for ch in ("{", "[") if ch != preferred_start and ch in text)
    for start_char in (*start_chars, *fallback_chars):
        start = 0
        while True:
            index = text.find(start_char, start)
            if index == -1:
                break
            yield text[index:]
            start = index + 1


def parse_structured_output(text: str, schema_name: str | type = "output") -> dict | list:
    """Extract a JSON object/list from an LLM response."""
    expect_list = _expects_list(schema_name)
    for candidate in _iter_json_candidates(text or "", expect_list):
        parsed = _try_raw_decode(candidate, expect_list)
        if parsed is not None:
            return parsed
    schema_desc = "list" if expect_list else "dict"
    raise ValueError(f"No valid JSON {schema_desc} found in {schema_name} response")
