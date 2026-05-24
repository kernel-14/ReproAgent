"""PaperBench case loader for the standalone reproduction pipeline."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from reproagent.pipeline.schemas import PaperBenchReproInput


_GITHUB_RE = re.compile(r"https?://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)", re.IGNORECASE)
_ARXIV_RE = re.compile(r"\barXiv[:\s]+(\d{4}\.\d{4,5})(?:v\d+)?\b", re.IGNORECASE)
_REFERENCE_GITHUB_LINE_RE = re.compile(r"^\s*GitHub\s*-\s*(.+?)\s*https?\s*:\s*//\s*github\.com/\s*.+$", re.IGNORECASE)
_BODY_CITATION_RE = re.compile(r"([A-Z][A-Za-z0-9\-\+]{2,}(?:\s+[A-Z][A-Za-z0-9\-\+]{2,}){0,4})\s*\(([^)]*20\d{2}[^)]*)\)")
_CODE_REPOSITORY_SIGNAL_RE = re.compile(
    r"\b(code|github|repo|repository|implementation|official|released|opensourced|open[-\s]?source|baseline)\b",
    re.IGNORECASE,
)
_STRONG_REFERENCE_REPO_SIGNAL_RE = re.compile(
    r"\b("
    r"official\s+(?:code|implementation|repository|repo)"
    r"|source\s+code"
    r"|code\s+(?:for|from|available|released|can\s+be\s+found)"
    r"|implementation\s+(?:of|from|for|is\s+taken\s+from)"
    r"|reference\s+implementations?\s+(?:of|for|from)|codebase\s+(?:for|from|of)"
    r"|github\s+(?:repository|repo|code)"
    r"|repository\s+(?:for|from|of)"
    r"|we\s+(?:use|used|adapt|adapted)\s+[^.]{0,120}\b(?:code|implementations?|repositories|repository|repo)"
    r"|(?:taken|adapted|borrowed)\s+from"
    r")\b",
    re.IGNORECASE,
)
_AUTHORISH_FRAGMENT_RE = re.compile(r"^(?:and\s+)?[A-Z][A-Za-z'`-]+,\s*[A-Z](?:\.[A-Z])?\.?$")
_REGARDING_TITLE_RE = re.compile(
    r"(?:as\s+for|regarding)\s+(?P<title>[^,.\n]+?)(?:\s*\([^)]*\))?\s*,\s*we\s+use",
    re.IGNORECASE,
)
_GENERIC_REFERENCE_TITLES = {
    "baseline approaches",
    "implementation",
    "our code",
    "code",
    "repository",
}
_AMBIGUOUS_REPO_SEARCH_TITLES = {
    "clip",
    "bert",
    "vit",
    "gpt",
    "llama",
    "transformer",
    "imagenet",
}
_REFERENCE_TITLE_BAD_PREFIXES = (
    "arxiv",
    "corr",
    "doi",
    "in ",
    "issn",
    "journal",
    "pp.",
    "proceedings",
    "url",
    "volume",
)
_REFERENCE_CONTEXT_RELEVANCE_RE = re.compile(
    r"\b("
    r"adapted?|adopt(?:ed)?|baseline|benchmark|code|dataset|environment|evaluation|follow(?:ed|ing)?|"
    r"hyper-?parameter|implementation|method|metric|repository|setting|setup|task|we\s+use|we\s+used"
    r")\b",
    re.IGNORECASE,
)
_STRONG_REFERENCE_USAGE_RE = re.compile(
    r"\b("
    r"baseline|baselines|compare(?:d)?|comparison|contrast|costs?|follow(?:ed|ing)?|"
    r"hyper-?parameter|implementation|in\s+particular|same\s+.*?(?:data|setting|setup)|"
    r"state-of-the-art|we\s+(?:use|used|compare|follow|adopt|adapt|edit)"
    r")\b",
    re.IGNORECASE,
)
_WEAK_REFERENCE_CONTEXT_RE = re.compile(
    r"\b(abstract|introduction|related\s+work|copyright|proceedings|conference)\b",
    re.IGNORECASE,
)
_REFERENCE_ENTRY_START_RE = re.compile(
    r"^\s*(?:\[\d+\]\s*|\\bibitem(?:\[[^\]]*\])?\{[^}]*\}\s*)?"
    r"[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'`.-]+(?:,|\s+and\s+|\s+et\s+al\.)",
    re.IGNORECASE,
)
_REFERENCE_AUTHOR_RE = re.compile(
    r"(?:^|[,;]\s+|\band\s+)"
    r"[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'`.-]+,\s*"
    r"(?:[A-Z](?:\.-|\.)\s*){1,4}",
)
_REFERENCE_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}[a-z]?\b", re.IGNORECASE)
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)


def default_paperbench_data_root() -> Path:
    """Return the local PaperBench data root inside this standalone workspace."""
    return Path(__file__).resolve().parents[2] / "paperbench_data"


def resolve_paperbench_case(case: str | Path, data_root: str | Path | None = None) -> Path:
    """Resolve either a PaperBench case id or an explicit case directory."""
    raw = Path(str(case)).expanduser()
    if raw.exists():
        return raw.resolve()
    root = Path(data_root).expanduser() if data_root else default_paperbench_data_root()
    candidate = root / str(case)
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"PaperBench case not found: {case} (data_root={root})")


def _read_text(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def _strip_yaml_scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _load_config(path: Path) -> dict[str, Any]:
    """Load simple PaperBench config.yaml without requiring PyYAML."""
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return dict(payload) if isinstance(payload, dict) else {}
    except Exception:
        payload: dict[str, Any] = {}
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            if key:
                payload[key] = _strip_yaml_scalar(value)
        return payload


def _read_blacklist(path: Path) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for raw_line in _read_text(path).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        normalized = normalize_github_url(line) or line
        if normalized in seen:
            continue
        seen.add(normalized)
        values.append(normalized)
    return values


def normalize_github_url(url: str) -> str:
    """Canonicalize a GitHub URL to https://github.com/owner/repo when possible."""
    match = _GITHUB_RE.search(str(url or "").strip())
    if not match:
        return ""
    owner = match.group(1).strip()
    repo = match.group(2).strip().removesuffix(".git").strip(").,;]")
    if not owner or not repo:
        return ""
    return f"https://github.com/{owner}/{repo}"


def _normalize_text_for_repository_urls(text: str) -> str:
    normalized = str(text or "")
    stopword_pattern = r"(?:and|or|the|a|an|of|in|on|for|from|with|to|by|at|via|as|is|are|this|that|these|those|then|when|while|also|note|but|not|https?)"
    normalized = re.sub(
        r"https?\s*:\s*/\s*/\s*github\.com\s*/\s*",
        "https://github.com/",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"https://github\.com/\s*", "https://github.com/", normalized, flags=re.IGNORECASE)
    normalized = re.sub(
        r"https://github\.com/([A-Za-z0-9_.-]+)\s*/\s*([A-Za-z0-9_.-]+)",
        r"https://github.com/\1/\2",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"https://github\.com/([A-Za-z0-9_.-]+)\s*/\s*([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)?)",
        lambda match: f"https://github.com/{match.group(1)}/{match.group(2).replace(' ', '')}",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        rf"(https://github\.com/[A-Za-z0-9_.-]+/)([A-Za-z0-9_.-]+(?:\s+(?!{stopword_pattern}\b)[a-z0-9][A-Za-z0-9_.-]*)*)",
        lambda match: f"{match.group(1)}{''.join(match.group(2).split())}",
        normalized,
    )
    return normalized.replace("https:// github.com/", "https://github.com/")


def _extract_github_urls(*sources: tuple[str, str]) -> list[dict[str, Any]]:
    urls: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source_name, text in sources:
        normalized_text = _normalize_text_for_repository_urls(text)
        for match in _GITHUB_RE.finditer(normalized_text):
            normalized = normalize_github_url(match.group(0))
            if not normalized or normalized in seen:
                continue
            start = max(0, match.start() - 240)
            end = min(len(normalized_text), match.end() + 240)
            context = " ".join(normalized_text[start:end].split())
            urls.append(
                {
                    "url": normalized,
                    "source": source_name,
                    "context": context[:700],
                }
            )
            seen.add(normalized)
    return urls


def _github_url_has_actionable_reference_signal(item: dict[str, Any]) -> bool:
    """Return true when an explicit GitHub URL is likely a code reference, not a paper artifact link."""
    context = str(item.get("context", "") or "")
    if not context.strip():
        return False
    if re.search(
        r"\b(?:references?\s+include|as\s+for|regarding|we\s+(?:use|used|adapt|adapted))\b",
        context,
        re.IGNORECASE,
    ):
        return True
    return bool(_STRONG_REFERENCE_REPO_SIGNAL_RE.search(context))


def _extract_reference_section(text: str) -> str:
    match = re.search(r"\\section\*\{References\}(.*)$", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r"(?im)^#+\s*references\s*$([\s\S]*)", text)
    if match:
        return match.group(1)
    return ""


def _clean_reference_fragment(text: str) -> str:
    value = " ".join(str(text or "").split())
    value = re.sub(r"\s*-\s+github\.com\.?$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+—\s+github\.com\.?$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^\s*(?:\\bibitem(?:\[[^\]]*\])?\{[^}]*\}|\[\d+\])\s*", "", value)
    value = re.sub(r"\$([^$]+)\$", r"\1", value)
    value = re.sub(r"\\[A-Za-z]+\s*", "", value)
    return value.strip(" .,;:")


def _split_reference_entries(reference_text: str) -> list[str]:
    """Split a References section into bibliography entries.

    PaperBench markdown is produced from PDFs, so entries are usually blank-line
    separated but may also wrap across lines or use bibitem/numbered labels.
    """
    entries: list[str] = []
    current: list[str] = []
    for raw_line in str(reference_text or "").splitlines():
        line = " ".join(raw_line.split())
        if not line:
            if current:
                entries.append(" ".join(current).strip())
                current = []
            continue
        if line.startswith("\\section") or re.match(r"^#{1,6}\s+", line):
            if current:
                entries.append(" ".join(current).strip())
                current = []
            break
        starts_new = bool(_REFERENCE_ENTRY_START_RE.match(line))
        if starts_new and current and (_REFERENCE_YEAR_RE.search(" ".join(current)) or len(" ".join(current)) > 260):
            entries.append(" ".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        entries.append(" ".join(current).strip())
    return [entry for entry in entries if entry]


def _strip_reference_urls(text: str) -> str:
    value = str(text or "")
    value = re.sub(r"https?://\S+", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"\bURL\s+\S+", " ", value, flags=re.IGNORECASE)
    value = _DOI_RE.sub(" ", value)
    return " ".join(value.split())


def _reference_title_sentence(entry: str) -> str:
    cleaned = _clean_reference_fragment(_strip_reference_urls(entry))
    if not cleaned:
        return ""
    author_matches = list(_REFERENCE_AUTHOR_RE.finditer(cleaned))
    if author_matches:
        cleaned = cleaned[author_matches[-1].end() :].lstrip(" ,.;:")
    cleaned = re.sub(r"^(?:et\s+al\.?|and\s+others)\s*[,.;:]?\s*", "", cleaned, flags=re.IGNORECASE)
    boundary_match = re.search(
        r"(?<=[.!?])\s+(?="
        r"In\s+|"
        r"arXiv\b|ArXiv\b|CoRR\b|"
        r"Proceedings\b|Journal\b|Advances\b|"
        r"OpenReview\b|PMLR\b|IEEE\b|Curran\b|Association\b|"
        r"[A-Z][A-Za-z. ]+\s+\d+\s*[:(]"
        r")",
        cleaned,
    )
    if boundary_match:
        return _clean_reference_fragment(cleaned[: boundary_match.start()])
    year_match = _REFERENCE_YEAR_RE.search(cleaned)
    if year_match:
        before_year = cleaned[: year_match.start()].rstrip(" ,.;:")
        sentence_match = re.search(r"^(.+?[.!?])(?:\s+|$)", before_year)
        if sentence_match:
            return _clean_reference_fragment(sentence_match.group(1))
        return _clean_reference_fragment(before_year)
    sentence_match = re.search(r"^(.+?[.!?])(?:\s+|$)", cleaned)
    if sentence_match:
        return _clean_reference_fragment(sentence_match.group(1))
    return _clean_reference_fragment(cleaned)


def _looks_like_reference_paper_title(title: str) -> bool:
    value = _clean_reference_fragment(title)
    if not _looks_like_repo_search_title(value):
        return False
    lowered = value.lower()
    if lowered in _GENERIC_REFERENCE_TITLES or lowered in _AMBIGUOUS_REPO_SEARCH_TITLES:
        return False
    if any(lowered.startswith(prefix) for prefix in _REFERENCE_TITLE_BAD_PREFIXES):
        return False
    tokens = [token for token in re.findall(r"[A-Za-z][A-Za-z0-9+-]*", value) if len(token) > 1]
    if len(tokens) < 2:
        return False
    if sum(1 for token in tokens if token[0].isupper()) == len(tokens) and len(tokens) <= 3:
        return False
    return True


def _reference_entry_citation_keys(entry: str) -> list[str]:
    cleaned = _clean_reference_fragment(entry)
    year_match = _REFERENCE_YEAR_RE.search(cleaned)
    if not year_match:
        return []
    before_year = cleaned[: year_match.start()]
    surnames = re.findall(r"\b([A-Z][A-Za-zÀ-ÖØ-öø-ÿ'`-]+)\b", before_year)
    blocked = {
        "A",
        "An",
        "And",
        "In",
        "International",
        "Journal",
        "Proceedings",
        "The",
    }
    surnames = [name for name in surnames if name not in blocked]
    if not surnames:
        return []
    year = year_match.group(0)[:4]
    keys = [f"{surnames[0].lower()} et al {year}", f"{surnames[0].lower()} {year}"]
    if len(surnames) >= 2:
        keys.append(f"{surnames[0].lower()} and {surnames[1].lower()} {year}")
    return keys


def _distinctive_title_terms(title: str) -> list[str]:
    weak = {
        "accurate",
        "analysis",
        "approach",
        "benchmark",
        "conference",
        "deep",
        "efficient",
        "effective",
        "evaluation",
        "framework",
        "general",
        "large",
        "learning",
        "machine",
        "method",
        "model",
        "models",
        "network",
        "neural",
        "paper",
        "pretrained",
        "simple",
        "system",
        "task",
        "towards",
        "training",
        "using",
        "with",
    }
    terms: list[str] = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9+-]{2,}", title):
        normalized = token.lower()
        if normalized in weak:
            continue
        if normalized not in terms:
            terms.append(normalized)
    return terms


def _body_context_for_reference_entry(body_text: str, entry: str, title: str) -> str:
    keys = _reference_entry_citation_keys(entry)
    title_terms = _distinctive_title_terms(title)[:8]
    contexts: list[str] = []
    seen_contexts: set[str] = set()

    def _append_context(start: int, end: int, *, max_chars: int = 520) -> None:
        context = _clean_reference_fragment(_local_citation_context(body_text, start, end, max_chars=max_chars))
        if not context or context in seen_contexts:
            return
        seen_contexts.add(context)
        contexts.append(context)

    for key in keys:
        parts = key.split()
        if len(parts) < 2:
            continue
        surname = parts[0]
        year = parts[-1]
        citation_pattern = (
            rf"(?<![A-Za-z0-9]){re.escape(surname)}"
            rf"(?:[\s,，.．;；:：()（）\\&-]|et|al|and|[A-Za-z]){{0,120}}?"
            rf"{re.escape(year)}"
        )
        for match in re.finditer(citation_pattern, body_text, flags=re.IGNORECASE):
            _append_context(match.start(), match.end())
            if len(contexts) >= 8:
                break
        if len(contexts) >= 8:
            break
    if title_terms:
        for token in title_terms[:5]:
            for match in re.finditer(rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])", body_text, flags=re.IGNORECASE):
                _append_context(match.start(), match.end(), max_chars=420)
                if len(contexts) >= 10:
                    break
            if len(contexts) >= 10:
                break
    contexts.sort(key=lambda item: _reference_context_strength(title, item), reverse=True)
    return " | ".join(contexts[:4])[:1200]


def _reference_context_strength(title: str, context: str) -> int:
    score = 0
    if _STRONG_REFERENCE_USAGE_RE.search(context):
        score += 20
    elif _REFERENCE_CONTEXT_RELEVANCE_RE.search(context):
        score += 6
    lowered_context = context.lower()
    for term in _distinctive_title_terms(title)[:5]:
        if term in lowered_context:
            score += 4
    if _WEAK_REFERENCE_CONTEXT_RE.search(context) and not _STRONG_REFERENCE_USAGE_RE.search(context):
        score -= 6
    return score


def _reference_search_relevance_score(title: str, context: str) -> int:
    title_terms = {token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9+-]{3,}", title)}
    method_terms = {
        "algorithm",
        "architecture",
        "benchmark",
        "dataset",
        "distillation",
        "environment",
        "evaluation",
        "learning",
        "model",
        "network",
        "optimization",
        "policy",
        "pruning",
        "reinforcement",
        "training",
    }
    score = _reference_context_strength(title, context)
    lowered_title = title.lower()
    score += min(5, len(title_terms.intersection(method_terms)))
    if any(token in lowered_title for token in ("survey", "review", "opportunities and risks")):
        score -= 6
    if any(token in lowered_title for token in ("dataset", "benchmark", "environment")):
        score += 1
    return score


def _extract_reference_bibliography_candidates(
    paper_text: str,
    blacklist_set: set[str],
    *,
    max_candidates: int | None = None,
) -> list[dict[str, Any]]:
    normalized_text = _normalize_text_for_repository_urls(paper_text)
    reference_text = _extract_reference_section(normalized_text)
    if not reference_text:
        return []
    reference_start = normalized_text.find(reference_text)
    body_text = normalized_text[:reference_start] if reference_start >= 0 else normalized_text
    candidates: list[tuple[int, dict[str, Any]]] = []
    seen_titles: set[str] = set()
    for entry in _split_reference_entries(reference_text):
        explicit_urls = [
            item
            for item in _extract_github_urls(("reference_entry", entry))
            if str(item.get("url", "") or "") not in blacklist_set
        ]
        title = _reference_title_sentence(entry)
        if not title and explicit_urls:
            title = _infer_reference_title_from_context(str(explicit_urls[0].get("url", "")), entry)
        if not _looks_like_reference_paper_title(title):
            continue
        title_key = title.lower()
        if title_key in seen_titles:
            continue
        seen_titles.add(title_key)
        context = _body_context_for_reference_entry(body_text, entry, title)
        score = _reference_search_relevance_score(title, context)
        if explicit_urls:
            score += 20
        if score <= 0:
            continue
        payload: dict[str, Any] = {
            "kind": "reference_bibliography_search",
            "title": title,
            "paper_url": "",
            "repository_origin": "unknown",
            "repository_type": "search",
            "reference_role": "paper_reference",
            "source": "paper_references_section",
            "context": context or _clean_reference_fragment(entry)[:700],
        }
        arxiv_match = _ARXIV_RE.search(entry)
        if arxiv_match:
            payload["paper_url"] = f"https://arxiv.org/abs/{arxiv_match.group(1)}"
        doi_match = _DOI_RE.search(entry.upper())
        if doi_match:
            payload["doi"] = doi_match.group(0)
        if explicit_urls:
            payload["repository_url"] = str(explicit_urls[0].get("url", "") or "")
            payload["repository_origin"] = "community"
            payload["repository_type"] = "explicit"
            payload["matched_repository_urls"] = [str(item.get("url", "") or "") for item in explicit_urls]
        candidates.append((score, payload))
    candidates.sort(key=lambda item: (-item[0], str(item[1].get("title", "")).lower()))
    default_limit = int(os.getenv("PAPERBENCH_REPRO_REFERENCE_BIBLIOGRAPHY_CANDIDATE_LIMIT", "24") or "24")
    limit = max_candidates if max_candidates is not None else default_limit
    return [payload for _score, payload in candidates[: max(0, limit)]]


def _looks_like_repo_search_title(title: str) -> bool:
    value = _clean_reference_fragment(title)
    if not value or len(value) < 3 or len(value) > 120:
        return False
    if value.lower().startswith("and "):
        return False
    if value.lower() in _AMBIGUOUS_REPO_SEARCH_TITLES:
        return False
    if "," in value:
        return False
    if _AUTHORISH_FRAGMENT_RE.match(value):
        return False
    tokens = [item.strip(".,;:()[]{}") for item in value.split() if item.strip(".,;:()[]{}")]
    if not tokens:
        return False
    if all(len(token) <= 2 for token in tokens):
        return False
    return any(any(char.isalpha() for char in token) for token in tokens)


def _repo_slug_title(url: str) -> str:
    normalized = normalize_github_url(url)
    if not normalized:
        return ""
    repo = normalized.rstrip("/").rsplit("/", 1)[-1]
    words = [part for part in re.split(r"[-_]+", repo) if part]
    if not words:
        return repo
    return " ".join(word if any(char.isupper() for char in word) else word.capitalize() for word in words)


def _meaningful_reference_title(title: str) -> str:
    value = _clean_reference_fragment(title)
    if not _looks_like_repo_search_title(value):
        return ""
    if value.lower() in _GENERIC_REFERENCE_TITLES:
        return ""
    return value


def _local_citation_context(text: str, start: int, end: int, *, max_chars: int = 360) -> str:
    left = max(0, start - max_chars)
    right = min(len(text), end + max_chars)
    for pattern in ("\n\n", ". ", "; ", "\\end{abstract}"):
        boundary = text.rfind(pattern, left, start)
        if boundary >= 0:
            left = max(left, boundary + len(pattern))
            break
    for pattern in (". ", "; ", "\n\n", "\\section"):
        boundary = text.find(pattern, end, right)
        if boundary >= 0:
            right = min(right, boundary + len(pattern))
            break
    return " ".join(text[left:right].split())


def _has_strong_reference_repo_signal(context: str, blacklist_set: set[str] | None = None) -> bool:
    if not context:
        return False
    explicit_urls = _extract_github_urls(("context", context))
    if explicit_urls:
        normalized_blacklist = {normalize_github_url(item) or str(item or "") for item in (blacklist_set or set())}
        non_blacklisted_urls = [
            item
            for item in explicit_urls
            if str(item.get("url", "") or "") not in normalized_blacklist
        ]
        if non_blacklisted_urls:
            return False
    return bool(_STRONG_REFERENCE_REPO_SIGNAL_RE.search(context))


def _infer_reference_title_from_context(url: str, context: str) -> str:
    normalized_url = normalize_github_url(url)
    text = _normalize_text_for_repository_urls(" ".join(str(context or "").split()))
    if normalized_url:
        url_re = re.escape(normalized_url)
        specific_display_re = re.compile(
            rf"GitHub\s*-\s*[^:]+:\s*(?P<title>.+?)\s*(?:-|—)\s*github\.com\.?\s*{url_re}",
            re.IGNORECASE,
        )
        match = specific_display_re.search(text)
        if match:
            title = _meaningful_reference_title(match.group("title"))
            if title:
                return title

        before, _sep, _after = text.partition(normalized_url)
        near_before = before[-320:]
        matches = list(_REGARDING_TITLE_RE.finditer(near_before))
        for match in reversed(matches):
            title = _meaningful_reference_title(match.group("title"))
            if title:
                return title

    for sentence in re.split(r"(?<=[.])\s+", text):
        if normalized_url and normalized_url not in sentence:
            continue
        matches = list(_REGARDING_TITLE_RE.finditer(sentence))
        for match in reversed(matches):
            title = _meaningful_reference_title(match.group("title"))
            if title:
                return title
    return _repo_slug_title(url)


def _derive_reference_repo_candidates(paper_text: str, blacklist_set: set[str]) -> list[dict[str, Any]]:
    normalized_text = _normalize_text_for_repository_urls(paper_text)
    reference_text = _extract_reference_section(normalized_text)
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    for raw_line in reference_text.splitlines():
        line = " ".join(raw_line.split())
        if not line:
            continue
        if _REFERENCE_GITHUB_LINE_RE.match(line):
            continue
    body_text = normalized_text[: max(0, normalized_text.find(r"\section*{References}"))] if r"\section*{References}" in normalized_text else normalized_text
    for match in _BODY_CITATION_RE.finditer(body_text):
        title = _clean_reference_fragment(match.group(1))
        if not _looks_like_repo_search_title(title):
            continue
        context = _local_citation_context(body_text, match.start(), match.end())
        if not title:
            continue
        if not _has_strong_reference_repo_signal(context, blacklist_set):
            continue
        key = f"body-cite::{title.lower()}"
        if key in seen:
            continue
        candidates.append(
            {
                "kind": "reference_search",
                "title": title,
                "paper_url": "",
                "source": "paper_body_citation",
                "context": context[:700],
            }
        )
        seen.add(key)

    filtered: list[dict[str, Any]] = []
    for item in candidates:
        direct_url = normalize_github_url(str(item.get("paper_url", "") or ""))
        if direct_url and direct_url in blacklist_set:
            continue
        filtered.append(item)
    return filtered[:24]


def _asset_paths(case_dir: Path) -> list[str]:
    assets_dir = case_dir / "assets"
    if not assets_dir.exists():
        return []
    return [
        str(path.relative_to(case_dir))
        for path in sorted(assets_dir.rglob("*"))
        if path.is_file()
    ]



_KNOWN_LIBRARY_REFERENCES = {
    "sbi": {
        "title": "sbi",
        "repository_url": "https://github.com/sbi-dev/sbi",
        "repository_origin": "library",
        "matched_patterns": (
            "sbi python library",
            "sbi library must be used",
            "utilize the sbi library",
            "use the sbi library",
        ),
    },
}


_KNOWN_BENCHMARK_REFERENCES = {
    "gsm8k": {
        "title": "GSM8K / Grade School Math",
        "repository_url": "https://github.com/openai/grade-school-math",
        "repository_origin": "official",
        "repository_type": "official",
        "reference_role": "dataset_benchmark",
        "matched_patterns": (
            "gsm8k",
            "grade school math",
            "grade-school math",
        ),
    },
    "strategyqa": {
        "title": "StrategyQA",
        "repository_url": "https://github.com/eladsegal/strategyqa",
        "repository_origin": "official",
        "repository_type": "official",
        "reference_role": "dataset_benchmark",
        "matched_patterns": (
            "strategyqa",
            "implicit reasoning strategies",
        ),
    },
    "truthfulqa": {
        "title": "TruthfulQA",
        "repository_url": "https://github.com/sylinrl/TruthfulQA",
        "repository_origin": "official",
        "repository_type": "official",
        "reference_role": "dataset_benchmark",
        "matched_patterns": (
            "truthfulqa",
            "truthful qa",
        ),
    },
    "scienceqa": {
        "title": "ScienceQA",
        "repository_url": "https://github.com/lupantech/ScienceQA",
        "repository_origin": "official",
        "repository_type": "official",
        "reference_role": "dataset_benchmark",
        "matched_patterns": (
            "scienceqa",
            "science question answering",
        ),
    },
    "toxigen": {
        "title": "ToxiGen",
        "repository_url": "https://github.com/microsoft/TOXIGEN",
        "repository_origin": "official",
        "repository_type": "official",
        "reference_role": "dataset_benchmark",
        "matched_patterns": (
            "toxigen",
            "toxicity probability",
            "implicit hate speech",
        ),
    },
    "chain-of-thought-hub": {
        "title": "Chain-of-Thought Hub",
        "repository_url": "https://github.com/FranxYao/chain-of-thought-hub",
        "repository_origin": "community",
        "repository_type": "explicit",
        "reference_role": "prompt_protocol",
        "matched_patterns": (
            "chain-of-thought hub",
            "chain of thought hub",
            "franxyao/chain-of-thought-hub",
            "prompt_simple_4_cases",
        ),
    },
}


def _known_library_references(text: str, blacklist_set: set[str]) -> list[dict[str, Any]]:
    lowered = str(text or "").lower()
    references: list[dict[str, Any]] = []
    for key, payload in _KNOWN_LIBRARY_REFERENCES.items():
        patterns = tuple(payload.get("matched_patterns", ()))
        matched = [pattern for pattern in patterns if pattern in lowered]
        if not matched:
            continue
        repository_url = str(payload.get("repository_url", "")).strip()
        if (normalize_github_url(repository_url) or repository_url) in blacklist_set:
            continue
        references.append(
            {
                "kind": "known_library",
                "title": str(payload.get("title", key)),
                "repository_url": repository_url,
                "repository_origin": str(payload.get("repository_origin", "library")),
                "source": "paperbench_known_library_reference",
                "context": "; ".join(matched),
            }
        )
    return references


def _known_benchmark_references(text: str, blacklist_set: set[str]) -> list[dict[str, Any]]:
    lowered = str(text or "").lower()
    references: list[dict[str, Any]] = []
    for key, payload in _KNOWN_BENCHMARK_REFERENCES.items():
        patterns = tuple(payload.get("matched_patterns", ()))
        matched = [pattern for pattern in patterns if pattern in lowered]
        if not matched:
            continue
        repository_url = str(payload.get("repository_url", "")).strip()
        if (normalize_github_url(repository_url) or repository_url) in blacklist_set:
            continue
        references.append(
            {
                "kind": "known_benchmark",
                "title": str(payload.get("title", key)),
                "repository_url": repository_url,
                "repository_origin": str(payload.get("repository_origin", "official")),
                "repository_type": str(payload.get("repository_type", "official")),
                "reference_role": str(payload.get("reference_role", "dataset_benchmark")),
                "source": "paperbench_known_benchmark_reference",
                "context": "matched benchmark/dataset mentions: " + "; ".join(matched[:8]),
            }
        )
    return references


def _dedupe_reference_repo_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in candidates:
        if not isinstance(item, dict):
            continue
        repository_url = normalize_github_url(str(item.get("repository_url", "") or ""))
        title = str(item.get("title", "") or "").strip().lower()
        key = repository_url.lower() if repository_url else title
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _dataset_hints(text: str) -> list[str]:
    hints: list[str] = []
    for raw_line in text.splitlines():
        line = " ".join(raw_line.strip().split())
        if not line:
            continue
        lowered = line.lower()
        if "dataset" in lowered or "download" in lowered or "data " in lowered or "data-" in lowered:
            hints.append(line)
        if len(hints) >= 80:
            break
    return hints


def load_paperbench_case(
    case: str | Path,
    *,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load a PaperBench case directory into a structured payload."""
    case_dir = resolve_paperbench_case(case, data_root)
    if not case_dir.is_dir():
        raise NotADirectoryError(f"PaperBench case path is not a directory: {case_dir}")
    config = _load_config(case_dir / "config.yaml")
    paper_path = case_dir / "paper.md"
    if not paper_path.exists():
        raise FileNotFoundError(f"PaperBench case is missing paper.md: {case_dir}")
    paper_text = _read_text(paper_path)
    addendum_text = _read_text(case_dir / "addendum.md")
    blacklist = _read_blacklist(case_dir / "blacklist.txt")
    blacklist_set = {normalize_github_url(item) or item for item in blacklist}
    github_urls = _extract_github_urls(
        ("paper.md", paper_text),
        ("addendum.md", addendum_text),
    )
    for item in github_urls:
        item["blacklisted"] = item["url"] in blacklist_set
    source_text = "\n".join([paper_text, addendum_text])
    reference_repo_candidates = _dedupe_reference_repo_candidates(
        [
            *_derive_reference_repo_candidates(paper_text, blacklist_set),
            *_extract_reference_bibliography_candidates(paper_text, blacklist_set),
            *_known_library_references(source_text, blacklist_set),
            *_known_benchmark_references(source_text, blacklist_set),
        ]
    )
    case_id = str(config.get("id") or case_dir.name).strip()
    title = str(config.get("title") or "").strip()
    return {
        "case_dir": str(case_dir),
        "case_id": case_id,
        "title": title,
        "config": config,
        "paper_path": str(paper_path.resolve()),
        "paper_text_present": bool(paper_text.strip()),
        "addendum_path": str((case_dir / "addendum.md").resolve()) if (case_dir / "addendum.md").exists() else "",
        "blacklist_path": str((case_dir / "blacklist.txt").resolve()) if (case_dir / "blacklist.txt").exists() else "",
        "addendum_text": addendum_text,
        "blacklist": blacklist,
        "github_urls": github_urls,
        "reference_repo_candidates": reference_repo_candidates,
        "assets": _asset_paths(case_dir),
        "dataset_hints": _dataset_hints("\n".join([paper_text, addendum_text])),
    }


def build_repro_input_from_paperbench(
    case: str | Path,
    *,
    data_root: str | Path | None = None,
    language: str = "zh",
    chunk_max_chars: int = 6000,
    clone_references: bool = True,
    max_iterations: int = 30,
) -> PaperBenchReproInput:
    """Build a PaperBenchReproInput from a PaperBench case."""
    loaded = load_paperbench_case(case, data_root=data_root)
    case_id = str(loaded["case_id"])
    title = str(loaded["title"] or case_id)
    non_blacklisted_urls = [
        item
        for item in loaded["github_urls"]
        if (
            isinstance(item, dict)
            and not bool(item.get("blacklisted"))
            and _github_url_has_actionable_reference_signal(item)
        )
    ]
    idea_references = []
    if clone_references:
        for index, item in enumerate(non_blacklisted_urls, start=1):
            inferred_title = _infer_reference_title_from_context(
                str(item.get("url", "")),
                str(item.get("context", "")),
            )
            idea_references.append(
                {
                    "ref_id": f"paperbench_ref_{index:03d}",
                    "title": inferred_title or f"PaperBench referenced repository {index}",
                    "repository_url": str(item.get("url", "")),
                    "repository_origin": "community",
                    "source": str(item.get("source", "")),
                    "context": str(item.get("context", "")),
                }
            )
        explicit_titles = {
            str(item.get("title", "") or "").strip().lower()
            for item in idea_references
            if str(item.get("title", "") or "").strip()
        }
        next_index = len(idea_references) + 1
        for item in loaded.get("reference_repo_candidates", []):
            if not isinstance(item, dict):
                continue
            ref_title = str(item.get("title", "") or "").strip()
            if not ref_title:
                continue
            repository_url = normalize_github_url(str(item.get("repository_url", "") or ""))
            title_key = ref_title.lower()
            existing_index = next(
                (
                    idx
                    for idx, existing in enumerate(idea_references)
                    if str(existing.get("title", "") or "").strip().lower() == title_key
                ),
                None,
            )
            if existing_index is not None and not repository_url:
                continue
            reference_payload = {
                "ref_id": (
                    str(idea_references[existing_index].get("ref_id", ""))
                    if existing_index is not None
                    else f"paperbench_ref_{next_index:03d}"
                ),
                "title": ref_title,
                "paper_path": "" if repository_url else str(loaded["paper_path"]),
                "paper_url": str(item.get("paper_url", "") or ""),
                "repository_origin": str(item.get("repository_origin", "") or "unknown"),
                "repository_type": str(item.get("repository_type", "") or "unknown"),
                "reference_role": str(item.get("reference_role", "") or ""),
                "source": str(item.get("source", "") or "paper_references"),
                "context": str(item.get("context", "") or ""),
                "search_only": not bool(repository_url),
            }
            if repository_url:
                reference_payload["repository_url"] = repository_url
            if existing_index is not None:
                idea_references[existing_index] = {**idea_references[existing_index], **reference_payload}
            else:
                idea_references.append(reference_payload)
                explicit_titles.add(title_key)
                next_index += 1
    github_summaries = [
        {
            "url": item.get("url", ""),
            "source": item.get("source", ""),
            "blacklisted": bool(item.get("blacklisted")),
            "context": item.get("context", ""),
        }
        for item in loaded["github_urls"]
        if isinstance(item, dict)
    ]
    experiment_design = {
        "task_family": "paperbench_code_reproduction",
        "code_only": True,
        "download_policy": "code_only_no_external_materialization",
        "paperbench_title": title,
        "paperbench_blacklist": loaded["blacklist"],
        "paperbench_constraints": [
            "Use paper.md as the primary input source.",
            "Use addendum.md as binding implementation clarification when present.",
            (
                "Generate a faithful, complete, judgeable standalone code repository. The generator must "
                "not execute expensive training/evaluation runs, but the repository must still implement "
                "the full training and evaluation code paths required by the paper."
            ),
            (
                "Smoke or dry-run modes may validate wiring, but they must exercise real implementation "
                "surfaces and cannot replace method, training, refinement, or evaluation code."
            ),
            "Do not use blacklisted repositories as source code or reference implementations.",
        ],
        "paperbench": {
            "title": title,
            "paper_path": loaded["paper_path"],
            "addendum_path": loaded["addendum_path"],
            "blacklist_path": loaded["blacklist_path"],
            "addendum_text": loaded["addendum_text"],
            "blacklist": loaded["blacklist"],
            "github_references": github_summaries,
            "reference_repo_candidates": loaded.get("reference_repo_candidates", []),
            "assets": loaded["assets"],
            "dataset_hints": loaded["dataset_hints"],
        },
        "expected_artifacts": [
            "faithful complete judgeable standalone generated code repository",
            "README with setup and reproduction commands",
            "requirements or environment specification",
            "scripts/modules implementing obligations from the paper and addenda",
        ],
        "forbidden_shortcuts": [
            "Do not copy or depend on blacklisted official repositories.",
            "Do not fabricate final metric values or pretend experiments were run.",
            "Do not introduce requirements that are only present in evaluator-only metadata.",
        ],
        "allowed_approximations": [
            "Dataset acquisition should be implemented as scripts/configuration in the generated repository; this pipeline must not materialize datasets.",
            (
                "Generation may skip executing expensive training/evaluation only when it exposes runnable "
                "full-mode commands, configs, lazy loaders, and code paths that a researcher can execute later."
            ),
            (
                "Bounded smoke fixtures are allowed for validation when they call the same environment, model, "
                "training, refinement, metric, and artifact-writing interfaces used by full runs."
            ),
        ],
    }
    blacklist_text = ", ".join(loaded["blacklist"][:8]) if loaded["blacklist"] else "none"
    target = (
        f"PaperBench code reproduction for the paper: {title}.\n"
        "Input is the paper, not a proposal. Generate a faithful, complete, judgeable standalone code "
        "repository that implements the methods, data processing, evaluation interfaces, baselines, metrics, "
        "configs, scripts, and artifacts required by paper.md, and addendum.md when present. This pipeline "
        "generates code without running expensive experiments: it must implement training and evaluation "
        "code paths while skipping expensive execution during generation. "
        "Do not use blacklisted repositories. "
        f"Blacklisted repositories: {blacklist_text}."
    )
    return PaperBenchReproInput(
        target=target,
        paper_path=str(loaded["paper_path"]),
        chunk_max_chars=chunk_max_chars,
        language=language,
        max_iterations=max_iterations,
        stage_review_repair_budget=3,
        experiment_design=experiment_design,
        idea_references=idea_references,
        idea_reference_summaries=github_summaries,
    )
