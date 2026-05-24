"""Reference repository search utility for reproagent."""

from __future__ import annotations

import base64
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

GITHUB_REPO_RE = re.compile(r"https?://github\.com/([^/\s]+)/([^/\s#?]+)", flags=re.I)

_HTTP_HEADERS = {
    "Accept": "application/vnd.github+json",
    "User-Agent": "exp-gen-ref-repo-search/1.0",
}

_PLACEHOLDER_RE = re.compile(
    r"(code\s+coming\s+soon|to\s+be\s+released|placeholder|release\s+soon|will\s+release)",
    re.IGNORECASE,
)
_OFFICIAL_CONTEXT_RE = re.compile(
    r"(source\s+code|official\s+code|official\s+implementation|our\s+code|code\s+can\s+be\s+found)",
    re.IGNORECASE,
)
_BASELINE_CONTEXT_RE = re.compile(
    r"(baseline|regarding|comparison\s+method|implementation of baseline methods)",
    re.IGNORECASE,
)
_ARXIV_URL_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([^/?#\s]+)", re.IGNORECASE)
_ARXIV_ID_RE = re.compile(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b", re.IGNORECASE)
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
_KEYWORD_RE = re.compile(r"\b(code|implementation|github|official code)\b", re.IGNORECASE)
_GITHUB_URL_RE = re.compile(r"https?://github\.com/[^\s)\]>]+", re.IGNORECASE)

_MEANINGFUL_FILE_NAMES = {
    "requirements.txt",
    "environment.yml",
    "environment.yaml",
    "pyproject.toml",
    "setup.py",
    "makefile",
    "dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
}
_MEANINGFUL_SUFFIXES = {
    ".py",
    ".ipynb",
    ".sh",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".cfg",
    ".ini",
    ".cu",
    ".cpp",
    ".cc",
    ".c",
    ".h",
    ".hpp",
    ".jl",
    ".r",
    ".m",
}
_MEANINGFUL_DIR_NAMES = {
    "src",
    "scripts",
    "script",
    "code",
    "configs",
    "config",
    "notebooks",
    "notebook",
    "experiments",
    "exp",
    "models",
    "model",
}
_IGNORED_DIR_NAMES = {
    ".github",
    "docs",
    "doc",
    "assets",
    "images",
    "figures",
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


def search_reference_repository(
    *,
    ref_id: str,
    paper_path: str,
    paper_title: str = "",
    paper_url: str = "",
    max_results: int = 5,
    github_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Find a trusted GitHub repository for one reference paper."""
    markdown = ""
    if str(paper_path or "").strip():
        path = Path(paper_path)
        if path.exists():
            markdown = _normalize_paper_markdown(path.read_text(encoding="utf-8"))
    paper_signals = extract_paper_repository_signals(
        markdown=markdown,
        paper_title=paper_title,
        paper_url=paper_url,
    )

    candidate_failures: list[str] = []
    title_aliases = _derive_title_aliases(paper_signals["title"])

    for candidate in paper_signals["github_candidates"]:
        evaluation = validate_official_candidate(candidate["url"], github_config=github_config)
        if evaluation["valid"]:
            return {
                "ref_id": ref_id,
                "paper_path": paper_path,
                "status": "found",
                "repository_type": "official",
                "repository_url": evaluation["repository_url"],
                "stars": evaluation["stars"],
                "matched_signals": [
                    "paper contains github url",
                    *(["github url appears near code keywords"] if candidate["context"] else []),
                    *evaluation["matched_signals"],
                ],
                "reason": "",
                "title": paper_signals["title"],
                "arxiv_id": paper_signals["arxiv_id"],
                "doi": paper_signals["doi"],
                "github_urls": [item["url"] for item in paper_signals["github_candidates"]],
                "keyword_contexts": paper_signals["keyword_contexts"],
            }
        candidate_failures.append(f"{candidate['url']}: {evaluation['reason']}")

    if paper_signals["title"]:
        reproduction = _search_reproduction_candidate(
            title=paper_signals["title"],
            title_aliases=title_aliases,
            arxiv_id=paper_signals["arxiv_id"],
            doi=paper_signals["doi"],
            max_results=max_results,
            github_config=github_config,
        )
        if reproduction is not None:
            return {
                "ref_id": ref_id,
                "paper_path": paper_path,
                "status": "found",
                "repository_type": "reproduction",
                "repository_url": reproduction["repository_url"],
                "stars": reproduction["stars"],
                "matched_signals": reproduction["matched_signals"],
                "reason": "",
                "title": paper_signals["title"],
                "arxiv_id": paper_signals["arxiv_id"],
                "doi": paper_signals["doi"],
                "github_urls": [item["url"] for item in paper_signals["github_candidates"]],
                "keyword_contexts": paper_signals["keyword_contexts"],
            }

    failure_reason = "; ".join(candidate_failures) if candidate_failures else "No trusted repository found"
    return {
        "ref_id": ref_id,
        "paper_path": paper_path,
        "status": "not_found",
        "repository_type": "not_found",
        "repository_url": "",
        "stars": 0,
        "matched_signals": [],
        "reason": failure_reason,
        "title": paper_signals["title"],
        "arxiv_id": paper_signals["arxiv_id"],
        "doi": paper_signals["doi"],
        "github_urls": [item["url"] for item in paper_signals["github_candidates"]],
        "keyword_contexts": paper_signals["keyword_contexts"],
    }


def extract_paper_repository_signals(
    *,
    markdown: str,
    paper_title: str = "",
    paper_url: str = "",
) -> dict[str, Any]:
    """Extract title, ids, GitHub URLs and nearby keyword contexts from paper markdown."""
    normalized_markdown = _normalize_paper_markdown(markdown)
    title = _extract_title(markdown, paper_title)
    arxiv_id = _extract_arxiv_id(normalized_markdown, paper_url)
    doi = _extract_doi(normalized_markdown, paper_url)
    github_candidates = _extract_github_candidates(normalized_markdown, title=title)
    keyword_contexts = _extract_keyword_contexts(normalized_markdown)
    return {
        "title": title,
        "arxiv_id": arxiv_id,
        "doi": doi,
        "github_candidates": github_candidates,
        "keyword_contexts": keyword_contexts,
    }


def validate_official_candidate(
    repository_url: str,
    *,
    github_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run lightweight exclusion checks for a paper-linked GitHub repo."""
    normalized = _normalize_github_repo_url(repository_url)
    if normalized is None:
        return {
            "valid": False,
            "repository_url": "",
            "stars": 0,
            "matched_signals": [],
            "reason": "URL is not a standard GitHub repository homepage",
        }

    snapshot = _fetch_repo_snapshot(normalized, github_config=github_config)
    if snapshot is None:
        return {
            "valid": False,
            "repository_url": normalized,
            "stars": 0,
            "matched_signals": [],
            "reason": "Repository does not exist or is not accessible",
        }

    readme_text = snapshot.get("readme_text", "")
    if _PLACEHOLDER_RE.search(readme_text):
        return {
            "valid": False,
            "repository_url": normalized,
            "stars": snapshot.get("stars", 0),
            "matched_signals": [],
            "reason": "Repository appears to be a placeholder",
        }

    if not _has_meaningful_project_content(snapshot.get("root_entries", [])):
        return {
            "valid": False,
            "repository_url": normalized,
            "stars": snapshot.get("stars", 0),
            "matched_signals": [],
            "reason": "Repository appears empty or lacks source/config/script files",
        }

    return {
        "valid": True,
        "repository_url": normalized,
        "stars": snapshot.get("stars", 0),
        "matched_signals": [
            "repository is accessible",
            "repository has non-placeholder contents",
        ],
        "reason": "",
    }


def _search_reproduction_candidate(
    *,
    title: str,
    title_aliases: list[str],
    arxiv_id: str,
    doi: str,
    max_results: int,
    github_config: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    candidates = search_github_repositories(
        title=title,
        title_aliases=title_aliases,
        arxiv_id=arxiv_id,
        max_results=max_results,
        github_config=github_config,
    )

    best: dict[str, Any] | None = None
    for candidate in candidates:
        evaluation = _evaluate_reproduction_candidate(
            repository_url=candidate["repository_url"],
            paper_title=title,
            arxiv_id=arxiv_id,
            doi=doi,
            github_config=github_config,
        )
        if evaluation is None:
            continue
        evaluation["matched_signals"] = [*candidate["matched_signals"], *evaluation["matched_signals"]]
        evaluation["score"] += candidate["score"]
        if best is None or evaluation["score"] > best["score"]:
            best = evaluation
    return best


def search_github_repositories(
    *,
    title: str,
    title_aliases: list[str] | None = None,
    arxiv_id: str = "",
    max_results: int = 5,
    github_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Search GitHub repositories directly for paper-aligned candidates."""
    title_aliases = title_aliases or _derive_title_aliases(title)
    if _requires_strong_identifier_for_search(title) and not arxiv_id:
        return []
    queries = _build_github_repo_queries(
        title=title,
        title_aliases=title_aliases,
        arxiv_id=arxiv_id,
    )
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    for query in queries:
        for item in _github_repository_search(
            query=query,
            per_page=max(10, max_results * 4),
            github_config=github_config,
        ):
            repository_url = _canonicalize_github_repo_url(item.get("html_url", ""))
            if not repository_url or repository_url in seen:
                continue
            score = _score_github_search_hit(item=item, title=title, title_aliases=title_aliases or [])
            arxiv_match = bool(arxiv_id and arxiv_id.lower() in _normalize_text(item.get("description", "")))
            if score <= 0 and not arxiv_match:
                continue
            matched_signals = ["github repository search hit"]
            if arxiv_match:
                matched_signals.append("search description mentions arxiv id")
            candidates.append(
                {
                    "repository_url": repository_url,
                    "stars": int(item.get("stargazers_count", 0) or 0),
                    "description": item.get("description", "") or "",
                    "matched_signals": matched_signals,
                    "score": score,
                }
            )
            seen.add(repository_url)

    candidates.sort(key=lambda item: (-item["score"], -item["stars"], item["repository_url"]))
    return candidates[:max_results]


def _evaluate_reproduction_candidate(
    *,
    repository_url: str,
    paper_title: str,
    arxiv_id: str,
    doi: str,
    github_config: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    normalized = _canonicalize_github_repo_url(repository_url)
    if normalized is None:
        return None

    snapshot = _fetch_repo_snapshot(normalized, github_config=github_config)
    if snapshot is None:
        return None
    if snapshot.get("stars", 0) <= 50:
        return None
    if _PLACEHOLDER_RE.search(snapshot.get("readme_text", "")):
        return None
    if not _has_meaningful_project_content(snapshot.get("root_entries", [])):
        return None

    combined_text = " ".join(
        [
            snapshot.get("description", ""),
            snapshot.get("readme_text", ""),
        ]
    )
    matched_signals = ["github search candidate", "stars > 50"]

    title_match = _title_matches(paper_title, combined_text)
    if title_match:
        matched_signals.append("readme or description mentions paper title")
    if arxiv_id and arxiv_id.lower() in combined_text.lower():
        matched_signals.append("readme mentions arxiv id")
    if doi and doi.lower() in combined_text.lower():
        matched_signals.append("readme mentions doi")

    if len(matched_signals) <= 2:
        return None

    return {
        "repository_url": normalized,
        "stars": snapshot.get("stars", 0),
        "matched_signals": matched_signals,
        "score": snapshot.get("stars", 0) + 100 * (len(matched_signals) - 2),
    }


def _extract_title(markdown: str, provided_title: str) -> str:
    if provided_title.strip():
        return provided_title.strip()

    latex_title_match = re.search(r"\\title\s*\{\s*(.*?)\s*\}", markdown, re.DOTALL)
    if latex_title_match:
        latex_title = " ".join(latex_title_match.group(1).split())
        if latex_title:
            return latex_title

    lines = [line.strip() for line in markdown.splitlines() if line.strip()]
    for index, line in enumerate(lines[:10]):
        if line == r"\title{" and index + 1 < len(lines):
            next_line = lines[index + 1].strip().rstrip("}")
            if next_line:
                return next_line
    for line in lines[:10]:
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return lines[0] if lines else ""


def _extract_arxiv_id(markdown: str, paper_url: str) -> str:
    for source in (paper_url, markdown):
        if not source:
            continue
        url_match = _ARXIV_URL_RE.search(source)
        if url_match:
            arxiv_id = url_match.group(1)
            return arxiv_id[:-4] if arxiv_id.endswith(".pdf") else arxiv_id
        id_match = _ARXIV_ID_RE.search(source)
        if id_match:
            return id_match.group(1)
    return ""


def _extract_doi(markdown: str, paper_url: str) -> str:
    for source in (paper_url, markdown):
        if not source:
            continue
        match = _DOI_RE.search(source.upper())
        if match:
            return match.group(0)
    return ""


def _extract_github_candidates(markdown: str, *, title: str = "") -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    lines = markdown.splitlines()
    title_aliases = _derive_title_aliases(title)
    for index, line in enumerate(lines):
        line_urls = _GITHUB_URL_RE.findall(line)
        for raw_url in line_urls:
            normalized = _canonicalize_github_repo_url(raw_url)
            if normalized is None or normalized in seen:
                continue
            start = max(0, index - 1)
            end = min(len(lines), index + 2)
            context = "\n".join(part.strip() for part in lines[start:end] if part.strip())
            has_keyword_context = bool(_KEYWORD_RE.search(context))
            score = _score_paper_link_candidate(context=context, title_aliases=title_aliases)
            candidates.append(
                {
                    "url": normalized,
                    "context": context if has_keyword_context else "",
                    "score": score,
                }
            )
            seen.add(normalized)

    candidates.sort(key=lambda item: (-item["score"], 0 if item["context"] else 1, item["url"]))
    return candidates


def _extract_keyword_contexts(markdown: str) -> list[str]:
    contexts: list[str] = []
    lines = markdown.splitlines()
    for index, line in enumerate(lines):
        if not _KEYWORD_RE.search(line):
            continue
        start = max(0, index - 1)
        end = min(len(lines), index + 2)
        context = "\n".join(part.strip() for part in lines[start:end] if part.strip())
        if context not in contexts:
            contexts.append(context)
    return contexts[:10]


def _normalize_github_repo_url(url: str) -> str | None:
    parsed = urlparse(url.strip())
    if parsed.netloc.lower() != "github.com":
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) != 2:
        return None
    owner, repo = parts
    repo = repo.removesuffix(".git")
    if not owner or not repo:
        return None
    return f"https://github.com/{owner}/{repo}"


def _canonicalize_github_repo_url(url: str) -> str | None:
    match = GITHUB_REPO_RE.search(url.strip())
    if not match:
        return None
    owner = match.group(1).strip()
    repo = re.sub(r"\.git$", "", match.group(2).strip().rstrip(").,]"))
    if not owner or not repo:
        return None
    return f"https://github.com/{owner}/{repo}"


def _fetch_repo_snapshot(
    repository_url: str,
    *,
    github_config: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    owner, repo = _split_repo_url(repository_url)
    try:
        meta_response = requests.get(
            f"{_github_api_base_url(github_config)}/repos/{owner}/{repo}",
            headers=_github_headers(github_config),
            timeout=20,
        )
    except requests.RequestException:
        return None
    if meta_response.status_code != 200:
        return None
    meta = meta_response.json()

    try:
        contents_response = requests.get(
            f"{_github_api_base_url(github_config)}/repos/{owner}/{repo}/contents",
            headers=_github_headers(github_config),
            timeout=20,
        )
    except requests.RequestException:
        return None
    if contents_response.status_code != 200:
        return None
    root_entries = contents_response.json()

    readme_text = ""
    try:
        readme_response = requests.get(
            f"{_github_api_base_url(github_config)}/repos/{owner}/{repo}/readme",
            headers=_github_headers(github_config),
            timeout=20,
        )
    except requests.RequestException:
        readme_response = None
    if readme_response is not None and readme_response.status_code == 200:
        payload = readme_response.json()
        content = payload.get("content", "")
        if payload.get("encoding") == "base64" and content:
            try:
                readme_text = base64.b64decode(content).decode("utf-8", errors="ignore")
            except Exception:
                readme_text = ""

    return {
        "repository_url": repository_url,
        "stars": meta.get("stargazers_count", 0),
        "description": meta.get("description", "") or "",
        "default_branch": meta.get("default_branch", ""),
        "root_entries": root_entries if isinstance(root_entries, list) else [],
        "readme_text": readme_text,
    }


def _has_meaningful_project_content(root_entries: list[dict[str, Any]]) -> bool:
    for entry in root_entries:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", ""))
        entry_type = str(entry.get("type", ""))
        lower_name = name.lower()
        suffix = Path(lower_name).suffix
        if entry_type == "dir" and lower_name not in _IGNORED_DIR_NAMES:
            if lower_name in _MEANINGFUL_DIR_NAMES:
                return True
            return True
        if lower_name in _MEANINGFUL_FILE_NAMES or suffix in _MEANINGFUL_SUFFIXES:
            return True
    return False


def _title_matches(title: str, text: str) -> bool:
    normalized_title = _normalize_text(title)
    normalized_text = _normalize_text(text)
    if not normalized_title or not normalized_text:
        return False
    title_tokens = [token for token in normalized_title.split() if len(token) > 2]
    if not title_tokens:
        return False
    if len(title_tokens) == 1:
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(title_tokens[0])}(?![a-z0-9])", normalized_text))
    if normalized_title in normalized_text:
        return True

    if len(title_tokens) < 3:
        return False
    overlap = sum(1 for token in title_tokens if token in normalized_text)
    return overlap / len(title_tokens) >= 0.6


def _normalized_token_matches(token: str, normalized_text: str) -> bool:
    token = _normalize_text(token)
    if not token or not normalized_text:
        return False
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", normalized_text))


def _alias_matches(alias: str, text: str) -> bool:
    normalized_alias = _normalize_text(alias)
    normalized_text = _normalize_text(text)
    if not normalized_alias or not normalized_text:
        return False
    alias_tokens = [token for token in normalized_alias.split() if len(token) > 2]
    if not alias_tokens:
        return False
    if len(alias_tokens) == 1:
        return _normalized_token_matches(alias_tokens[0], normalized_text)
    if normalized_alias in normalized_text:
        return True
    if len(alias_tokens) < 3:
        return False
    overlap = sum(1 for token in alias_tokens if _normalized_token_matches(token, normalized_text))
    return overlap / len(alias_tokens) >= 0.6


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(value).lower())).strip()


def _requires_strong_identifier_for_search(title: str) -> bool:
    normalized_title = _normalize_text(title)
    tokens = normalized_title.split()
    if not tokens:
        return True
    if normalized_title in _AMBIGUOUS_REPO_SEARCH_TITLES:
        return True
    if len(tokens) == 1 and (len(tokens[0]) <= 5 or str(title).strip().isupper()):
        return True
    return False


def _split_repo_url(repository_url: str) -> tuple[str, str]:
    parsed = urlparse(repository_url)
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        raise ValueError(f"Invalid repository URL: {repository_url}")
    return parts[0], parts[1]


def _normalize_paper_markdown(markdown: str) -> str:
    normalized = markdown
    normalized = re.sub(r"https?\s*:\s*/\s*/\s*github\.com\s*/\s*", "https://github.com/", normalized, flags=re.IGNORECASE)
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
    normalized = normalized.replace("https:// github.com/", "https://github.com/")
    return normalized


def _derive_title_aliases(title: str) -> list[str]:
    aliases: list[str] = []
    cleaned = title.strip()
    if not cleaned:
        return aliases
    aliases.append(cleaned)
    prefix = cleaned.split(":", 1)[0].strip()
    if prefix and prefix not in aliases:
        aliases.append(prefix)
    acronym_tokens = re.findall(r"\b[A-Z][A-Z0-9]{1,}\b", cleaned)
    for token in acronym_tokens:
        if token not in aliases:
            aliases.append(token)
    condensed = re.sub(r"[^A-Za-z0-9]+", " ", cleaned).strip()
    if condensed and condensed not in aliases:
        aliases.append(condensed)
    return aliases


def _score_paper_link_candidate(*, context: str, title_aliases: list[str]) -> int:
    score = 0
    if _OFFICIAL_CONTEXT_RE.search(context):
        score += 10
    if _KEYWORD_RE.search(context):
        score += 2
    if _BASELINE_CONTEXT_RE.search(context):
        score -= 4
    for alias in title_aliases:
        if _alias_matches(alias, context):
            score += 8
            break
    return score


def _build_github_repo_queries(*, title: str, title_aliases: list[str], arxiv_id: str) -> list[str]:
    queries: list[str] = []
    title = title.strip()
    if title:
        queries.append(f'"{title}" archived:false fork:false')
        queries.append(f'"{title}" (code OR implementation OR official) archived:false fork:false')
    if arxiv_id:
        queries.append(f'"{arxiv_id}" archived:false fork:false')
    for alias in title_aliases:
        cleaned = alias.strip()
        if not cleaned or cleaned == title:
            continue
        queries.append(f'"{cleaned}" archived:false fork:false')
        if len(cleaned.split()) <= 6:
            queries.append(f'"{cleaned}" (rl OR reinforcement OR code OR implementation) archived:false fork:false')
    return queries


def _github_repository_search(
    *,
    query: str,
    per_page: int = 5,
    github_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    try:
        response = requests.get(
            f"{_github_api_base_url(github_config)}/search/repositories",
            headers=_github_headers(github_config),
            params={
                "q": query,
                "sort": "stars",
                "order": "desc",
                "per_page": max(1, min(per_page, 20)),
            },
            timeout=20,
        )
    except requests.RequestException:
        return []
    if response.status_code != 200:
        return []
    payload = response.json()
    items = payload.get("items", [])
    return [item for item in items if isinstance(item, dict)]


def _github_headers(github_config: dict[str, Any] | None = None) -> dict[str, str]:
    headers = dict(_HTTP_HEADERS)
    token = _github_api_token(github_config)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _github_api_token(github_config: dict[str, Any] | None = None) -> str:
    configured = str((github_config or {}).get("api_key", "")).strip()
    if configured:
        return configured
    return (
        os.getenv("PAPERBENCH_REPRO_GITHUB_TOKEN", "").strip()
        or os.getenv("EXP_GEN_GITHUB_TOKEN", "").strip()
        or os.getenv("GITHUB_TOKEN", "").strip()
    )


def _github_api_base_url(github_config: dict[str, Any] | None = None) -> str:
    configured = str((github_config or {}).get("api_base_url", "")).strip()
    if configured:
        return configured.rstrip("/")
    return (
        os.getenv("PAPERBENCH_REPRO_GITHUB_API_BASE_URL", "").strip()
        or os.getenv("EXP_GEN_GITHUB_API_BASE_URL", "https://api.github.com").strip()
        or "https://api.github.com"
    ).rstrip("/")


def _score_github_search_hit(*, item: dict[str, Any], title: str, title_aliases: list[str]) -> int:
    text = " ".join(
        [
            str(item.get("name", "") or ""),
            str(item.get("full_name", "") or ""),
            str(item.get("description", "") or ""),
        ]
    )
    score = 0
    if _title_matches(title, text):
        score += 150
    normalized_name = _normalize_text(str(item.get("name", "") or ""))
    for alias in title_aliases:
        if _alias_matches(alias, text):
            score += 80
            break
        normalized_alias = _normalize_text(alias)
        alias_tokens = normalized_alias.split()
        if alias_tokens and all(
            _normalized_token_matches(token, normalized_name)
            for token in alias_tokens[: min(3, len(alias_tokens))]
        ):
            score += 40
            break
    return score
