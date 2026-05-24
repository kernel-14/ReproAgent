"""Resource normalization and preparation helpers for reproagent."""

from __future__ import annotations

import logging
import re
import shutil
import ssl
import urllib.request
from pathlib import Path
from typing import Any

from reproagent.pipeline.schemas import PaperBenchReproState
from reproagent.pipeline.utils.dataset_download_tool import download_datasets
from reproagent.pipeline.utils.ref_repo_clone import clone_reference_repository
from reproagent.pipeline.utils.ref_repo_search_tool import search_reference_repository

logger = logging.getLogger(__name__)

_GITHUB_REPO_URL_RE = re.compile(
    r"https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?",
    re.IGNORECASE,
)
_EXTERNAL_SERVICE_MARKERS = {
    "service",
    "external_service",
    "google scholar",
    "semantic scholar",
    "openai",
    "chatgpt",
    "web search",
}

def _get_dataset_preparation(state: PaperBenchReproState) -> dict[str, Any]:
    """Return prepared dataset metadata from temp_data."""
    payload = state.temp_data.get("dataset_preparation", {})
    return payload if isinstance(payload, dict) else {}


def _get_resource_manifest(state: PaperBenchReproState) -> dict[str, Any]:
    """Return the unified prepare-stage resource manifest from temp_data."""
    payload = state.temp_data.get("resource_manifest", {})
    return payload if isinstance(payload, dict) else {}

def _normalize_dataset_requests(experiment_design: dict[str, Any]) -> list[dict[str, str]]:
    """Normalize runtime data requests into a stable deduplicated list.

    Sources:
    1) experiment_design.datasets (download by name or use explicit path)
    """
    if not isinstance(experiment_design, dict):
        return []

    raw_dataset_items = experiment_design.get("datasets", [])
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()

    def _append_request(item: Any, *, source: str, require_path: bool = False) -> None:
        if isinstance(item, str):
            name = item.strip()
            description = ""
            role = ""
            path = ""
        elif isinstance(item, dict):
            name = str(item.get("name", "")).strip()
            description = str(item.get("description", "")).strip()
            role = str(item.get("role", "")).strip()
            path = str(
                item.get("path", "")
                or item.get("dataset_path", "")
                or item.get("local_path", "")
                or item.get("benchmark_path", "")
            ).strip()
        else:
            return

        if require_path and not path:
            return
        if not name:
            name = path

        if not name:
            return
        key = (path or name).lower()
        if key in seen:
            return
        seen.add(key)
        normalized.append({
            "name": name,
            "description": description,
            "role": role,
            "path": path,
            "source": source,
        })

    for item in raw_dataset_items if isinstance(raw_dataset_items, list) else []:
        _append_request(item, source="paper2exp_dataset", require_path=False)

    return normalized


def _safe_resource_name(value: str, fallback: str = "resource") -> str:
    """Return a filesystem-safe resource directory name."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()).strip("._")
    return safe or fallback


def _path_has_contents(path: Path) -> bool:
    """Detect whether a local resource already has usable contents."""
    if not path.exists():
        return False
    if path.is_file():
        return path.stat().st_size > 0
    try:
        return any(path.iterdir())
    except OSError:
        return False


def _resource_progress(path: Path) -> dict[str, Any]:
    """Return a lightweight download-progress/reuse status for a target path."""
    if _path_has_contents(path):
        return {"status": "ready", "path": str(path.resolve()), "reason": "target already has contents"}
    partial_markers = sorted(path.parent.glob(f"{path.name}*.partial")) if path.parent.exists() else []
    if partial_markers:
        return {
            "status": "partial",
            "path": str(path),
            "reason": "partial download marker exists",
            "markers": [str(item) for item in partial_markers],
        }
    return {"status": "missing", "path": str(path), "reason": ""}


def _item_text_values(value: Any) -> list[str]:
    """Flatten string-like fields from nested upstream payloads."""
    values: list[str] = []
    if isinstance(value, str):
        if value.strip():
            values.append(value.strip())
    elif isinstance(value, dict):
        for nested in value.values():
            values.extend(_item_text_values(nested))
    elif isinstance(value, list):
        for nested in value:
            values.extend(_item_text_values(nested))
    return values


def _first_existing_field(item: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = str(item.get(key, "") or "").strip()
        if value:
            return value
    return ""


def _first_url(item: dict[str, Any]) -> str:
    explicit = _first_existing_field(
        item,
        (
            "url",
            "download_url",
            "source_url",
            "data_url",
            "benchmark_url",
            "homepage",
        ),
    )
    if explicit.startswith(("http://", "https://")):
        return explicit
    for value in _item_text_values(item):
        if value.startswith(("http://", "https://")) and "github.com" not in value.lower():
            return value
    return ""


def _first_github_url(item: Any) -> str:
    for value in _item_text_values(item):
        match = _GITHUB_REPO_URL_RE.search(value)
        if match:
            return match.group(0).removesuffix(".git")
    return ""


def _normalize_named_items(raw_items: Any, *, source: str) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if not isinstance(raw_items, list):
        return normalized
    seen: set[str] = set()
    for index, item in enumerate(raw_items):
        if isinstance(item, str):
            payload: dict[str, Any] = {"name": item.strip()}
        elif isinstance(item, dict):
            payload = dict(item)
        else:
            continue
        name = str(payload.get("name", "") or payload.get("title", "") or payload.get("id", "")).strip()
        if not name:
            name = f"{source}_{index + 1}"
        payload["name"] = name
        payload["source"] = source
        key = (str(payload.get("local_path", "") or payload.get("path", "") or name)).lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(payload)
    return normalized


def _copy_or_symlink_resource(source_path: Path, target_path: Path) -> dict[str, Any]:
    """Materialize a local resource under the prepare output tree."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if _path_has_contents(target_path):
        return {"success": True, "method": "ReusedLocalMaterialization", "path": str(target_path.resolve())}
    try:
        if target_path.exists() or target_path.is_symlink():
            if target_path.is_dir() and not target_path.is_symlink():
                shutil.rmtree(target_path)
            else:
                target_path.unlink()
        target_path.symlink_to(source_path.resolve(), target_is_directory=source_path.is_dir())
        return {"success": True, "method": "LocalSymlink", "path": str(target_path.resolve())}
    except OSError:
        try:
            if source_path.is_dir():
                shutil.copytree(source_path, target_path, dirs_exist_ok=True)
            else:
                target_path.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, target_path / source_path.name)
            return {"success": True, "method": "LocalCopy", "path": str(target_path.resolve())}
        except Exception as exc:
            return {"success": False, "method": "LocalCopy", "path": "", "reason": str(exc)}


def _download_url_to_dir(url: str, target_path: Path) -> dict[str, Any]:
    """Download one URL into a resource directory, reusing completed targets."""
    progress = _resource_progress(target_path)
    if progress["status"] == "ready":
        return {"success": True, "method": "ReusedDownload", "path": progress["path"]}
    target_path.mkdir(parents=True, exist_ok=True)
    filename = Path(url.split("?", 1)[0]).name or "downloaded_resource"
    destination = target_path / filename
    partial = target_path / f"{filename}.partial"
    try:
        logger.info("Downloading resource URL %s to %s", url, destination)
        with urllib.request.urlopen(url, timeout=60, context=_urlopen_ssl_context()) as response, partial.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        partial.replace(destination)
        return {"success": True, "method": "URLDownload", "path": str(target_path.resolve())}
    except Exception as exc:
        return {"success": False, "method": "URLDownload", "path": str(target_path), "reason": str(exc)}


def _urlopen_ssl_context() -> ssl.SSLContext | None:
    """Use certifi's CA bundle when the active Python env does not configure one."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return None


def prepare_benchmarks(
    experiment_design: dict[str, Any],
    benchmark_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Prepare benchmark resources into an isolated benchmark directory."""
    benchmark_root.mkdir(parents=True, exist_ok=True)
    requests = _normalize_named_items(
        experiment_design.get("benchmarks", []) if isinstance(experiment_design, dict) else [],
        source="paper2exp_benchmark",
    )
    prepared: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []

    for item in requests:
        name = str(item.get("name", "")).strip()
        target = benchmark_root / _safe_resource_name(name, "benchmark")
        local_path = _first_existing_field(item, ("path", "benchmark_path", "local_path"))
        url = _first_url(item)
        result: dict[str, Any]
        original_path = ""
        if local_path:
            candidate = Path(local_path).expanduser()
            original_path = str(candidate)
            if candidate.exists() and _path_has_contents(candidate):
                result = _copy_or_symlink_resource(candidate, target)
            else:
                result = {"success": False, "method": "LocalPath", "path": "", "reason": f"Local path not found or empty: {candidate}"}
        elif url:
            result = _download_url_to_dir(url, target)
        else:
            progress = _resource_progress(target)
            if progress["status"] == "ready":
                result = {"success": True, "method": "ReusedDownload", "path": progress["path"]}
            else:
                download_result = (download_datasets([name], str(benchmark_root)) or [{}])[0]
                result = {
                    "success": bool(download_result.get("success", False)),
                    "method": str(download_result.get("method", "")),
                    "path": str(download_result.get("path", "")),
                    "reason": str(download_result.get("reason", download_result.get("error", ""))),
                }

        payload = {
            "name": name,
            "source": "paper2exp_benchmark",
            "status": "local_ready" if result.get("success") and local_path else "downloaded" if result.get("success") else "unresolved",
            "local_path": str(result.get("path", "")),
            "original_path": original_path,
            "download_method": str(result.get("method", "")),
            "error_message": str(result.get("reason", "")),
        }
        if result.get("success"):
            prepared.append(payload)
        else:
            unresolved.append(payload)

    return {
        "download_root": str(benchmark_root.resolve()),
        "requested_benchmarks": requests,
        "prepared_benchmarks": prepared,
        "unresolved_benchmarks": unresolved,
    }, prepared + unresolved


def _is_external_service_baseline(item: dict[str, Any]) -> bool:
    subtype = str(item.get("type", "") or item.get("subtype", "")).strip().lower()
    if subtype == "external_service":
        return True
    combined = " ".join(_item_text_values(item)).lower()
    return any(re.search(rf"\b{re.escape(marker)}\b", combined) for marker in _EXTERNAL_SERVICE_MARKERS)


def _paper_search_payloads(source_papers: Any) -> list[dict[str, str]]:
    payloads: list[dict[str, str]] = []
    if not isinstance(source_papers, list):
        return payloads
    for item in source_papers:
        if not isinstance(item, dict):
            continue
        paper_path = str(item.get("paper_path", "") or item.get("path", "") or "").strip()
        if not paper_path:
            continue
        payloads.append(
            {
                "ref_id": str(item.get("ref_id", "") or item.get("paper_id", "") or item.get("id", "") or "").strip(),
                "paper_path": paper_path,
                "title": str(item.get("title", "") or "").strip(),
                "paper_url": str(item.get("paper_url", "") or item.get("url", "") or "").strip(),
            }
        )
    return payloads


def prepare_baselines(
    experiment_design: dict[str, Any],
    baseline_root: Path,
    *,
    github_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Prepare baseline code resources under baselines/ without mixing them with refs."""
    baseline_root.mkdir(parents=True, exist_ok=True)
    requests = _normalize_named_items(
        experiment_design.get("baselines", []) if isinstance(experiment_design, dict) else [],
        source="paper2exp_baseline",
    )
    prepared: list[dict[str, Any]] = []

    for item in requests:
        name = str(item.get("name", "")).strip()
        source_papers = list(item.get("source_papers", []) or []) if isinstance(item.get("source_papers", []), list) else []
        if _is_external_service_baseline(item):
            prepared.append(
                {
                    "name": name,
                    "type": "external_service",
                    "status": "external_service",
                    "source_papers": source_papers,
                    "local_path": "",
                    "repository_url": "",
                }
            )
            continue

        repository_url = (
            _first_github_url(item)
            or _first_existing_field(item, ("repository_url", "repo_url", "github_url", "code_url"))
        )
        resolve_reason = ""
        if not repository_url:
            for paper in _paper_search_payloads(source_papers):
                try:
                    resolve_result = search_reference_repository(
                        ref_id=paper["ref_id"],
                        paper_path=paper["paper_path"],
                        paper_title=paper["title"],
                        paper_url=paper["paper_url"],
                        github_config=github_config,
                    )
                except Exception as exc:
                    resolve_reason = str(exc)
                    continue
                repository_url = str(resolve_result.get("repository_url", "") or "").strip()
                resolve_reason = str(resolve_result.get("reason", "") or "").strip()
                if repository_url and str(resolve_result.get("status", "") or "") == "found":
                    break

        if repository_url:
            clone_result = clone_reference_repository(
                repository_url,
                baseline_root,
                github_config=github_config,
            )
            success = str(clone_result.get("status", "")) in {"cloned", "reused"}
            prepared.append(
                {
                    "name": name,
                    "type": "external_repo",
                    "status": str(clone_result.get("status", "")) if success else "clone_failed",
                    "local_path": str(clone_result.get("local_repo_path", "")),
                    "repository_url": str(clone_result.get("repository_url", repository_url)),
                    "source_papers": source_papers,
                    "error_message": str(clone_result.get("reason", "")),
                }
            )
            continue

        prepared.append(
            {
                "name": name,
                "type": "implement_from_scratch",
                "status": "deferred_to_milestone",
                "local_path": "",
                "repository_url": "",
                "source_papers": source_papers,
                "error_message": resolve_reason,
            }
        )

    return {
        "clone_root": str(baseline_root.resolve()),
        "requested_baselines": requests,
        "prepared_baselines": prepared,
    }


def build_resource_manifest(
    *,
    dataset_preparation: dict[str, Any],
    benchmark_preparation: dict[str, Any],
    baseline_preparation: dict[str, Any],
    reference_repo_preparation: dict[str, Any],
    reference_repo_surveys: list[Any] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Build the prepare-stage resource_manifest.json payload."""
    survey_by_ref_id = {
        str(item.get("ref_id", "") if isinstance(item, dict) else getattr(item, "ref_id", "")).strip(): (
            item if isinstance(item, dict) else item.model_dump(mode="json")
        )
        for item in list(reference_repo_surveys or [])
    }
    datasets: list[dict[str, Any]] = []
    for item in list(dataset_preparation.get("downloaded_datasets", []) or []):
        method = str(item.get("download_method", "") or "")
        datasets.append(
            {
                "name": item.get("name", ""),
                "source": item.get("source", "paper2exp_dataset"),
                "status": "local_ready" if method.startswith("Local") or method.startswith("Reused") else "downloaded",
                "local_path": item.get("local_path", ""),
                "download_method": method,
            }
        )
    for item in list(dataset_preparation.get("failed_datasets", []) or []):
        datasets.append(
            {
                "name": item.get("name", ""),
                "source": item.get("source", "paper2exp_dataset"),
                "status": "unresolved",
                "local_path": item.get("local_path", ""),
                "download_method": item.get("download_method", ""),
                "error_message": item.get("error_message", ""),
            }
        )

    ref_repos: list[dict[str, Any]] = []
    for item in list(reference_repo_preparation.get("prepared_repositories", []) or []):
        ref_id = str(item.get("ref_id", "") or "").strip()
        ref_repos.append(
            {
                "ref_id": ref_id,
                "title": item.get("title", ""),
                "repository_url": item.get("repository_url", ""),
                "status": item.get("status", ""),
                "local_path": item.get("local_repo_path", ""),
                "survey": survey_by_ref_id.get(ref_id, {}),
            }
        )
    for item in list(reference_repo_preparation.get("failed_repositories", []) or []):
        ref_repos.append(
            {
                "ref_id": item.get("ref_id", ""),
                "title": item.get("title", ""),
                "repository_url": item.get("repository_url", ""),
                "status": item.get("status", "resolve_failed"),
                "local_path": item.get("local_repo_path", ""),
                "error_message": item.get("error_message", ""),
            }
        )

    return {
        "schema_version": "2.0",
        "prepare_status": "completed",
        "datasets": datasets,
        "benchmarks": list(benchmark_preparation.get("prepared_benchmarks", []) or [])
        + list(benchmark_preparation.get("unresolved_benchmarks", []) or []),
        "baselines": list(baseline_preparation.get("prepared_baselines", []) or []),
        "ref_repos": ref_repos,
        "warnings": list(warnings or []),
    }

def _build_dataset_preparation_payload(
    dataset_requests: list[dict[str, str]],
    download_root: Path,
    download_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a stable dataset-preparation artifact payload."""
    result_by_name = {
        str(item.get("name", "")).lower(): item
        for item in download_results
        if isinstance(item, dict) and item.get("name")
    }
    downloaded_datasets: list[dict[str, Any]] = []
    failed_datasets: list[dict[str, Any]] = []

    for request in dataset_requests:
        result = result_by_name.get(request["name"].lower(), {})
        payload = {
            "name": request["name"],
            "description": request["description"],
            "role": request["role"],
            "source": request["source"],
            "local_path": str(result.get("path", "")),
            "download_method": str(result.get("method", "")),
            "success": bool(result.get("success", False)),
            "error_message": str(result.get("reason", result.get("error", ""))),
        }
        if payload["success"]:
            downloaded_datasets.append(payload)
        else:
            failed_datasets.append(payload)

    return {
        "download_root": str(download_root.resolve()),
        "requested_datasets": dataset_requests,
        "downloaded_datasets": downloaded_datasets,
        "failed_datasets": failed_datasets,
    }

def _update_input_dataset_status(
    experiment_design: dict[str, Any],
    download_results: list[dict[str, Any]] | None = None,
) -> None:
    """Write stable dataset download status back into experiment_design.datasets."""
    if not isinstance(experiment_design, dict):
        return

    raw_items = experiment_design.get("datasets", [])
    if not isinstance(raw_items, list):
        return

    result_by_name = {
        str(item.get("name", "")).lower(): item
        for item in (download_results or [])
        if isinstance(item, dict) and item.get("name")
    }

    updated_items: list[Any] = []
    for item in raw_items:
        if isinstance(item, str):
            payload: dict[str, Any] = {
                "name": item,
                "description": "",
                "role": "",
            }
        elif isinstance(item, dict):
            payload = dict(item)
        else:
            updated_items.append(item)
            continue

        payload.setdefault("status", "pending")
        payload.setdefault("local_path", "unknow")

        dataset_name = str(payload.get("name", "")).strip()
        if dataset_name:
            result = result_by_name.get(dataset_name.lower())
            if result:
                if result.get("success", False):
                    payload["status"] = "downloaded"
                    payload["local_path"] = str(result.get("path", "unknow") or "unknow")
                else:
                    payload["status"] = "failed"
                    payload["local_path"] = "unknow"

        updated_items.append(payload)

    experiment_design["datasets"] = updated_items


def _update_input_benchmark_status(
    experiment_design: dict[str, Any],
    download_results: list[dict[str, Any]] | None = None,
) -> None:
    """Write stable benchmark local/download status back into experiment_design.benchmarks."""
    if not isinstance(experiment_design, dict):
        return

    raw_items = experiment_design.get("benchmarks", [])
    if not isinstance(raw_items, list):
        return

    result_by_name = {
        str(item.get("name", "")).lower(): item
        for item in (download_results or [])
        if isinstance(item, dict) and item.get("name")
    }

    updated_items: list[Any] = []
    for item in raw_items:
        if isinstance(item, str):
            payload: dict[str, Any] = {
                "name": item,
                "description": "",
                "role": "",
            }
        elif isinstance(item, dict):
            payload = dict(item)
        else:
            updated_items.append(item)
            continue

        payload.setdefault("status", "pending")
        payload.setdefault("local_path", "unknow")

        benchmark_name = str(payload.get("name", "")).strip()
        if benchmark_name:
            result = result_by_name.get(benchmark_name.lower())
            if result:
                if result.get("success", False):
                    payload["status"] = "ready"
                    payload["local_path"] = str(result.get("path", "unknow") or "unknow")
                else:
                    payload["status"] = "failed"
                    payload["local_path"] = "unknow"

        updated_items.append(payload)

    experiment_design["benchmarks"] = updated_items
