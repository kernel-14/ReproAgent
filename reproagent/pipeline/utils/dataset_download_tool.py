"""Module-local dataset download support for reproagent."""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class DatasetDownloader:
    """Dataset downloader with HuggingFace, ModelScope and Kaggle fallback."""

    def __init__(self, download_dir: str):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def download_multiple(self, dataset_names: list[str]) -> list[dict]:
        """Download multiple datasets."""
        results = []
        for name in dataset_names:
            results.append(self._download_single(name))
        return results

    def _download_single(self, name: str) -> dict:
        """Download one dataset with fallback across supported sources."""
        errors = []

        hf_result = self._try_huggingface(name)
        if hf_result["success"]:
            logger.info("[%s] Downloaded via HuggingFace to %s", name, hf_result["path"])
            return {"name": name, "success": True, "method": "HuggingFace", "path": hf_result["path"]}
        errors.append(f"HuggingFace: {hf_result['error']}")

        ms_result = self._try_modelscope(name)
        if ms_result["success"]:
            logger.info("[%s] Downloaded via ModelScope to %s", name, ms_result["path"])
            return {"name": name, "success": True, "method": "ModelScope", "path": ms_result["path"]}
        errors.append(f"ModelScope: {ms_result['error']}")

        kaggle_result = self._try_kaggle(name)
        if kaggle_result["success"]:
            logger.info("[%s] Downloaded via Kaggle to %s", name, kaggle_result["path"])
            return {"name": name, "success": True, "method": "Kaggle", "path": kaggle_result["path"]}
        errors.append(f"Kaggle: {kaggle_result['error']}")

        logger.error("[%s] All methods failed: %s", name, "; ".join(errors))
        return {"name": name, "success": False, "reason": "; ".join(errors)}

    def _try_huggingface(self, name: str) -> dict:
        """Download from HuggingFace."""
        try:
            if "HF_ENDPOINT" not in os.environ:
                os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            from datasets import load_dataset
            from huggingface_hub import HfApi

            api = HfApi()
            datasets = list(api.list_datasets(search=name, limit=5))
            if not datasets:
                return {"success": False, "error": "No datasets found"}

            best = next((ds for ds in datasets if ds.id.lower() == name.lower()), None) or max(
                datasets,
                key=lambda d: getattr(d, "downloads", 0),
            )
            dataset_id = best.id
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in dataset_id)
            save_dir = self.download_dir / safe_name
            save_dir.mkdir(parents=True, exist_ok=True)

            ds = load_dataset(dataset_id, split="train", trust_remote_code=True)
            try:
                ds.to_json(str(save_dir / "data.jsonl"))
            except Exception:
                ds.to_parquet(str(save_dir / "data.parquet"))

            return {"success": True, "path": str(save_dir)}
        except ImportError:
            return {"success": False, "error": "huggingface_hub not installed"}
        except Exception as exc:
            logger.error("[%s] HuggingFace error: %s", name, exc, exc_info=True)
            return {"success": False, "error": str(exc)}

    def _try_modelscope(self, name: str) -> dict:
        """Download from ModelScope."""
        try:
            from modelscope.msdatasets import MsDataset

            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
            save_dir = self.download_dir / safe_name
            save_dir.mkdir(parents=True, exist_ok=True)

            ds = MsDataset.load(name, split="train")
            try:
                ds.to_json(str(save_dir / "data.jsonl"))
            except Exception:
                ds.to_parquet(str(save_dir / "data.parquet"))

            return {"success": True, "path": str(save_dir)}
        except ImportError:
            return {"success": False, "error": "modelscope not installed"}
        except Exception as exc:
            logger.error("[%s] ModelScope error: %s", name, exc, exc_info=True)
            return {"success": False, "error": str(exc)}

    def _try_kaggle(self, name: str) -> dict:
        """Download from Kaggle."""
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()

            datasets = list(api.dataset_list(search=name))[:5]
            if not datasets:
                return {"success": False, "error": "No datasets found"}

            best = next((ds for ds in datasets if name.lower() in ds.ref.lower()), None) or datasets[0]
            dataset_ref = best.ref
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in dataset_ref)
            save_dir = self.download_dir / safe_name
            save_dir.mkdir(parents=True, exist_ok=True)

            api.dataset_download_files(dataset_ref, path=str(save_dir), unzip=True)
            return {"success": True, "path": str(save_dir)}
        except ImportError:
            return {"success": False, "error": "kaggle not installed"}
        except Exception as exc:
            logger.error("[%s] Kaggle error: %s", name, exc, exc_info=True)
            return {"success": False, "error": str(exc)}


def download_datasets(dataset_names: list[str], download_dir: str) -> list[dict]:
    """Download datasets into a target directory."""
    downloader = DatasetDownloader(download_dir=download_dir)
    return downloader.download_multiple(dataset_names)
