"""Push/pull model checkpoints on Hugging Face Hub.

The app pulls the latest tagged checkpoint on startup; the nightly retrain script pushes a
new tag on each successful eval gate. We keep this layer optional — if `huggingface_hub` isn't
installed or `HF_TOKEN` isn't set, the functions raise a clear error and callers fall back to
a local checkpoint path.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    _HAS_HF = True
except Exception:  # pragma: no cover
    _HAS_HF = False


def _require_hf() -> None:
    if not _HAS_HF:
        raise RuntimeError(
            "huggingface_hub not installed. Install with `pip install -e .[app]` or `.[train]`."
        )


def download_checkpoint(repo_id: str, revision: str | None = None, cache_dir: str | None = None) -> Path:
    """Download a checkpoint folder from HF Hub. Returns the local snapshot path."""
    _require_hf()
    path = snapshot_download(repo_id=repo_id, revision=revision, cache_dir=cache_dir)
    return Path(path)


def latest_checkpoint_path(repo_id: str, filename: str = "strategy.pt", cache_dir: str | None = None) -> Path:
    """Download a single file from the latest revision. Cheaper than snapshot_download."""
    _require_hf()
    token = os.environ.get("HF_TOKEN")
    local = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir, token=token)
    return Path(local)


def upload_checkpoint(local_path: str | Path, repo_id: str, tag: str, commit_message: str = "upload ckpt") -> None:
    """Push a local file to `repo_id`, creating the tag if it doesn't exist."""
    _require_hf()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN env var not set — cannot upload checkpoint.")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=Path(local_path).name,
        repo_id=repo_id,
        commit_message=commit_message,
    )
    api.create_tag(repo_id=repo_id, tag=tag, exist_ok=True)
