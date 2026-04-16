"""Load the latest bot checkpoint into the Streamlit session.

Resolution order:
1. `st.secrets["HF_REPO_ID"]` + HF token → download latest tag from Hugging Face Hub.
2. Local path `checkpoints/local/deepcfr.pt` if it exists.
3. Raise — the app can't play without a bot.

Cached with `@st.cache_resource` so the heavy model load happens once per worker.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import streamlit as st

from iip.agents.deep_cfr import DeepCFRAgent

log = logging.getLogger(__name__)


_LOCAL_CANDIDATES = (
    Path("checkpoints/local/deepcfr.pt"),
    Path("checkpoints/local/deepcfr_hulhe.pt"),
)


@st.cache_resource(show_spinner="Loading bot checkpoint…")
def load_bot_and_meta() -> tuple[DeepCFRAgent, dict[str, str]]:
    hf_repo = _get_secret("HF_REPO_ID")
    hf_token = _get_secret("HF_TOKEN")

    if hf_repo and hf_token:
        os.environ["HF_TOKEN"] = hf_token
        try:
            from iip.io.hf_hub import latest_checkpoint_path

            path = latest_checkpoint_path(hf_repo, filename="strategy.pt")
            return DeepCFRAgent.load(path), {
                "source": f"hf:{hf_repo}",
                "revision": path.parent.name,
                "algo": "deepcfr",
            }
        except Exception as e:  # pragma: no cover — fall back if HF is unreachable
            log.warning("HF checkpoint load failed: %s — falling back to local", e)

    for local_path in _LOCAL_CANDIDATES:
        if local_path.exists():
            return DeepCFRAgent.load(local_path), {
                "source": f"local:{local_path.name}",
                "revision": "dev",
                "algo": "deepcfr",
            }

    raise RuntimeError(
        "No bot checkpoint found. Run `iip train --game hulhe --iters 5 --output checkpoints/local/deepcfr.pt` "
        "or set HF_REPO_ID + HF_TOKEN in .streamlit/secrets.toml."
    )


def _get_secret(key: str) -> str | None:
    try:
        v = st.secrets.get(key)
        if v:
            return str(v)
    except Exception:
        pass
    return os.environ.get(key)
