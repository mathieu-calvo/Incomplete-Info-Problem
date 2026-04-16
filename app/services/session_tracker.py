"""Anonymous usage tracking — one Supabase `sessions` row per Streamlit session.

The UUID is generated client-side on first render per browser tab and stashed in
`st.session_state`. Scope implications:
    - Different browsers / tabs = different UUIDs (so User A and User B are distinguished).
    - Same tab across reruns = same UUID (correctly counted as one session).
    - New tab / incognito / next day = new UUID (no cross-session identity today).

To upgrade to persistent identity (same human across visits), persist the UUID to browser
localStorage via `streamlit-js-eval` or a cookie manager and rehydrate on boot.
"""

from __future__ import annotations

import logging
import uuid

import streamlit as st

from iip.io.supabase_client import HandStore

log = logging.getLogger(__name__)

_SESSION_ID_KEY = "iip_session_id"
_TRACKED_KEY = "iip_session_tracked"


def get_session_id() -> str:
    """Return a stable UUID for the current Streamlit session. Generated on first call."""
    if _SESSION_ID_KEY not in st.session_state:
        st.session_state[_SESSION_ID_KEY] = str(uuid.uuid4())
    return st.session_state[_SESSION_ID_KEY]


def track_session_once(app_version: str | None = None, bot_checkpoint: str | None = None) -> None:
    """Insert one `sessions` row per Streamlit session. No-op if Supabase isn't configured."""
    if st.session_state.get(_TRACKED_KEY):
        return
    session_id = get_session_id()
    store = HandStore()
    if not store.is_configured:
        st.session_state[_TRACKED_KEY] = True
        return
    try:
        store.log_session(
            session_id=session_id,
            user_agent=_user_agent(),
            app_version=app_version,
            bot_checkpoint=bot_checkpoint,
        )
    except Exception as e:  # pragma: no cover — telemetry never blocks the UI
        log.warning("session tracking failed: %s", e)
    st.session_state[_TRACKED_KEY] = True


def _user_agent() -> str | None:
    try:
        return st.context.headers.get("User-Agent")
    except Exception:
        return None
