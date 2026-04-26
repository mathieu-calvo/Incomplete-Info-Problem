"""Supabase client for logging hands played in the Streamlit app and reading them back for retraining.

The Supabase project is shared with other hobby apps under a schema-per-app
layout (see ``scripts/supabase_schema.sql`` and ``docs/deployment-guide.md``):

    - ``iip.hands``          — one row per completed hand
    - ``shared.app_events``  — one row per interesting event across all apps
                                (session_start for IIP; session tracking lives
                                here rather than a dedicated iip.sessions table)

The REST client reaches those tables via ``.schema("iip")`` / ``.schema("shared")``;
``iip`` and ``shared`` must be listed in Project Settings → API → Exposed schemas.

Environment variables:
    SUPABASE_URL
    SUPABASE_KEY (anon for inserts from the app; service-role for the retrain script)
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

try:
    from supabase import Client, create_client
    _HAS_SB = True
except Exception:  # pragma: no cover
    _HAS_SB = False


@dataclass
class HandRecord:
    user_id: str
    hero_seat: int
    bot_checkpoint: str
    hole_cards_hero: str
    hole_cards_bot: str
    board: str
    action_log: list[dict[str, Any]]
    bot_policies: list[dict[str, float]]
    payoff_hero: int
    payoff_bot: int


class HandStore:
    """Lazy wrapper around a Supabase client. No-ops gracefully if creds are missing."""

    def __init__(self, url: str | None = None, key: str | None = None) -> None:
        self.url = url or os.environ.get("SUPABASE_URL")
        self.key = key or os.environ.get("SUPABASE_KEY")
        self._client: Client | None = None
        if _HAS_SB and self.url and self.key:
            self._client = create_client(self.url, self.key)

    @property
    def is_configured(self) -> bool:
        return self._client is not None

    def insert_hand(self, rec: HandRecord) -> bool:
        # `returning="minimal"` sends Prefer: return=minimal so PostgREST skips
        # the RETURNING clause. Anon has INSERT but not SELECT on iip.hands —
        # without this the insert errors with 42501 "permission denied for table hands".
        if not self.is_configured:
            return False
        payload = asdict(rec)
        payload["action_log"] = json.dumps(rec.action_log)
        payload["bot_policies"] = json.dumps(rec.bot_policies)
        self._client.schema("iip").table("hands").insert(  # type: ignore[union-attr]
            payload, returning="minimal"
        ).execute()
        return True

    def log_session(
        self,
        session_id: str,
        user_agent: str | None = None,
        country: str | None = None,
        app_version: str | None = None,
        bot_checkpoint: str | None = None,
    ) -> bool:
        """Log one ``session_start`` row to ``shared.app_events``.

        user_id mirrors session_id for IIP (no login, so the session UUID is
        the closest thing to an identity). Everything else lands in ``meta``
        so ``shared.app_events`` stays the same tiny shape across apps.
        """
        if not self.is_configured:
            return False
        payload = {
            "app": "iip",
            "event": "session_start",
            "user_id": session_id,
            "session_id": session_id,
            "app_version": app_version,
            "meta": {
                "user_agent": user_agent,
                "country": country,
                "bot_checkpoint": bot_checkpoint,
            },
        }
        self._client.schema("shared").table("app_events").insert(  # type: ignore[union-attr]
            payload, returning="minimal"
        ).execute()
        return True

    def fetch_hands_since(self, since: datetime | None = None, limit: int = 10_000) -> list[dict[str, Any]]:
        if not self.is_configured:
            return []
        q = (
            self._client.schema("iip")  # type: ignore[union-attr]
            .table("hands")
            .select("*")
            .order("created_at", desc=False)
            .limit(limit)
        )
        if since is not None:
            q = q.gte("created_at", since.astimezone(UTC).isoformat())
        resp = q.execute()
        return list(resp.data or [])
