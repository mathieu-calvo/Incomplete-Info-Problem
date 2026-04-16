"""Supabase client for logging hands played in the Streamlit app and reading them back for retraining.

Table schema (create once in the Supabase SQL editor):

    create table hands (
        id uuid primary key default gen_random_uuid(),
        created_at timestamptz default now(),
        user_id text,
        hero_seat int2 not null,
        bot_checkpoint text,
        hole_cards_hero text,
        hole_cards_bot text,
        board text,
        action_log jsonb not null,
        bot_policies jsonb,
        payoff_hero int4,
        payoff_bot int4
    );

    alter table hands enable row level security;
    create policy "inserts from app" on hands for insert with check (true);
    create policy "reads from service role" on hands for select using (true);

    create table sessions (
        id uuid primary key default gen_random_uuid(),
        session_id text not null,
        created_at timestamptz default now(),
        user_agent text,
        country text,
        app_version text,
        bot_checkpoint text
    );
    create index idx_sessions_created_at on sessions(created_at);
    alter table sessions enable row level security;
    create policy "inserts from app" on sessions for insert with check (true);
    create policy "reads from service role" on sessions for select using (true);

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

    def insert_hand(self, rec: HandRecord) -> dict[str, Any] | None:
        if not self.is_configured:
            return None
        payload = asdict(rec)
        payload["action_log"] = json.dumps(rec.action_log)
        payload["bot_policies"] = json.dumps(rec.bot_policies)
        resp = self._client.table("hands").insert(payload).execute()  # type: ignore[union-attr]
        return resp.data[0] if resp.data else None

    def log_session(
        self,
        session_id: str,
        user_agent: str | None = None,
        country: str | None = None,
        app_version: str | None = None,
        bot_checkpoint: str | None = None,
    ) -> dict[str, Any] | None:
        if not self.is_configured:
            return None
        payload = {
            "session_id": session_id,
            "user_agent": user_agent,
            "country": country,
            "app_version": app_version,
            "bot_checkpoint": bot_checkpoint,
        }
        resp = self._client.table("sessions").insert(payload).execute()  # type: ignore[union-attr]
        return resp.data[0] if resp.data else None

    def fetch_hands_since(self, since: datetime | None = None, limit: int = 10_000) -> list[dict[str, Any]]:
        if not self.is_configured:
            return []
        q = self._client.table("hands").select("*").order("created_at", desc=False).limit(limit)  # type: ignore[union-attr]
        if since is not None:
            q = q.gte("created_at", since.astimezone(UTC).isoformat())
        resp = q.execute()
        return list(resp.data or [])
