-- Supabase / PostgreSQL schema for Incomplete-Info-Problem (IIP) on the
-- SHARED `hobby-apps` project.
--
-- This project hosts multiple small apps (Lexico, IIP, and any future
-- hobby app) under one Supabase free-tier project. Each app owns its own
-- schema so tables never collide, and all apps write lightweight traffic
-- events to a single `shared.app_events` table.
--
-- Portfolio-Simulator stays on its OWN Supabase project.
--
-- Paste this entire file into the Supabase SQL editor and click Run.
-- Safe to re-run: every CREATE uses IF NOT EXISTS and every CREATE POLICY
-- is guarded by a drop-if-exists, so re-running on schema changes is fine.

-- ---------------------------------------------------------------------------
-- Schemas
-- ---------------------------------------------------------------------------

CREATE SCHEMA IF NOT EXISTS iip;
CREATE SCHEMA IF NOT EXISTS shared;

-- ---------------------------------------------------------------------------
-- iip.hands — every hand a human plays against the bot
--
-- This is the biggest growth table in the shared project. Storage tips:
--   - Running the nightly retrain sets `used_for_training_at`. A periodic
--     purge (see deployment-guide.md "Storage hygiene") deletes rows that
--     have been consumed AND are older than 30 days.
--   - `bot_policies` JSONB is the largest column. If storage gets tight
--     we can drop it — policies are recomputable from the checkpoint +
--     action_log. Keep it for now; it's useful for offline study.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS iip.hands (
    id                     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_id                TEXT,
    hero_seat              INT2 NOT NULL,
    bot_checkpoint         TEXT,
    hole_cards_hero        TEXT,
    hole_cards_bot         TEXT,
    board                  TEXT,
    action_log             JSONB NOT NULL,
    bot_policies           JSONB,
    payoff_hero            INT4,
    payoff_bot             INT4,
    used_for_training_at   TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_hands_created_at
    ON iip.hands (created_at);
CREATE INDEX IF NOT EXISTS idx_hands_training_cleanup
    ON iip.hands (used_for_training_at, created_at)
    WHERE used_for_training_at IS NOT NULL;

ALTER TABLE iip.hands ENABLE ROW LEVEL SECURITY;

-- Anon key (used by the Streamlit app) can insert but cannot read back.
DROP POLICY IF EXISTS "inserts from app" ON iip.hands;
CREATE POLICY "inserts from app"
    ON iip.hands FOR INSERT WITH CHECK (true);

-- Service-role key (used only by the nightly retrain script) can read.
DROP POLICY IF EXISTS "reads from service role" ON iip.hands;
CREATE POLICY "reads from service role"
    ON iip.hands FOR SELECT USING (auth.role() = 'service_role');

-- Service-role can also mark rows as consumed by retraining.
DROP POLICY IF EXISTS "training marks from service role" ON iip.hands;
CREATE POLICY "training marks from service role"
    ON iip.hands FOR UPDATE USING (auth.role() = 'service_role');

-- NOTE: the old `iip.sessions` table has been removed. Session tracking
-- now writes one `session_start` row per tab to `shared.app_events` below.

-- ---------------------------------------------------------------------------
-- shared.app_events — cross-app traffic / engagement log
--
-- ONE row per interesting event across every app on this Supabase project.
-- For IIP, expect two events:
--   - session_start  (one per browser tab, on first render)
--   - hand_played    (one per completed hand — optional; hands themselves
--                     already live in iip.hands, so logging the event here
--                     is purely for unified analytics)
--
-- `user_id` is app-specific. For IIP it's the browser-session UUID
-- (same as `session_id`). For Lexico it's the login username.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS shared.app_events (
    id           BIGSERIAL PRIMARY KEY,
    app          TEXT NOT NULL,
    event        TEXT NOT NULL,
    user_id      TEXT,
    session_id   TEXT,
    app_version  TEXT,
    occurred_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    meta         JSONB
);

CREATE INDEX IF NOT EXISTS idx_app_events_app_time
    ON shared.app_events (app, occurred_at);
CREATE INDEX IF NOT EXISTS idx_app_events_app_user_time
    ON shared.app_events (app, user_id, occurred_at);

ALTER TABLE shared.app_events ENABLE ROW LEVEL SECURITY;

-- Anon key (the public-facing Streamlit apps) can insert only. This is
-- safe: each row costs ~100 bytes, so a compromised key can at worst bloat
-- this table — nothing in the iip.hands content is reachable.
DROP POLICY IF EXISTS "inserts from app" ON shared.app_events;
CREATE POLICY "inserts from app"
    ON shared.app_events FOR INSERT WITH CHECK (true);

-- Service-role (you, via the SQL editor) can read everything.
DROP POLICY IF EXISTS "reads from service role" ON shared.app_events;
CREATE POLICY "reads from service role"
    ON shared.app_events FOR SELECT USING (auth.role() = 'service_role');

-- ---------------------------------------------------------------------------
-- Exposing the schemas to PostgREST (one-time, via the Supabase dashboard)
-- ---------------------------------------------------------------------------
-- The Supabase JS-style REST API (`supabase-py` client + anon key) only
-- reaches tables in schemas listed in `Project Settings → API → Exposed
-- schemas`. Add `iip, shared` alongside the default `public`, save, and
-- wait ~30 s for PostgREST to reload. See deployment-guide.md Step 4.
