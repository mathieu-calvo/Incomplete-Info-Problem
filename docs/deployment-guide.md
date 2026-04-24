# Deployment Guide — Streamlit Cloud + Supabase + Hugging Face Hub

This guide walks through deploying the Incomplete-Info-Problem bot as a
24/7 hosted Streamlit app where humans play the bot, hands land in
Supabase, and a nightly GitHub Actions job retrains the bot and publishes
a new checkpoint to Hugging Face Hub.

**Stack:**
- **Hosting:** Streamlit Community Cloud (free)
- **Model storage:** Hugging Face Hub (free)
- **Database:** Supabase PostgreSQL (free tier, 500 MB). **Shared** with
  other hobby apps via the schema-per-app layout below — one Supabase
  project hosts Lexico, IIP, and any future small app. Portfolio-Simulator
  stays on its own separate project.
- **Auth:** none — anonymous sessions tracked via `shared.app_events`
  (see Step 4)
- **Retraining:** GitHub Actions nightly cron (free tier)

### Shared-project layout

The Supabase free tier allows two projects per account; to make room for
Portfolio-Simulator and any future small app, IIP shares a project with
Lexico (and future apps) under a **schema-per-app** convention:

| Schema          | Owner app                | What's in it                                       |
|-----------------|--------------------------|----------------------------------------------------|
| `lexico`        | Lexico                   | decks, cards, review logs, LLM usage, liked quotes |
| `iip`           | Incomplete-Info-Problem  | `hands` (the only content table)                   |
| `shared`        | All apps                 | `app_events` — cross-app traffic / engagement log  |

IIP's only content table is `iip.hands`; sessions and other traffic go to
`shared.app_events` so analytics queries span every app with one
`GROUP BY`. Adding a future app is one more schema — no other app changes.

---

## Prerequisites

- A GitHub account with this repo pushed to it
- A Hugging Face account
- A Supabase account (sign up with GitHub is fine)
- Python 3.11+ locally, for the one-off initial checkpoint upload

Install the package locally with training extras:

```bash
pip install -e ".[dev,app,train]"
```

---

## Step 1: Commit and push the code to GitHub

Make sure all changes are committed and pushed:

```bash
cd "C:/Users/mathi/Documents/Github repos/Incomplete-Info-Problem"
git add -A
git commit -m "your commit message"
git push origin master
```

---

## Step 2: Create the Hugging Face model repo

1. Go to https://huggingface.co and log in.
2. Click your avatar → **New model**.
3. Fill in:
   - **Owner**: your username (e.g. `mathieu-calvo`)
   - **Model name**: `iip-hulhe`
   - **Visibility**: Private is fine; Public is also fine if you don't mind
     people downloading your bot.
4. Click **Create model**. Note the full repo id — `your-username/iip-hulhe`.
5. Go to your profile → **Settings** → **Access Tokens** → **New token**:
   - **Name**: `iip-write`
   - **Role**: `Write` (the app needs Read; the retrain job needs Write)
6. Copy the token starting with `hf_...` — you'll need it in Steps 3 and 6.

---

## Step 3: Train and upload an initial checkpoint

The Streamlit app needs a checkpoint to load on first boot. Train a small
one locally and push it:

```bash
iip train --game hulhe --iters 5 --traversals 1000 --output checkpoints/local/deepcfr.pt
```

This runs in ~1–2 minutes and produces a playable-but-weak bot. For a real
checkpoint you want `--iters 50+ --traversals 10000+`, which takes hours
on CPU.

Upload the checkpoint:

```bash
export HF_TOKEN=hf_your_token_here
python -c "from iip.io.hf_hub import upload_checkpoint; \
  upload_checkpoint('checkpoints/local/deepcfr.pt', 'your-username/iip-hulhe', tag='ckpt-initial')"
```

Verify in the Hugging Face web UI that the tag `ckpt-initial` exists and
contains `strategy.pt`.

---

## Step 4: Set up the shared Supabase project

**If this is the first hobby app you're deploying**, create the shared
project:

1. Go to https://supabase.com and click **Start your project**.
2. Click **New project**:
   - **Project name**: `hobby-apps` (anything — it will host multiple apps).
   - **Database password**: pick a strong password, save it.
   - **Region**: closest to your users (e.g. `West EU (Ireland)`).
3. Click **Create new project** — wait ~2 minutes for it to spin up.

**If you've already set up the shared project for another app** (e.g.
Lexico), skip project creation and reuse the existing one.

### Create IIP's schema

1. In the Supabase dashboard, click **SQL Editor** → **New query**.
2. Paste the contents of
   [`scripts/supabase_schema.sql`](../scripts/supabase_schema.sql) and click
   **Run**. The script is idempotent — safe to re-run on schema changes.

You should see "Success. No rows returned." The SQL creates:

- the `iip` and `shared` schemas (if absent),
- `iip.hands` + RLS policies (anon inserts, service-role reads/updates),
- `shared.app_events` (if absent — idempotent across apps) + RLS.

> **Why the split.** The anon key can insert into `iip.hands` and
> `shared.app_events` but cannot read them back (RLS blocks SELECT for
> non-service-role users). The retrain job uses the service-role key,
> bypasses RLS, and can both read hands and mark them as consumed.

### Expose the schemas to the REST API

The `supabase-py` client (used by the Streamlit app + retrain script) goes
through PostgREST, which only sees schemas explicitly listed in the API
config. One-time setup:

1. **Project Settings** → **API** → **Exposed schemas**.
2. Add `iip` and `shared` to the comma-separated list (alongside the
   default `public`). Final value: `public, iip, shared`.
3. Click **Save** and wait ~30 s for PostgREST to reload.

### Grab the connection credentials

1. **Project Settings** (gear icon) → **API**.
2. Copy the following, each into its own secure note:
   - **Project URL** (`https://xxxx.supabase.co`)
   - **Project API keys → `anon` `public`** — used by the app (insert-only).
   - **Project API keys → `service_role` `secret`** — used by the retrain
     job (reads hands, marks them as consumed).

> **Key distinction.** The anon key can't read `iip.hands` thanks to RLS.
> The service-role key bypasses RLS. Keeping the Streamlit app on the
> anon key means even if it's fully compromised, attackers can't dump the
> hands table.

---

## Step 5: Query usage without an auth layer

Once the app is live, you'll want to see who used it. Because there is
**no login**, we count **sessions** (one per browser tab, logged as
`session_start` events in `shared.app_events`) and **hands** (engagement
per session, from `iip.hands`).

From the Supabase SQL Editor:

```sql
-- Sessions per day for the last 30 days (IIP only).
SELECT date_trunc('day', occurred_at)::date AS day, count(*) AS sessions
FROM shared.app_events
WHERE app = 'iip' AND event = 'session_start'
  AND occurred_at > now() - interval '30 days'
GROUP BY 1
ORDER BY 1 DESC;

-- Hands per session — a rough "engagement" histogram.
SELECT user_id AS session_id, count(*) AS hands
FROM iip.hands
GROUP BY 1
ORDER BY 2 DESC
LIMIT 50;

-- Sessions that played zero hands (visited but bounced).
SELECT count(*)
FROM shared.app_events e
LEFT JOIN iip.hands h ON h.user_id = e.session_id
WHERE e.app = 'iip' AND e.event = 'session_start' AND h.id IS NULL;

-- Cross-app traffic overview (works because every app writes to the same table).
SELECT app, date_trunc('day', occurred_at)::date AS day,
       count(DISTINCT session_id) AS sessions,
       count(*) AS events
FROM shared.app_events
WHERE occurred_at > now() - interval '30 days'
GROUP BY 1, 2
ORDER BY 2 DESC, 1;
```

**What this measures:**

- **Sessions** — one `session_start` row per browser tab on first render.
  Three tabs = three sessions. Good enough for "how many people touched
  the app this week". Same grain across every app on the project.
- **Hands** — every completed hand against the bot (in `iip.hands`). The
  `user_id` column is populated with the session id, so hands-per-session
  joins cleanly with `shared.app_events`.
- **Cannot** tell you if the same person returned yesterday — there is no
  cross-session identity. If that matters later, persist the UUID to
  browser localStorage via `streamlit-js-eval`; that's a one-file change
  to `app/services/session_tracker.py`.

Streamlit Cloud's admin panel also shows a rough "viewers" count — free
bonus telemetry on top of the Supabase queries above.

---

## Step 6: Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **New app**.
3. Fill in:
   - **Repository**: `your-username/Incomplete-Info-Problem`
   - **Branch**: `master`
   - **Main file path**: `app/streamlit_app.py`
4. Click **Advanced settings**.
5. Set **Python version** to `3.11`.
6. In the **Secrets** text box, paste (replacing placeholders with real
   values from Steps 2 and 4):

```toml
HF_REPO_ID = "your-username/iip-hulhe"
HF_TOKEN   = "hf_your_read_or_write_token_here"

SUPABASE_URL = "https://xxxxxxxxxxxx.supabase.co"
SUPABASE_KEY = "eyJhbGciOi...your_anon_public_key..."

APP_VERSION = "0.2.0"
```

> **Important:** `SUPABASE_KEY` on Streamlit Cloud must be the **anon**
> key, NOT the service-role key. The service-role key belongs only in
> GitHub Actions secrets.

7. Click **Deploy!**

---

## Step 7: Verify the app works end-to-end

1. Wait 2–3 minutes for the build. Watch the logs in the Cloud dashboard.
2. Once deployed, you'll get a URL like
   `https://incomplete-info-problem-<hash>.streamlit.app`.
3. Open it — the app should show the table with you sitting at one seat
   and the bot at the other.
4. Play one complete hand to showdown.
5. In the Supabase dashboard → **Table Editor**, switch the schema
   selector to `iip` → `hands`, confirm a new row appeared with the hole
   cards, board, action log, and payoff.
6. Switch the schema selector to `shared` → `app_events`, confirm a row
   with `app='iip'` and `event='session_start'` appeared.
7. Open the app in a new incognito tab — a **new** `session_start` row
   should appear in `shared.app_events` with a different `session_id`.

If any of these fail, the Streamlit Cloud **Manage app → Logs** pane is
the first place to look.

---

## Step 8: Wire up the nightly retrain pipeline

The GitHub Actions workflow at `.github/workflows/nightly-retrain.yml`
runs every night at 03:00 UTC and:

1. Pulls the latest checkpoint from HF Hub.
2. Fetches new hands from Supabase since the last run.
3. Runs `scripts/nightly_retrain.py` (PPO self-play).
4. Runs `scripts/evaluate_checkpoint.py` (head-to-head + LBR eval gate).
5. On pass, uploads a new `ckpt-YYYYMMDD-HHMMSS` tag to HF Hub and
   commits `reports/latest.json`.
6. The Streamlit app picks up the new tag on its next cold-start.

### Configure the GitHub repo secrets

1. Go to your repo on GitHub → **Settings** → **Secrets and variables** →
   **Actions** → **New repository secret**.
2. Add these four secrets:

| Name | Value |
|---|---|
| `HF_REPO_ID` | `your-username/iip-hulhe` |
| `HF_TOKEN` | Hugging Face token with **Write** role (from Step 2) |
| `SUPABASE_URL` | Your Supabase project URL (from Step 4) |
| `SUPABASE_SERVICE_KEY` | Supabase **service_role** key (from Step 4) |

> **Why the service-role key in GitHub Actions but the anon key on
> Streamlit Cloud.** The retrain job needs to **read** new hands to
> train on them. The Streamlit app only needs to **insert**. Keeping the
> service-role key out of the client-side stack means even if the app
> is compromised, the hands table can't be exfiltrated.

3. To test the workflow without waiting for 03:00 UTC, go to **Actions**
   → **Nightly retrain** → **Run workflow**. A manual dispatch runs the
   full pipeline on demand.

---

## Step 9: Rotating the bot in production

Whenever a nightly retrain passes the eval gate, a new tag
`ckpt-YYYYMMDD-HHMMSS` lands on HF Hub. The Streamlit app's
`@st.cache_resource`-decorated bot loader picks it up **on next cold-start**.

To force a rotation right now:

1. Streamlit Cloud dashboard → your app → **Manage app** → **Reboot app**.
2. The first page load after the reboot will pull the latest tag.

If a retrain produces a regression you want to roll back:

```bash
# List checkpoints on the HF repo.
huggingface-cli repo info your-username/iip-hulhe --type model

# Point the app at a specific older tag by editing `latest_checkpoint_path`
# in src/iip/io/hf_hub.py, or by overwriting the default tag with an
# explicit HF_CHECKPOINT_TAG secret on Streamlit Cloud (if you add one).
```

---

## Step 10: Local development (unchanged)

None of the above affects local development. With no environment variables
or `.streamlit/secrets.toml` set, the app:

- Falls back to `checkpoints/local/deepcfr.pt` instead of HF Hub.
- Skips Supabase logging entirely (`HandStore.is_configured == False`).
- Skips session tracking for the same reason.

```bash
# Train a small checkpoint (required once).
iip train --game hulhe --iters 3 --traversals 1000 --output checkpoints/local/deepcfr.pt

# Run the app.
streamlit run app/streamlit_app.py
```

To test the cloud integration locally, create `.streamlit/secrets.toml`
(gitignored) with the same format as the Streamlit Cloud secrets box
above.

---

## Storage hygiene (shared 500 MB budget)

The whole shared project shares the free tier's 500 MB ceiling. Because
`iip.hands` is the fastest-growing table across the project, watch its
size and purge consumed rows periodically.

```sql
-- Per-schema size.
SELECT schemaname,
       pg_size_pretty(sum(pg_total_relation_size(schemaname||'.'||tablename))::bigint) AS size
FROM pg_tables
WHERE schemaname IN ('lexico', 'iip', 'shared')
GROUP BY schemaname
ORDER BY sum(pg_total_relation_size(schemaname||'.'||tablename)) DESC;

-- Hands stats (how many are still unconsumed by training?).
SELECT
    count(*) AS total,
    count(*) FILTER (WHERE used_for_training_at IS NOT NULL) AS consumed,
    pg_size_pretty(pg_total_relation_size('iip.hands')) AS size
FROM iip.hands;
```

After each successful nightly retrain, the retrain script should mark the
hands it trained on:

```sql
-- Run at end of nightly_retrain.py after upload succeeds.
UPDATE iip.hands
SET used_for_training_at = now()
WHERE used_for_training_at IS NULL
  AND created_at <= :fetched_until_ts;
```

Then a weekly (or monthly) cleanup deletes old consumed rows:

```sql
-- Keep 30 days of consumed hands for safety, drop the rest.
DELETE FROM iip.hands
WHERE used_for_training_at IS NOT NULL
  AND used_for_training_at < now() - interval '30 days';
```

If storage ever runs hot even with purging, the cheapest next cut is to
set `bot_policies = NULL` on already-consumed rows \u2014 policies are
recomputable from checkpoint + action_log, and they're the biggest column.

---

## Migration from the legacy (per-app) Supabase project

If you're moving IIP off its old dedicated Supabase project onto the
shared one, the steps are:

1. Run the new `scripts/supabase_schema.sql` on the shared project (Step 4).
2. Dump `hands` from the old project (Table Editor \u2192 `hands` \u2192 Export
   \u2192 CSV) and import into `iip.hands` on the shared project (SQL editor
   `\copy` or the Table Editor's import UI). The column set matches; add
   `used_for_training_at = NULL` on import.
3. Old `sessions` rows can be backfilled into `shared.app_events` as
   `session_start` events, or discarded \u2014 they're pure telemetry.
4. Update Streamlit Cloud secrets + GitHub Actions secrets to the shared
   project's URL and keys.
5. Delete the old Supabase project to free up the free-tier slot (e.g.
   for Portfolio-Simulator).

> **Code follow-up required.** The current `src/iip/io/supabase_client.py`
> writes to `public.hands` / `public.sessions`. For the shared project to
> work, two small changes are needed:
>
> - Qualify the hands table with the `iip` schema:
>   `self._client.schema("iip").table("hands")` in both `insert_hand` and
>   `fetch_hands_since`.
> - Replace `log_session(...)` so it inserts one row into
>   `shared.app_events` with `app='iip'`, `event='session_start'`,
>   `user_id=session_id`, and the old session fields folded into `meta`:
>   `self._client.schema("shared").table("app_events").insert({...})`.
>
> The app-facing API (`HandStore.log_session`, `HandStore.insert_hand`,
> `HandStore.fetch_hands_since`) stays the same, so callers don't change.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| App boots but shows "No bot checkpoint found" | Either the HF Hub creds are missing/wrong or no tags exist on the model repo. Repeat Step 3. |
| App logs show `supabase.AuthApiError` on insert | The `SUPABASE_KEY` on Streamlit Cloud is wrong or is the service-role key (should be anon). |
| `schema "iip" does not exist` or `relation "iip.hands" does not exist` via the REST API | You forgot to add `iip, shared` to **Project Settings \u2192 API \u2192 Exposed schemas**. Add them, save, wait ~30 s. |
| Inserts return `new row violates row-level security policy` | The client is not using `.schema("iip")` / `.schema("shared")` \u2014 the default `public` schema has no RLS policy for these tables. Apply the code follow-up above. |
| Nightly retrain fails with `AuthError: Not enough permissions` | The `HF_TOKEN` secret on GitHub is a Read token, not a Write token. Regenerate with Write role. |
| Nightly retrain fails reading hands | The `SUPABASE_SERVICE_KEY` secret on GitHub is the anon key, not the service-role key. RLS blocks anon reads by design. |
| UnicodeEncodeError with `'\u2192'` on Windows CLI | Set `PYTHONIOENCODING=utf-8` before running `iip ...`, or use an ASCII shell. Cloud Linux is unaffected. |
| App cold-start is slow (~30s) | Streamlit Cloud free tier sleeps after ~15min of inactivity. Expected and harmless. |

---

## Architecture notes

- **Config switch:** the app auto-detects Supabase + HF from secrets. If
  they're absent, the app runs in local-only mode.
- **Session isolation:** anonymous sessions are tracked by a UUID in
  `st.session_state`, not a cookie — closing the tab ends the session.
- **Model storage cost:** one `.pt` file per tagged checkpoint on HF Hub.
  HF's free quota comfortably fits months of nightly retrains.
- **Retrain compute:** CPU-only on GitHub's hosted runners. A full nightly
  (20 iters × 5000 hands) comfortably fits in the 2-hour workflow limit.
- **Retrain budget:** free tier GitHub Actions gives 2000 minutes/month
  on public repos — 30 nights × ~45 min each fits with headroom.
