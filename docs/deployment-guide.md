# Deployment Guide — Streamlit Cloud + Supabase + Hugging Face Hub

This guide walks through deploying the Incomplete-Info-Problem bot as a
24/7 hosted Streamlit app where humans play the bot, hands land in
Supabase, and a nightly GitHub Actions job retrains the bot and publishes
a new checkpoint to Hugging Face Hub.

**Stack:**
- **Hosting:** Streamlit Community Cloud (free)
- **Model storage:** Hugging Face Hub (free)
- **Database:** Supabase PostgreSQL (free tier, 500 MB)
- **Auth:** none — anonymous sessions tracked via Supabase (see Step 4)
- **Retraining:** GitHub Actions nightly cron (free tier)

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

## Step 4: Create the Supabase project and tables

1. Go to https://supabase.com and click **Start your project**.
2. Click **New project**:
   - **Project name**: `iip-hulhe`
   - **Database password**: pick a strong password, save it
   - **Region**: closest to your users (e.g. `West EU (Ireland)`)
3. Click **Create new project** — wait ~2 minutes for it to spin up.

### Create the `hands` and `sessions` tables

1. In the Supabase dashboard, click **SQL Editor** → **New query**.
2. Paste this and click **Run**:

```sql
-- Hands played against the bot.
create table if not exists hands (
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

create index if not exists idx_hands_created_at on hands(created_at);

alter table hands enable row level security;

-- Public app (anon key) can insert, but cannot read back.
create policy "inserts from app" on hands
    for insert with check (true);
-- Service-role key (used only by the retrain script) can read.
create policy "reads from service role" on hands
    for select using (auth.role() = 'service_role');

-- Anonymous usage tracking — one row per Streamlit session.
create table if not exists sessions (
    id uuid primary key default gen_random_uuid(),
    session_id text not null,
    created_at timestamptz default now(),
    user_agent text,
    country text,
    app_version text,
    bot_checkpoint text
);

create index if not exists idx_sessions_created_at on sessions(created_at);

alter table sessions enable row level security;
create policy "inserts from app" on sessions
    for insert with check (true);
create policy "reads from service role" on sessions
    for select using (auth.role() = 'service_role');
```

You should see "Success. No rows returned." — both tables are now live.

### Grab the connection credentials

1. **Project Settings** (gear icon) → **API**.
2. Copy the following, each into its own secure note:
   - **Project URL** (`https://xxxx.supabase.co`)
   - **Project API keys → `anon` `public`** — used by the app (insert-only).
   - **Project API keys → `service_role` `secret`** — used by the retrain
     job (reads hands).

> **Why the split.** The anon key can't read hands back thanks to RLS. The
> service-role key bypasses RLS. Keeping the Streamlit app on the anon key
> means even if it's fully compromised, attackers can't dump the hands
> table.

---

## Step 5: Query usage without an auth layer

Once the app is live, you'll want to see who used it. Because there is
**no login**, we count **sessions** (one per browser tab) and **hands**
(engagement per session).

From the Supabase SQL Editor:

```sql
-- Sessions per day for the last 30 days.
select date_trunc('day', created_at)::date as day, count(*) as sessions
from sessions
where created_at > now() - interval '30 days'
group by 1
order by 1 desc;

-- Hands per session — a rough "engagement" histogram.
select session_id, count(*) as hands
from hands
group by 1
order by 2 desc
limit 50;

-- Sessions that played zero hands (visited but bounced).
select count(*)
from sessions s
left join hands h on h.user_id = s.session_id
where h.id is null;
```

**What this measures:**

- **Sessions** — one per browser tab when someone opens the app. A single
  person opening three tabs counts as three sessions. Good enough for "how
  many people touched the app this week".
- **Hands** — every completed hand against the bot. The `user_id` column in
  `hands` is populated with the session id, so hands-per-session joins
  cleanly.
- **Cannot** tell you if the same person returned yesterday — there is no
  cross-session identity. If that matters later, add a cookie via
  `st.query_params` and reuse the id across visits; that's a one-file
  change to `app/services/session_tracker.py`.

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
5. In the Supabase dashboard → **Table Editor** → `hands`, confirm a new
   row appeared with the hole cards, board, action log, and payoff.
6. In `sessions`, confirm a row appeared with your session id and
   user-agent string.
7. Open the app in a new incognito tab — a **new** session row should
   appear with a different `session_id`.

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

## Troubleshooting

| Symptom | Fix |
|---|---|
| App boots but shows "No bot checkpoint found" | Either the HF Hub creds are missing/wrong or no tags exist on the model repo. Repeat Step 3. |
| App logs show `supabase.AuthApiError` on insert | The `SUPABASE_KEY` on Streamlit Cloud is wrong or is the service-role key (should be anon). |
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
