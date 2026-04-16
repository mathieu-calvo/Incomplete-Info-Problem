# Incomplete-Info-Problem — Architecture & Technology Decisions

This document records the architectural decisions made while building the
Incomplete-Info-Problem (IIP) bot and its companion Streamlit app. Each
section describes **what** was chosen, and — more importantly — **why** that
choice was made for *this* project. The goal is to give a future maintainer
(or a future me) enough context to extend the system confidently or
re-evaluate a decision when requirements change.

---

## 1. Project goals and constraints

IIP is a **heads-up Limit Texas Hold'em (HULHE) bot** trained with Deep CFR
and fine-tuned with PPO self-play, plus a Streamlit app where humans play the
bot and contribute hands to a nightly retraining pipeline. It is the
successor to an earlier DQN/DRQN prototype that lived under `pokerbot/`.

Concrete goals:

- Replace the old DQN prototype with a **theoretically grounded solver**
  (Deep CFR → regret matching → approximate Nash) rather than a value-based
  RL method that ignores the imperfect-information structure of poker.
- Be **free or nearly free** to run, both for training and for hosting the
  play-against-the-bot app. Personal project, no SaaS budget.
- Keep the door open for **No-Limit later** — bet-sizing abstractions live
  behind config, but HULHE is the only target today.
- Remain **fully usable locally** — clone, `pip install -e .[dev,app,train]`,
  `pytest -q`, done. Cloud integrations (Supabase, HF Hub) are optional and
  the code no-ops gracefully when their credentials are missing.
- Prioritize **simple, readable code** over clever abstractions. One-person
  codebase with a handful of players.

These constraints shape every decision below: small scale, low budget,
local-first, cloud-optional, and optimized for one developer's cognitive
load.

---

## 2. Language and build system

### Python 3.11+

**Why Python.** PyTorch, numpy, treys, huggingface_hub, supabase-py and
streamlit are all Python-first. The RL / game-AI ecosystem is Python-first.
Rewriting any of this in another language would be a multi-year effort for
zero research gain.

**Why 3.11+.** Modern typing (`str | None`, `dict[str, ...]` generics,
`from __future__ import annotations`) is cleaner in 3.10+, and 3.11 brings
meaningful interpreter speedups. Pinned `>=3.11` in `pyproject.toml` so I
don't accidentally run on an older interpreter.

### Hatchling + `pyproject.toml` (PEP 621)

**Why hatchling.** It's the default modern Python build backend, zero
boilerplate, plays well with `pip install -e .` for local development. I
didn't need Poetry's lock-file machinery because Streamlit Cloud and GitHub
Actions both install via `pip install -e ".[...]"`, so a second source of
truth (a `poetry.lock`) would only create drift.

### `src/` layout

`src/iip/` instead of a flat `iip/` at the repo root. The `src/` layout
forces you to install the package before importing it, which catches
"works on my machine because of CWD" bugs that a flat layout lets through.
The `pythonpath = ["src"]` entry in `pyproject.toml`'s `[tool.pytest.ini_options]`
also lets `pytest` find the package without a full `pip install -e .` for
quick iteration.

### Optional extras: `[dev]`, `[app]`, `[train]`

- `[dev]` — pytest, ruff, black, mypy, hypothesis.
- `[app]` — streamlit, supabase, huggingface_hub.
- `[train]` — wandb, huggingface_hub.

This means a Streamlit Cloud deploy only needs `pip install -e ".[app]"` and
a GitHub Actions retrain job only needs `pip install -e ".[train,app]"`.
Neither pulls the dev tooling.

---

## 3. Algorithm — Deep CFR, PPO on top

This is the most load-bearing decision in the project. The old repo used
DQN / DRQN, which is a value-based RL algorithm designed for
**perfect-information** or single-agent environments. HULHE is
imperfect-information and adversarial — DQN has no regret, no self-play
equilibrium notion, and plateaus at exploiting one fixed opponent.

### Deep CFR (primary solver)

`src/iip/agents/deep_cfr.py` + `src/iip/train/deepcfr_trainer.py`.

Two MLPs per player:

- **Advantage net** `V_p(I, a)` — regressed on sampled counterfactual
  regrets. Re-initialized each CFR iteration (Brown et al. 2019 recipe).
- **Strategy net** `π̄(I)` — regressed on sampled strategies from a reservoir
  buffer with **linear-CFR** iteration weighting. Represents the average
  strategy, which is what converges to Nash.

Regret matching at each decision node produces the behavioural strategy for
the next traversal. Traversals are **external-sampling MCCFR**: one player
traverses every action, the other samples from the current strategy.

**Why Deep CFR, not tabular CFR.** HULHE has ~10^14 information sets.
Tabular CFR is memory-infeasible; even lossy abstractions (like Johanson's
HULHE solvers) need specialized hand clustering. Deep CFR lets the advantage
net **generalize across infosets** via the feature encoding, so unseen
situations get a principled regret estimate.

**Why Deep CFR, not NFSP.** Heinrich & Silver's NFSP is a neat DQN + SL
combo but converges slower than Deep CFR in practice and is more sensitive
to hyperparameters. Deep CFR has clean theoretical guarantees (regret
minimization → approximate Nash) and was the SOTA baseline at the time
Pluribus and Brown's ReBeL work were published.

**Why not start from an open-source solver (Pluribus, PokerRL).** Those are
substantial frameworks, tightly coupled to specific game variants and
specific abstractions. A focused in-house Deep CFR is ~600 lines, fully
understood, and easier to adapt.

### PPO self-play + league (secondary)

`src/iip/agents/ppo.py` + `src/iip/train/ppo_trainer.py` +
`src/iip/train/league.py`.

Once Deep CFR gives a decent average strategy, PPO takes over for
fine-tuning:

- **Warm-start** the PPO actor from the Deep CFR strategy net
  (`PPOAgent.warm_start_from_deepcfr`).
- **League training**: keep a JSON-persisted pool of past checkpoints plus
  fixed baselines. Sample an opponent per episode via **ELO-softmax** so
  tough checkpoints are picked more often.
- Standard PPO with clip objective, GAE, masked softmax over legal actions,
  entropy bonus.

**Why PPO on top of Deep CFR.** Deep CFR gives a theoretically sound
starting point near Nash. PPO lets the bot **exploit non-Nash opponents**
— real humans who play suboptimally — while the league prevents the bot
from forgetting how to defend against Nash-ish play.

**Why not just PPO from scratch.** Self-play PPO without a regret-minimizing
warm-start is prone to rock-paper-scissors cycles and local optima. Deep
CFR gives a much better initialization and a concrete "defend" opponent
to include in the league.

---

## 4. Engine — single-file HULHE state machine

`src/iip/engine/game.py` consolidates four legacy modules
(`hugame.py`, `headsupgame.py`, `handplayed.py`, `hdplayed.py`) into one
clean state machine:

- `HULHEState` — frozen dataclass carrying pot, stacks, contributions,
  board, hole cards, betting history, street, actor.
- `ActionType` — FOLD, CHECK_CALL, BET_RAISE. The fixed bet size comes from
  street-dependent rules (small bet preflop+flop, big bet turn+river, cap
  at 4 raises).
- `HULHE.step(state, action, rng)` — the single entry point every trainer,
  metric, and UI view funnels through.

**Why consolidate.** The legacy layout had subtle duplication (`hdplayed.py`
and `handplayed.py` were near-identical). One engine with one state
machine makes bugs visible and tests targeted.

**Why frozen dataclasses.** Immutability makes state transitions explicit:
`step` returns a *new* state, so accidental sharing between trainer
traversals (a classic CFR bug) is impossible. The extra allocation cost is
negligible at HULHE tree sizes.

**Kuhn poker sidekick.** `src/iip/engine/kuhn.py` provides a tiny Kuhn
variant used exclusively in `tests/agents/test_deepcfr_kuhn.py` as a
convergence gate for the Deep CFR implementation. Kuhn's Nash is known
analytically, so we can assert exploitability < ε on every CI run before
burning HULHE compute.

---

## 5. Evaluation stack — treys + Monte Carlo equity

`src/iip/eval/ranker.py` uses the `treys` library for 7-card hand
evaluation, with a pure-Python `evaluate_five_card_fallback` used only as a
**ground-truth cross-check** in tests.

**Why treys.** Fast, battle-tested, MIT-licensed, pure-Python (no C
extension build headaches on Windows or Streamlit Cloud). Writing another
7-card ranker from scratch is a several-week yak-shave for zero gain.

**Why keep a pure-Python fallback for tests only.** Defensive cross-check:
if treys ever ships a subtle bug, property tests against an independent
implementation will catch it.

`src/iip/eval/equity.py` adds Monte Carlo equity estimation and the
**169-bucket preflop encoding** (unique isomorphism classes of two-card
hands, accounting for suitedness). The preflop bucket id is a 169-wide
one-hot in the infoset feature vector.

---

## 6. Features — fixed 200-dim infoset vector

`src/iip/features/infoset.py` maps the public+private state to a
fixed-size float vector:

| Block | Dim | What |
|---|---|---|
| Preflop bucket | 169 | One-hot over the canonical preflop classes |
| Board rank multi-hot | 13 | Ranks visible on the board |
| Board suit summary | 4 | Suit counts on the board |
| Street | 4 | One-hot preflop/flop/turn/river |
| Scalars | 7 | Pot, effective stacks, contributions, raise count, position |
| Legal-action mask | 3 | FOLD / CHECK_CALL / BET_RAISE validity |
| **Total** | **200** | |

**Why a fixed-size float vector, not a sequence.** Deep CFR's advantage /
strategy nets are plain MLPs. A Transformer over betting sequences would be
more expressive but 10× slower to train, and the 169-bucket preflop id +
summary stats already capture the information that matters for HULHE. The
betting-sequence information is compressed into the raise count + pot state
rather than represented verbatim.

**Why the legal-action mask is part of the vector, not only applied at the
softmax.** Two reasons: (a) the advantage net benefits from knowing which
actions are legal when regressing sampled regrets, and (b) downstream the
mask is also applied at the action-selection softmax in
`regret_matching` / the PPO actor. Having it in one consistent place
prevents the "trainer disagrees with agent about legal actions" class of
bug.

---

## 7. UI framework — Streamlit

**Chosen:** Streamlit (`>=1.36`), rendered as a multi-page app from
`app/streamlit_app.py` with views under `app/pages/` and shared services
in `app/services/`.

**Alternatives considered:**

| Option | Why not |
|---|---|
| **Gradio** | Great for ML demos, poor for multi-page data-dense apps with custom widgets. |
| **Dash / Plotly** | More flexible but forces callbacks and hand-managed state. Too much plumbing for a small app. |
| **FastAPI + React** | The "real" answer for a production SaaS. For a personal project it's massive overbuild — two languages, two deploys, CORS plumbing. |
| **Notebooks** | No shareable UI for non-technical users. |

**Why Streamlit wins:**

1. **Pure Python, top-to-bottom.** Every view is one `render()` / top-level
   script. I can write analytics, tables, and UI in the same file without
   frontend/backend context-switching.
2. **Free hosting that's purpose-built for it** (Streamlit Cloud — see §8).
3. **Session state + widget model** is sufficient for a small app — no
   Redux needed.
4. **`st.cache_resource` fits the bot-loading pattern exactly** (see §11).

**Trade-offs accepted:**

- Streamlit reruns the whole script on every widget interaction. Fine for
  this workload as long as the bot loader and the Supabase client are
  cached with `@st.cache_resource`.
- Streamlit Cloud free apps sleep after ~15 minutes of inactivity and
  cold-start in ~30 seconds. Fine for a personal tool.

---

## 8. Hosting — Streamlit Community Cloud

**Chosen:** Streamlit Community Cloud (free tier).

**Why.** Same reasons as the sister Portfolio-Simulator project: free,
GitHub-integrated, HTTPS by default, TOML-based secrets manager, no Docker
to maintain.

**Deployment shim.** Streamlit Cloud runs `streamlit run
app/streamlit_app.py`. `pip install -e ".[app]"` happens automatically
because `pyproject.toml` is at the repo root. The `src/iip/` package is
importable because `pip install -e .` adds the `src/` directory to
`sys.path` — no `sys.path` hack needed (unlike Portfolio-Simulator, which
ships without `pip install -e .`).

**Training does NOT run on Streamlit Cloud.** The Cloud tier is CPU-only
with tight memory limits. All training happens either on a local machine
or in the nightly GitHub Actions job (§13). The app only ever **loads a
pretrained checkpoint** and calls `agent.act()` on infosets.

---

## 9. Model storage — Hugging Face Hub

**Chosen:** `huggingface_hub` Python SDK + a private or public model repo
(e.g. `mathieu-calvo/iip-hulhe`) tagged per checkpoint.

`src/iip/io/hf_hub.py` wraps:

- `download_checkpoint(repo_id, tag, filename)` — used by the Streamlit app
  loader.
- `upload_checkpoint(path, repo_id, tag)` — used by the nightly retrain
  workflow.
- `latest_checkpoint_path(repo_id, filename)` — resolves the most recent
  tag matching `ckpt-YYYYMMDD-HHMMSS`.

**Why HF Hub.**

- **Free** for public + private repos up to generous size limits (well
  above a PyTorch `.pt` file).
- **Versioned and tagged** — every nightly retrain pushes a new tag so
  rollback is one string change.
- **`huggingface_hub` is battle-tested** — handles auth, retries,
  resumable downloads, local caching out of the box.
- **Streamlit Cloud already handles the `HF_TOKEN` secret** via
  `st.secrets`; no separate auth plumbing.

**Alternatives considered:**

| Option | Why not |
|---|---|
| **S3 / R2** | Requires credit card or separate billing; signed URLs; more plumbing. |
| **Git LFS in this repo** | LFS quotas are tiny on GitHub's free tier. Would tie model versions to source commits, which is exactly what I *don't* want — the nightly retrain should ship new models without touching source. |
| **Commit `.pt` to the repo** | Bloats git history forever. Hard no. |
| **Self-hosted model server** | A whole new service to babysit. |

---

## 10. Hand logging — Supabase PostgreSQL

**Chosen:** Supabase free tier (500 MB Postgres) + `supabase-py` client
wrapped in `src/iip/io/supabase_client.py`.

**Schema** (documented in the module docstring and re-run-safe):

```sql
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
create policy "inserts from app"    on hands for insert with check (true);
create policy "reads from service role" on hands for select using (true);
```

**Why Supabase.**

- **Free Postgres**, no credit card, good web dashboard.
- **Row-Level Security** lets the Streamlit app use the **anon key** to
  insert hands but NOT read them. The nightly retrain job uses the
  **service-role key** to read back the full dataset. This is the right
  security posture for an anonymous public-facing app.
- **Real SQL** — trivial to query "hands per day", "win rate vs each
  checkpoint", "longest pot" from the Supabase SQL Editor.

**Why this shape of schema.**

- `action_log` and `bot_policies` as `jsonb` — flexible, queryable, and
  naturally shaped for the heterogeneous per-decision data. An enforced
  relational schema would have three extra tables for a gain of near-zero.
- `user_id` is a string (not a foreign key) because the app has **no
  authentication** (see §12). It carries a client-generated anonymous
  session id.
- `bot_checkpoint` is recorded on every hand so we can later filter
  training data to hands played against a specific model version.

**Why not SQLite like Portfolio-Simulator.** Streamlit Cloud's filesystem
is ephemeral — any SQLite DB would be wiped on every cold-start. Hands are
the *primary artifact* of the project, not a cache; they must live in a
real database.

---

## 11. Bot loading — `@st.cache_resource`

`app/services/bot_loader.py` resolves, in order:

1. `HF_REPO_ID` + `HF_TOKEN` in `st.secrets` → latest tag on HF Hub.
2. Local `checkpoints/local/deepcfr.pt` if it exists (dev fallback).
3. Raise with a helpful message.

Decorated with `@st.cache_resource(show_spinner="Loading bot checkpoint…")`
so the PyTorch state dict is only deserialized once per worker, not once
per script rerun.

**Why `@st.cache_resource` and not `@st.cache_data`.** The loaded
`DeepCFRAgent` is a stateful object holding PyTorch modules. It's
**connection-like**, not a data snapshot, so `cache_resource` is the
correct decorator.

---

## 12. Anonymous usage tracking (no auth layer)

A specific product requirement: **"see how many people use the app without
building an authentication layer."** There are a few ways to do this, with
trade-offs:

### Option A — Supabase `sessions` table (**recommended**)

Add a second Supabase table and log one row per Streamlit session. The app
already has the Supabase client; reusing it avoids pulling in another SaaS.

```sql
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
```

App side (pseudocode living in `app/services/session_tracker.py`):

```python
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    HandStore().log_session(
        session_id=st.session_state.session_id,
        user_agent=st.context.headers.get("User-Agent"),
        app_version=APP_VERSION,
    )
```

Count queries:

- **Sessions per day**:
  `select date_trunc('day', created_at), count(*) from sessions group by 1;`
- **Hands per session** (engagement):
  `select session_id, count(*) from hands group by 1 order by 2 desc;`

Trade-offs:

- One row per **browser session**, not per person. A user who opens three
  tabs counts as three. Good enough for "how many people touched the app
  this week".
- No cross-session identity — can't tell if the same person came back
  yesterday. If that matters later, persist a cookie via
  `st.query_params` or `streamlit-cookies-manager` and reuse the id.

### Option B — Privacy-friendly third-party analytics

Embed a tiny script via `st.components.v1.html` or `st.markdown(unsafe_allow_html=True)`:

- **GoatCounter** (free for personal use, open-source, no cookies).
- **Plausible** (paid but reasonable, privacy-friendly).
- **Umami** (self-hosted).

Trade-offs: pulls a third party into the page, needs JavaScript embedding,
and doesn't trivially join against the Supabase `hands` table.

### Option C — Streamlit Cloud's built-in analytics panel

Streamlit Cloud shows a rough "viewers" count in the app's admin panel.
Zero setup, but zero drill-down and not exposed to the app itself.

**Recommendation:** Option A. It's 15 lines of code, reuses existing
infra, gives both a per-day user count and a natural join against the
hands table (sessions-with-hands vs sessions-without tells you drop-off).
Option C is free bonus telemetry on top.

The deployment guide in `docs/deployment-guide.md` includes the `sessions`
table in its setup SQL so this is wired in from day one.

---

## 13. Human-in-the-loop nightly retrain

`scripts/nightly_retrain.py` + `scripts/evaluate_checkpoint.py` +
`.github/workflows/nightly-retrain.yml` form the loop:

1. **Pull** the previous checkpoint from HF Hub.
2. **Fetch** new hands from Supabase since the last run (via
   `HandStore.fetch_hands_since`). Currently used as an **evaluation
   signal**; direct behavioural-cloning injection is noted as future work
   inside the script.
3. **Retrain** with PPO self-play against baselines (`FishAgent`,
   `StrengthHandAgent`, `StartingHandAgent`, `RandomAgent`).
4. **Evaluate** the new checkpoint vs the previous — head-to-head + LBR.
   Fail the run if new regresses by more than `--regression-margin-mbb`
   (default 30 mbb/h).
5. **Publish** on pass: upload `ckpt-YYYYMMDD-HHMMSS` to HF Hub, commit
   `reports/latest.json`.

**Why PPO for retraining, not Deep CFR.** Deep CFR is expensive per
iteration and doesn't benefit much from small amounts of new human data.
PPO on top of an established average strategy is the right fit for
"incorporate the last 24 hours of play without regressing".

**Why the eval gate is automatic rather than advisory.** This is a
public-facing bot; a bad retrain that silently ships to Streamlit would
embarrass the project. Hard gate on regression is the right trade-off
against the rare false positive (a run fails for noise reasons).

**Why the CI job on GitHub Actions, not a self-hosted runner.** Free 2000
minutes/month on public repos is plenty for a nightly run, CPU-only is
sufficient for the PPO fine-tune sizes we use, and there's no runner to
keep alive.

---

## 14. Metrics — mbb/h + LBR exploitability

`src/iip/metrics/mbb.py` implements head-to-head evaluation in
**milli-big-blinds per hand** (mbb/h) — the standard poker-AI unit — with
**seat alternation** (each agent plays both seats over the eval) and a 95%
confidence interval from the standard error.

`src/iip/metrics/exploitability.py` implements **Local Best Response**
(Lisý & Bowling 2017): a cheap 1-ply lookahead over the opponent's
actions, giving a **lower bound** on true exploitability. Full best
response is too expensive for HULHE, and LBR is a well-accepted
regression-guard proxy.

**Why not a proper best-response search.** It's intractable at HULHE tree
sizes without a specialized abstraction. LBR is ~seconds per eval, works
on the raw neural agent, and is exactly what the literature uses for
"did this checkpoint get more/less exploitable".

---

## 15. CLI — Typer

`src/iip/cli.py` exposes `iip train | eval | play` as a single Typer app,
registered as a console script in `pyproject.toml`:

```toml
[project.scripts]
iip = "iip.cli:app"
```

**Why Typer.** Type-annotated function signatures become CLI flags
automatically, help strings are auto-generated from docstrings, and it
layers cleanly on Click. `argparse` would be ~3× more boilerplate for the
same surface area.

Windows note: use `"->"` (ASCII) in log / print strings rather than `"→"`.
The cp1252 default stdout encoding on Windows chokes on the arrow; setting
`PYTHONIOENCODING=utf-8` is a workaround but not a fix.

---

## 16. Testing — pytest

`tests/` is split by concern:

- `tests/engine/` — rules, showdown, side pots, Kuhn.
- `tests/eval/` — ranker cross-check vs pure-Python fallback; MC equity
  property tests.
- `tests/agents/` — fixed-policy sanity, Deep CFR convergence on Kuhn
  (the solver gate).
- `tests/test_metrics.py` — mbb/h and LBR smoke.

25 tests total, ~30 seconds end-to-end on a laptop. Fast enough that CI
on every push is free.

**Why a Kuhn convergence test.** HULHE training runs take hours. A Kuhn
convergence gate takes seconds, has an **analytically known Nash**, and
catches any regression in the Deep CFR implementation immediately. It's
the single most valuable test in the suite.

---

## 17. Secrets and configuration

| File | Purpose | Gitignored? |
|---|---|---|
| `.env` | Local env vars (HF / Supabase keys for scripts) | yes |
| `.streamlit/secrets.toml` | Local mirror of Streamlit Cloud secrets | yes |
| `configs/**/*.yaml` | Hydra configs for training (game, algo, train) | no — committed |
| `.claude/` | Claude Code session state | yes |

Streamlit Cloud secrets are edited in its web UI and injected as
`st.secrets`. GitHub Actions secrets live in the repo settings. Neither
touches the filesystem in source control.

Required secrets:

- Streamlit Cloud: `HF_REPO_ID`, `HF_TOKEN`, `SUPABASE_URL`, `SUPABASE_KEY`
  (anon).
- GitHub Actions (nightly retrain): `HF_REPO_ID`, `HF_TOKEN`,
  `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` (service-role, because it needs
  to **read** hands back out).

---

## 18. What I deliberately left for later

- **Behavioural cloning from human trajectories.** The nightly retrain
  currently uses human hands as an eval signal only. Adding a BC head
  that directly imitates human play is noted as a TODO in
  `scripts/nightly_retrain.py`.
- **No-Limit bet sizing.** The action enum is structured to allow bet-size
  abstractions (0.5·pot, 1·pot, all-in), but HULHE's fixed-size rules are
  the only path exercised today.
- **Opponent modelling.** Inferring per-user style (LAG, TAG, fish) and
  adapting is a natural extension — the `user_id` column in `hands` is
  there for this — but out of scope for v0.2.
- **Exploitability via full best response.** LBR is a lower bound. Full
  best response against the neural policy would be better but needs a
  proper abstraction framework.
- **W&B experiment tracking.** `wandb` is in the `[train]` extras but not
  wired into the trainers yet. Added when the first multi-run sweep
  becomes necessary.
- **User preference persistence in the app.** The play page resets seat
  and bot-checkpoint choice on every cold-start. Would need a cookie or a
  Supabase user-preferences row.

---

## 19. Summary — the one-paragraph version

IIP is a **heads-up Limit Hold'em bot** written in **Python 3.11+** on
**PyTorch**, trained with **Deep CFR** (primary solver) and **PPO
self-play with an ELO-weighted league** (fine-tune), deployed as a free
**Streamlit Community Cloud** app where humans play the bot and hands
land in **Supabase PostgreSQL**. Model checkpoints are versioned and
tagged on **Hugging Face Hub**. A **nightly GitHub Actions job** pulls
new hands, fine-tunes, evaluates against the previous checkpoint with
**LBR exploitability + mbb/h head-to-head**, and publishes a new tag on
pass. Usage tracking is done via an anonymous `sessions` table in
Supabase — no authentication layer. Every cloud integration no-ops
gracefully when its credentials are missing, so the whole stack runs
locally with `pip install -e ".[dev,app,train]" && pytest -q`.
