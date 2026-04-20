# Improvement Workflow

The bot served by the Streamlit app improves through two channels. This
doc is the operating manual for keeping them **additive**, never
destructive, and for making every local training run count. At this
stage, no run should be ephemeral — each either advances the published
chain or is discarded with intent.

Pair with [`training.md`](training.md) (Deep CFR mechanics),
[`evaluation.md`](evaluation.md) (what the gate measures), and
[`deployment-guide.md`](deployment-guide.md) (cloud wiring).

---

## 1. The two channels

| Channel | Source | How it enters the loop |
|---|---|---|
| **RL retraining** (primary) | PPO self-play warm-started from the latest checkpoint, run either by the nightly GitHub Action or locally on your machine. | Uploaded to HF Hub as a new `ckpt-...` tag after passing the eval gate. |
| **Human play** (evaluation signal) | Hands played on the deployed Streamlit app. | Inserted into Supabase `hands` by the app; the nightly retrain reads them. Direct use as training trajectories (behavioural cloning) is future work — see `scripts/nightly_retrain.py`. |

Neither channel is authoritative on its own. The published checkpoint
chain on HF Hub is.

---

## 2. Single source of truth: HF Hub

A checkpoint that is not on HF Hub is invisible to the rest of the
system. Everything else is scratch:

- `checkpoints/local/` is gitignored (`.gitignore:37`) — files there
  never leave your machine.
- The nightly runs on a fresh GitHub Actions runner: it pulls from HF,
  trains, uploads to HF, runner is torn down. It cannot touch your
  local disk.
- The Streamlit app loads the latest HF tag on cold start.

Improving the bot your users see **means publishing a better tag**.
Nothing else counts.

---

## 3. The lineage rule

Every new tag warm-starts from the previous tag. Don't branch the chain
without intent:

1. HF Hub has `ckpt-20260418-030000` (bot currently in prod).
2. Nightly wakes at 03:00 UTC → pulls `ckpt-20260418-030000` as
   `prev.pt` → trains 20 iters of PPO → eval gate → uploads
   `ckpt-20260419-030000` if it passed.
3. App cold-starts next → loads `ckpt-20260419-030000`.

A local run that ignores the current HF `latest` branches off an
outdated base. Your result may beat *your previous local result* while
still being weaker than what prod is serving. **Always re-pull `latest`
before a serious local run.**

---

## 4. Process: incremental local training

Follow this every time you run an `iip train` or a local PPO run you
intend to keep.

### Before training

1. **Refresh `prev.pt` from HF `latest`** — this is your training base:
   ```bash
   export HF_TOKEN=hf_...
   export HF_REPO_ID=your-username/iip-hulhe
   python -c "from iip.io.hf_hub import latest_checkpoint_path; \
     import shutil, os; \
     p = latest_checkpoint_path(os.environ['HF_REPO_ID'], filename='deepcfr.pt'); \
     shutil.copy(p, 'checkpoints/local/prev.pt')"
   ```
   Treat `prev.pt` as read-only — never overwrite it with a training
   output.

2. **Pick a distinct, descriptive output path**:
   ```
   checkpoints/local/cand-<yyyymmdd>-<short-note>.pt
   ```
   e.g. `cand-20260420-ppo-league.pt`. Never use the CLI default
   (`checkpoints/local/deepcfr.pt`) for a run you want to keep — it
   gets clobbered by the next default run.

### During training

- Deep CFR:
  ```bash
  iip train --game hulhe --iters 40 --traversals 5000 \
    --output checkpoints/local/cand-20260420-deepcfr.pt \
    --checkpoint-every 1
  ```
- PPO (reuse the nightly script — same code path, same opponents):
  ```bash
  python scripts/nightly_retrain.py \
    --previous checkpoints/local/prev.pt \
    --output   checkpoints/local/cand-20260420-ppo-league.pt \
    --iters 40 --traversals 5000
  ```
  Supabase env vars are optional — human hands are not yet injected as
  training trajectories.

### After training: mandatory gate

Never publish without running the gate locally:

```bash
python scripts/evaluate_checkpoint.py \
  --checkpoint checkpoints/local/cand-20260420-ppo-league.pt \
  --previous   checkpoints/local/prev.pt \
  --hands 20000 \
  --output reports/local-gate.json
```

Inspect `reports/local-gate.json`:

- **Passed** → publish (next step).
- **Failed** → do not publish. Either keep the candidate on disk for
  later diagnosis, or delete it. If you believe the gate is over-strict
  for a legitimate refactor, raise `--regression-margin-mbb`
  deliberately and note it in the commit that accompanies the push.

### Publishing to HF Hub

Only after the gate passes:

```bash
python -c "from datetime import datetime, timezone; \
  from iip.io.hf_hub import upload_checkpoint; \
  upload_checkpoint('checkpoints/local/cand-20260420-ppo-league.pt', \
                    'your-username/iip-hulhe', \
                    tag='ckpt-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'))"
```

The Streamlit app picks it up on its next cold start. To force a
rotation: Streamlit Cloud → **Manage app → Reboot app**.

---

## 5. Interaction with the nightly

- The nightly always warm-starts from the most recent HF tag. Publish
  mid-day and the next 03:00 UTC run builds on *your* checkpoint.
- The nightly **cannot touch your local disk** — it runs on a tearaway
  GitHub Actions runner. Your `checkpoints/local/` is safe from it.
- The nightly has its own eval gate. If a regression somehow slipped
  past your local gate, the nightly refuses to build on it — but it
  won't roll back. To revert, follow `deployment-guide.md` §9.

---

## 6. Human-play channel — no action needed

Hands flow automatically:

1. User plays a hand on the Streamlit app.
2. App inserts into Supabase `hands` via the anon key (insert-only).
3. Nightly reads the new hands via the service-role key (see
   `scripts/nightly_retrain.py`).

To check volume, see the SQL in `deployment-guide.md` §5.

---

## 7. Anti-patterns

- **Running `iip train` with the default `--output`.** Every default-path
  run clobbers `deepcfr.pt`. Always pass an explicit `cand-...` path.
- **Skipping the `prev.pt` refresh.** A stale `prev.pt` bakes an
  outdated baseline into the gate and into warm-start.
- **Uploading to HF without the eval gate.** The gate is the only thing
  preventing regressions from reaching users.
- **Overwriting a published tag.** Tags are timestamp-unique by
  convention — keep it that way so the history is auditable.
- **"Just exploring" runs.** At this stage, every run should either
  improve the chain or be logged as a deliberate diagnostic. If you
  have nothing to promote, you probably shouldn't have burned the CPU.

---

## 8. Quick checklist

Before you run training locally:

```
[ ] Refreshed checkpoints/local/prev.pt from HF latest
[ ] Chose --output checkpoints/local/cand-<yyyymmdd>-<note>.pt
```

Before you publish:

```
[ ] Ran scripts/evaluate_checkpoint.py vs prev.pt
[ ] Gate passed (reports/local-gate.json)
[ ] Uploaded to HF with a fresh ckpt-<timestamp> tag
[ ] Rebooted the Streamlit app if the change is urgent
```
