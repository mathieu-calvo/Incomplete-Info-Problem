# Incomplete-Info-Problem

A heads-up Limit Texas Hold'em bot trained with **Deep CFR** and fine-tuned with **PPO self-play**, plus a **Streamlit Cloud** app where humans play the bot and contribute hands to a nightly retraining pipeline.

> Successor to the original DQN/DRQN prototype. The prior TF/Keras implementation and fixed-policy evaluation have been replaced by a theoretically-grounded solver (Deep CFR → regret matching → approximate Nash) + a human-in-the-loop improvement cycle.

## Highlights

- **Deep CFR** — neural advantage + strategy networks trained via external-sampling MCCFR (Brown et al., 2019). Converges toward a Nash equilibrium strategy.
- **PPO self-play league** — warm-started from the Deep CFR average strategy, opponents sampled from an ELO-weighted checkpoint pool.
- **Fast engine** — clean state-machine HULHE, `treys`-backed 7-card ranker, Monte-Carlo equity, 169-bucket preflop encoder.
- **Human-in-the-loop** — every hand you play on the web app is saved to Supabase; a nightly GitHub Actions job retrains the bot and publishes a new checkpoint on Hugging Face Hub after passing an eval gate.
- **Streamlit Cloud** — one-click deploy from this repo; the app auto-pulls the latest bot version on boot.

## Repo layout

```
configs/                 Hydra configs (game, algo, train)
src/iip/
  engine/                HULHE + Kuhn state machines, cards, deck
  eval/                  treys-backed ranker + MC equity + 169-bucket preflop
  features/              infoset -> fixed-size tensor
  agents/                base protocol, random, fixed-policy, Deep CFR, PPO
  train/                 trainers (Deep CFR, PPO), league, reservoir buffers
  metrics/               mbb/h, LBR exploitability
  io/                    HF Hub + Supabase clients
  cli.py                 `iip train | eval | play`
app/                     Streamlit Cloud app (play vs bot + leaderboard + insights)
scripts/                 nightly_retrain.py, evaluate_checkpoint.py
tests/                   pytest suites (engine, eval, agents, metrics)
notebooks/               run_experiments.ipynb
.github/workflows/       CI + nightly retrain
```

## Quickstart

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows: .venv/Scripts/activate
pip install -e ".[dev,app,train]"

# Verify the build.
pytest -q

# Train a small Deep CFR checkpoint on HULHE (fast smoke run).
iip train --game hulhe --iters 3 --traversals 1000 --output checkpoints/local/deepcfr.pt

# Evaluate vs baselines + LBR.
iip eval --checkpoint checkpoints/local/deepcfr.pt --hands 5000 --opponents fish,strength,random --with-lbr

# Play a few hands in the terminal.
iip play --checkpoint checkpoints/local/deepcfr.pt --hands 3
```

For real results expect `iters >= 50` and `traversals >= 10_000` — a short Kuhn convergence test lives in `tests/agents/test_deepcfr_kuhn.py` so you can verify the solver is wired correctly before burning HULHE compute. See [`docs/training.md`](docs/training.md) for the full iters-vs-traversals trade-off, progress / checkpoint behaviour, and how to tune via [`docs/evaluation.md`](docs/evaluation.md).

## Streamlit app

Local:
```bash
streamlit run app/streamlit_app.py
```

### Deploy to Streamlit Cloud

1. Push this repo to GitHub.
2. In Streamlit Cloud, **New app → Deploy from repo**, point at `app/streamlit_app.py`.
3. In **Secrets**, set:
   ```toml
   HF_REPO_ID = "your-username/iip-hulhe"
   HF_TOKEN   = "hf_..."
   SUPABASE_URL = "https://xxxx.supabase.co"
   SUPABASE_KEY = "anon-key"         # service-role key only for retrain; app uses anon
   ```
4. In Supabase, create the `hands` table using the schema in `src/iip/io/supabase_client.py`.
5. The app loads the latest checkpoint on boot. If HF isn't configured it falls back to `checkpoints/local/deepcfr.pt`.

## Human-in-the-loop retraining

The nightly GitHub Actions job (`.github/workflows/nightly-retrain.yml`) does:

1. Pulls the previous checkpoint from HF Hub.
2. Fetches new hands from Supabase via `HandStore`.
3. Runs `scripts/nightly_retrain.py` — PPO self-play with baselines + human trajectory logging.
4. Runs `scripts/evaluate_checkpoint.py` — head-to-head + LBR; fails the run if the new bot regresses vs previous by more than `--regression-margin-mbb`.
5. On pass: uploads the new checkpoint tagged `ckpt-YYYYMMDD-HHMMSS` to HF Hub and commits `reports/latest.json` to the repo. The app picks up the new version on its next cold boot.

Required secrets in GitHub: `HF_TOKEN`, `HF_REPO_ID`, `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`.

## Algorithmic notes

- **Deep CFR**: two MLPs per player — advantage (`V_p(I, a)`) regressed on sampled counterfactual regrets, strategy (`π̄(I)`) regressed on sampled strategies from a reservoir buffer with linear-CFR weighting. Regret matching at each decision node produces the behavioural strategy.
- **Feature encoding** (`src/iip/features/infoset.py`): one-hot 169-bucket preflop id + board rank multi-hot + suit summary + street + pot/stacks + contributions + raises + position + legal-action mask. 200-dim float vector.
- **PPO**: standard clip objective with GAE, masked softmax over legal actions, warm-started from the Deep CFR strategy net.
- **League** (`src/iip/train/league.py`): JSON-persisted ELO + ELO-softmax sampling of opponents, so tough checkpoints are picked more often.
- **Exploitability**: LBR (Lisý & Bowling, 2017) gives a cheap exploitability lower bound; adequate as a regression guard between checkpoints.

## Testing

```bash
pytest -q
```

Runs engine/ rules + showdown, eval/ ranker + MC equity property tests, agents/ baselines sanity, Deep CFR convergence on Kuhn (smoke), mbb/h + LBR smoke.

## References

- Brown, N., Lerer, A., Gross, S., Sandholm, T. (2019). *Deep Counterfactual Regret Minimization.* ICML.
- Heinrich, J., Silver, D. (2016). *Deep Reinforcement Learning from Self-Play in Imperfect-Information Games.* (context for NFSP alternative path.)
- Lisý, V., Bowling, M. (2017). *Equilibrium Approximation Quality of Current No-Limit Poker Bots.* (LBR.)
- Southey et al. (2005). *Bayes' Bluff: Opponent Modelling in Poker.* (project motivation.)

## License

MIT.
