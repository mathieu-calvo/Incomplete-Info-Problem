# Training guide

This document explains how `iip train` works in practice — what the knobs
mean, where time is spent, and how to pick sensible settings without
burning hours on CPU. Pair it with [`architecture.md`](architecture.md) §3
for the "why" behind Deep CFR, and [`evaluation.md`](evaluation.md) for the
measurement loop you use to tune these knobs.

---

## 1. The command

```bash
iip train \
  --game hulhe \
  --iters 20 \
  --traversals 5000 \
  --output checkpoints/local/deepcfr-200k.pt \
  --checkpoint-every 1
```

Source: `src/iip/cli.py` (the `train` command) and
`src/iip/train/deepcfr_trainer.py` (the trainer itself).

Flags:

| Flag | What | Default |
|---|---|---|
| `--game` | `hulhe` or `kuhn` (Kuhn is for the CI convergence gate) | `hulhe` |
| `--iters` | Number of Deep CFR outer iterations | 5 |
| `--traversals` | Traversals per iter **per player** | 2000 |
| `--output` | Where to save the strategy net | `checkpoints/local/deepcfr.pt` |
| `--checkpoint-every` | Save every N iters; 0 disables mid-run saves | 1 |
| `--seed` | RNG seed | 0 |
| `--device` | `cpu` or `cuda` | `cpu` |

---

## 2. What a traversal is

A traversal is one walk through the game tree starting from a fresh random
deal. This project uses **external-sampling MCCFR** (see `_traverse` in
`src/iip/train/deepcfr_trainer.py`):

- At each **traverser** decision node: branch over *every* legal action,
  recurse into each, compute counterfactual values, and store the regret
  vector in `adv_buffers[traverser]`.
- At each **opponent** decision node: sample one action from its current
  regret-matched policy, store that policy vector in `strategy_buffer`,
  and continue down a single path.
- At a terminal: return the traverser's payoff.

One traversal produces roughly *(depth of the walk)* advantage samples for
the traverser plus some strategy-buffer samples for the opponent's decisions
along the way. Traversals are expensive because the traverser side
fans out at every decision — HULHE has up to 3 legal actions per node
across four streets.

The name *external sampling* refers to the fact that only the **external**
randomness (the opponent and chance) is sampled; the traverser's own
actions are enumerated.

---

## 3. What an iteration is

One iter is a single outer CFR sweep, implemented in `iterate()`:

1. For each player `p ∈ {0, 1}`:
   - Run `traversals` walks with `p` as traverser → advantage buffer fills.
   - **Retrain `adv_net[p]` from scratch** on the advantage buffer
     (`advantage_train_steps = 500` at batch 256). Re-initializing each
     iter is the Deep CFR recipe from Brown et al. 2019.
2. Strategy buffer accumulates throughout. It's only trained into a usable
   `strategy_net` when `finalize_strategy()` runs — either at checkpoint
   time or at end-of-run.

So: **traversals = data per iter. Iterations = how many improvement rounds.**

---

## 4. The trade-off: iters vs traversals

|  | Too few traversals | Too few iters |
|---|---|---|
| What breaks | Advantage net fits noise; next iter's policy is biased; error compounds | Regret never converges; strategy is a regret-match on near-random play |
| Symptom | Advantage training loss is unstable iter-to-iter; eval scores wobble | Strategy improves for a few iters then plateaus well above Nash |
| Fix | Increase `--traversals` | Increase `--iters` |

CFR convergence is **iteration-driven** — regret shrinks roughly like
`1/√T` in the vanilla algorithm. But each iter is only useful if its
regret estimates are cleaner than the noise, and that's the floor
traversals set.

The advantage buffer needs enough samples to usefully train 500 steps ×
batch 256 = ~128k examples without degenerate reuse. One HULHE traversal
deposits ~5–20 advantage samples (one per traverser decision visited on the
walk), so ~2000 traversals/iter/player already saturates that target.
Beyond that, extra traversals have diminishing returns on the current
iter's advantage net, while extra iters keep compounding self-correction.

### Reference point

The Deep CFR paper (Brown et al. 2019) used **~10⁶ traversals/iter × ~400
iters** on HUNL, with big GPUs. That's the research-scale regime. This
project runs 3+ orders of magnitude smaller — enough for a prototype that
reliably beats the fish/starting/random baselines, but not near Nash.

### CPU budget heuristic

For a local CPU run, shift toward more iters than you'd naively pick:

- If you have time for ~10⁵ total traversals: `--iters 20 --traversals 2500`
  beats `--iters 4 --traversals 12500` at the same wall time.
- Scale `--traversals` up (say to ~5000) only once you see the advantage
  training loss jumping around between iters — that's the signal that
  per-iter regret estimates are noisy.

---

## 5. Progress visibility

The trainer prints three nested tqdm bars and a per-iter completion line:

```
Deep CFR iters:   5%|▌    | 1/20 [06:12<1:57:47, 372.0s/iter]
  iter 2 p0 traversals:  42%|████▏     | 2100/5000 [02:34<03:32, 13.63trav/s]
  train adv p0:  78%|███████▊  | 389/500 [00:02<00:00, loss=0.082]
  train strategy:  64%|██████▍   | 640/1000 [00:04<00:02, loss=0.391]
[iter 1] adv buffers: [48231, 48102], strat buffer: 4612
[iter 1] checkpoint -> checkpoints/local/deepcfr-200k.pt
```

Inner bars (`leave=False`) clear after each player/iter completes so the
screen doesn't pile up. Buffer sizes on the completion line are a cheap
sanity gauge: if `adv buffers` are suspiciously small (well under ~20× the
batch size after a few iters), something is wrong with the traversal loop.

The `Deep CFR iters` bar's `s/iter` estimate is your wall-time oracle —
glance at it early to decide whether to let a run finish or kill it.

---

## 6. Checkpointing and safe interrupts

Each iter (by default) writes a full, loadable strategy net to `--output`:

- The trainer calls `finalize_strategy()` — trains `strategy_net` for
  `strategy_train_steps=1000` gradient steps on the accumulated strategy
  buffer, then builds a `DeepCFRAgent` around the result.
- Save is **atomic**: write to `<output>.tmp`, then `os.replace` over
  `<output>`. Ctrl-C during a save leaves the previous checkpoint intact
  rather than a truncated file.

The checkpoint gets **progressively better** across iters:

- The strategy buffer grows each iter (more samples of better policies).
- `_train_strategy()` fine-tunes `self.strategy_net` in place across
  calls — it doesn't reinitialize.
- So a checkpoint at iter 12 is trained on 12 iters of data, not just the
  12th iter's slice.

### Cost

- Disk: trivial (~300 KB for a 2×256 MLP).
- `_train_strategy()`: 1000 gradient steps, batch 256, on CPU = ~5–30 s.
- As a fraction of a ~6 min/iter HULHE run: ~2–8% overhead.

For a fast Kuhn run where iters are seconds each, that overhead is larger
as a percentage — use `--checkpoint-every 5` or similar if it bothers you.

### How to interrupt safely

- **Between iters:** Ctrl-C, done. The file on disk reflects the last
  completed iter.
- **During an iter:** Ctrl-C loses that iter's in-flight traversals. The
  file on disk is still the last-completed-iter checkpoint — unchanged.
- **During a checkpoint save:** Ctrl-C leaves `<output>` at the previous
  checkpoint. The temp file `<output>.tmp` may be left behind and can be
  deleted.

Resuming *training* (rather than inference from a checkpoint) is **not
supported** — the trainer's state (advantage buffers, advantage nets,
RNGs) is not serialized. Only the strategy net is saved. To "resume" you
restart from iter 1, but warm-starting from a prior checkpoint as PPO
initialization is what the nightly retrain script does (see
`architecture.md` §3 and §13).

---

## 7. Finding the sweet spot empirically

The per-iter checkpoint makes this straightforward:

1. Run training with `--checkpoint-every 1` (or 5 for short runs).
2. Every few iters, copy the current `--output` file aside with a distinct
   name (e.g. `deepcfr-iter5.pt`, `deepcfr-iter10.pt`, ...).
3. Run `iip eval --checkpoint deepcfr-iterN.pt --with-lbr` on each snapshot.
4. Plot **mbb/h vs each baseline** and **LBR exploitability** against iter
   count.

You're looking for:

- **mbb/h vs fish/starting/strength saturating** → more iters aren't
  helping beat these particular opponents. Diminishing returns, not
  necessarily convergence.
- **LBR flattening** → the strategy has stopped getting less exploitable
  for your current `--traversals` setting. This is the real signal.
- **LBR flat but noisy across checkpoints** → per-iter regret estimates
  are under-sampled. Bump `--traversals` and rerun.
- **LBR still improving linearly** → you stopped too early. Add iters.

Typical personal-project target: LBR exploitability in the **low hundreds
of mbb/h** (open-play HULHE humans are far higher; tabular near-Nash
solutions get into the tens). LBR is a *lower* bound on true
exploitability, so low-hundreds already represents a fairly solid
strategy.

See [`evaluation.md`](evaluation.md) for how to read mbb/h and LBR
numbers in detail.

---

## 8. Quick sanity runs

Before committing to an overnight HULHE run, verify the pipeline on Kuhn
— it takes seconds and has a known-Nash convergence gate:

```bash
iip train --game kuhn --iters 10 --traversals 500 \
  --output checkpoints/local/_kuhn-smoke.pt
```

`tests/agents/test_deepcfr_kuhn.py` is the canonical version of this — it
asserts exploitability under an ε threshold on every CI run. Any
regression in the Deep CFR implementation surfaces there first.

---

## 9. Common failure modes

| Symptom | Likely cause | What to try |
|---|---|---|
| `adv buffers` stays tiny after multiple iters | Traversals reaching terminal without traverser decisions (rare but possible at depth-0 sampling bugs) | Check `_traverse` hasn't been modified to skip storing on traverser nodes |
| Advantage training loss NaN / huge | Batch contains invalid advantage vectors | Check `adv = (child_values - v) * mask` masking; illegal actions should be 0 |
| Eval winrate worse than `RandomAgent` after training | Strategy net under-trained (buffer too small, too few strategy steps) | Run more iters so strategy buffer accumulates; `strategy_train_steps` is already 1000 per checkpoint |
| Slow traversals (<10/sec on HULHE CPU) | Python allocations in deepcopy of state per action | Expected — HULHE state has hole cards, board, history. Numbers in the ~10–30 trav/s range on CPU are typical |
| Windows console shows `UnicodeEncodeError` | cp1252 default stdout on Windows | Set `PYTHONIOENCODING=utf-8` or avoid non-ASCII in log strings (already cleaned up in `cli.py`; see `architecture.md` §15) |

---

## 10. What's not covered here

- **PPO fine-tuning on top of Deep CFR** — see `src/iip/train/ppo_trainer.py`
  and `architecture.md` §3. This doc is Deep CFR-only.
- **The nightly retrain job** — documented in `architecture.md` §13.
- **Hyperparameter config via Hydra** — `configs/**/*.yaml` exists but
  the current CLI takes flat flags. Hydra-driven training is listed as
  future work.
