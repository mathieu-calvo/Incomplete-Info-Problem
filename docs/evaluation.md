# Evaluation guide

This document explains what `iip eval` measures, how to read its output, and
what to change when the numbers look bad. Pair it with
[`architecture.md`](architecture.md) for the "why" behind the training stack.

---

## 1. The command

```bash
iip eval \
  --checkpoint checkpoints/local/deepcfr.pt \
  --hands 5000 \
  --opponents fish,strength,random \
  --with-lbr
```

What it does:

1. Loads the Deep CFR strategy net from the given checkpoint.
2. For each opponent in `--opponents`, plays `--hands` hands head-to-head,
   **alternating seats every hand** so positional advantage cancels.
3. Reports the hero's winrate in **mbb/h** (milli-big-blinds per hand) with a
   95% confidence interval, plus win/loss/tie counts.
4. With `--with-lbr`, additionally runs Local Best Response on
   `min(hands, 1000)` hands as a cheap exploitability lower bound.

Source:

- `src/iip/cli.py` — the `eval` command.
- `src/iip/metrics/mbb.py` — `head_to_head_mbb`.
- `src/iip/metrics/exploitability.py` — `local_best_response`.
- `src/iip/agents/fixed_policy.py` — `FishAgent`, `StrengthHandAgent`,
  `StartingHandAgent`.
- `src/iip/agents/random_agent.py` — `RandomAgent`.

---

## 2. The metric: mbb/h

`head_to_head_mbb` plays `n_hands` hands and computes, per hand:

```
per_hand_mbb = payoff * 1000 / big_blind
```

where `payoff` is the hero's net chip change for the hand. It reports:

- `mbbph = mean(per_hand_mbb)` — the hero's average winrate in
  milli-big-blinds per hand.
- `ci95 = 1.96 * std / sqrt(n)` — the 95% CI half-width.
- `W/L/T` — hand-outcome counts.

Positive means the hero is winning. For context, serious HU Limit bots tend
to win **40–80 mbb/h** against each other; margins of hundreds are cartoonish
and almost always indicate the loser is badly under-trained or the baseline
is weak.

Seats alternate (`hero_seat = h % 2`), so positional advantage cancels to
zero in expectation over an even number of hands.

---

## 3. The baselines

| Agent | Policy | What it tests |
|---|---|---|
| `RandomAgent` | Uniform over legal actions | Sanity floor — anything with a pulse should beat random. |
| `FishAgent` | Always check-calls, never raises, never folds | Realized-equity test: you can only win by showing down better hands. |
| `StartingHandAgent` | Preflop equity only → threshold policy | Preflop hand selection. |
| `StrengthHandAgent` | MC equity every street → threshold policy (raise ≥60%, call ≥50%, else fold) | Holistic equity test: loses fold equity to *any* agent that raises correctly. |

Threshold policy details (`fixed_policy._threshold_policy`):

- `p >= 0.6` and raise available → **BET_RAISE**
- `p >= 0.5` → **CHECK_CALL**
- `p < 0.5` → check if free, else **FOLD**

---

## 4. LBR: exploitability lower bound

`local_best_response` runs an **LBR exploiter** against the bot:

- At each hero decision it considers FOLD, CHECK_CALL, BET_RAISE.
- For each, it does `rollouts_per_decision=4` 1-ply rollouts where the
  exploiter just check-calls and the bot plays its policy to terminal.
- It picks the action with highest mean rollout value.

The returned number is the **exploiter's mbb/h against the bot**. Higher
means the bot has more leaks an adversary can capitalize on. LBR is a *lower
bound* on true exploitability — the bot is at least this exploitable, and
usually more.

Rough interpretation:

| LBR (mbb/h) | Quality |
|---|---|
| < 50 | Near-equilibrium (strong) |
| 50–200 | Decent, still clearly exploitable |
| 200–500 | Substantial leaks |
| 500+ | Major structural issues |

---

## 5. Worked example: a 3-iter smoke run

The checkpoint was produced by

```bash
iip train --game hulhe --iters 3 --traversals 1000 --output checkpoints/local/deepcfr.pt
```

Eval output:

```
DeepCFR vs Fish:        -407.80 ± 59.26  mbb/h over 5000 hands (1181W/3744L/75T)
DeepCFR vs StrengthHand: -1097.80 ± 114.21 mbb/h over 5000 hands (1139W/3803L/58T)
DeepCFR vs Random:       +399.70 ± 62.88  mbb/h over 5000 hands (2669W/2327L/4T)
LBR exploitability:       1057.00 mbb/h (higher = more exploitable)
```

Reading each line:

### vs Fish: −407.80 mbb/h

Losing ~0.41 BB/hand to an opponent that never folds and never raises.
Since Fish never folds, all losses come from showdown. A negative result
means the policy is committing chips with hands whose realized equity at
showdown is below 50%. Classic "overvalues weak holdings" signature of an
undertrained strategy net.

### vs StrengthHand: −1,097.80 mbb/h

Losing ~1.1 BB/hand. StrengthHand folds below 50% equity and raises above
60%, so it never pays off a value bet with a weak hand. The bot has no fold
equity of its own (doesn't bluff correctly) and pays off StrengthHand's
raises. This is typically the **most diagnostic** baseline — a small
edge-case mistake becomes a huge leak because StrengthHand's threshold
policy is unforgiving.

### vs Random: +399.70 mbb/h

The policy beats uniform noise by ~0.4 BB/hand. Non-trivially better than
random, but any rule that just folds junk preflop should be well into
4-digit mbb/h vs Random — so this is still weak. The "beats Random but
loses to Fish" profile is the hallmark of an under-trained checkpoint.

### LBR: 1,057 mbb/h

An LBR exploiter can print **1.06 BB/hand** out of the checkpoint. For a
trained bot this should be in the tens to low hundreds. 1,000+ means the
policy has large, easily found holes (e.g. folds too often to a single bet,
or under-raises a class of hands an exploiter can bluff-catch).

### Summary table

| Opponent | Result | Reading |
|---|---|---|
| Random | +400 mbb/h | Above chance |
| Fish | −408 | Overcommits with weak hands |
| StrengthHand | −1,098 | No fold equity, bleeds against any filter |
| LBR | 1,057 | Adversary can take 1 BB/hand |

This is the expected profile of a **smoke run** — the pipeline is
functional, the network just hasn't trained long enough to produce real
strategy.

---

## 6. How to improve the numbers

Listed roughly in "biggest effect first" order. Most of these are tuning
knobs on `iip train` / `DeepCFRConfig`; the rest are training-data or
eval-setup changes.

### 6.1. Train longer — **the main lever**

Deep CFR learns two networks from regret samples. Both need a lot of data.
`--traversals` is **per iteration per player**; one iter runs
`2 × traversals` calls because the trainer loops over both traversers.

| Stage | Command | Total traversals | Approx. CPU time |
|---|---|---|---|
| Smoke | `--iters 3 --traversals 1000` | 6k | ~2–5 min |
| Dry run | `--iters 10 --traversals 20_000` | 400k | ~30–60 min |
| First-pass serious | `--iters 20 --traversals 50_000` | 2M | ~3–8 hours |
| Research-grade | `--iters 40 --traversals 200_000+` | 16M+ | multi-day |

Expect Fish to flip positive and StrengthHand to close most of the gap
around the 2M-total mark. Getting LBR below a few hundred mbb/h generally
requires research-grade scale *and* PPO fine-tuning on top.

Budget: on CPU, each traversal is ~a few ms; 2M/iter ≈ hours per iter. Use
a GPU (`--device cuda`) for the net training phases or plan an overnight
run. The nightly retrain action in `.github/workflows/nightly-retrain.yml`
uses CPU-only torch, so it should be treated as *continual refinement*
rather than from-scratch training.

### 6.2. Increase network capacity

`DeepCFRConfig.hidden_sizes_advantage` and `hidden_sizes_strategy` default
to small MLPs. Once traversal count is healthy, bump hidden sizes (e.g.
`[256, 256, 128]` → `[512, 512, 256]`) so the net can represent HULHE's
info-set structure. Too-small networks saturate regret memory well before
the policy is tight.

### 6.3. Grow the buffers

`advantage_buffer_size` / `strategy_buffer_size` cap how many samples the
reservoir buffers hold. When you 10× the traversal count, buffer caps
become the bottleneck — Deep CFR relies on old samples weighted by their
iter to stabilize. Raise both to 1M+ if memory allows.

### 6.4. Better feature encoding

`GameAdapter.encode` produces the input vector every forward pass consumes.
The policy can only be as sharp as its features. Candidate additions if
not already present:

- Pot odds / equity bucket for the current decision.
- Raises-this-street and remaining raise slots (capped betting changes EV
  meaningfully in Limit).
- Position one-hot (button vs big blind) since HU positional EV is large.
- Street one-hot already helps the net not smear learning across streets.

Anything that makes the input more discriminative cuts the traversals
required to reach a given LBR.

### 6.5. Tune CFR itself

- **Sampling scheme:** external sampling (current) is the standard; if
  convergence plateaus, try outcome sampling for noisier but cheaper
  traversals.
- **Regret matching+ / linear CFR weighting:** weighting iterations
  linearly (as done via the `iter_t` factor in the loss) accelerates
  convergence. Make sure `iters` in the advantage/strategy loss is
  actually being used (it is, in `_train_advantage` /
  `_train_strategy`).
- **Finalize more often:** `finalize_strategy()` trains the strategy net
  once at the end. Calling it periodically and evaluating lets you catch
  divergence early.

### 6.6. PPO fine-tune on top of Deep CFR

The architecture doc describes a PPO self-play step after Deep CFR
(`src/iip/train/ppo_trainer.py`). CFR gives a balanced policy; PPO
self-play sharpens exploit-resistance against specific opponents. It is
especially good at pushing LBR down because PPO's reward signal
naturally fixes the tactical holes LBR exploits. Run it after Deep CFR
plateaus — never before.

### 6.7. Feed real human hands into retrain

The `save_completed_hand` path in `app/services/hand_store.py` captures
every completed hand from the Streamlit app into Supabase. The nightly
retrain workflow pulls them and continues training. Two wins here:

- Coverage of situations self-play under-samples (weird bet sequences from
  human play).
- Target distribution closer to "real opponents" than self-play alone.

Nothing to code — just play the bot more.

### 6.8. Make the evaluator itself sharper

Numbers improve faster when you can *see* them improve:

- **More hands.** 5,000 hands gives ±60–115 mbb/h CI; that's fine for a
  smoke read but noisy for gating. Nightly gating in
  `scripts/evaluate_checkpoint.py` already uses 20,000 — keep it there
  or raise.
- **More baselines.** Add `StartingHand` to the default set; it isolates
  preflop from postflop mistakes. Consider a `TightAggAgent` with a
  lower raise threshold as a second check.
- **LBR hands.** The default LBR runs on `min(hands, 1000)`. For
  gating, 2,000–5,000 tightens the exploitability estimate at the cost
  of a few more minutes.
- **Per-street breakdown.** Reporting mbb/h split by Street (preflop,
  flop, turn, river) tells you *where* the policy is leaking. Worth
  adding to `HeadToHeadResult` if gap analysis becomes blocking.

### 6.9. Sanity checks that catch "silent" regressions

Things to verify whenever the numbers look wrong in a confusing way:

- Both players use the **same** `HULHE` config (SB/BB/bet sizes). The
  `iip eval` command uses `HULHE()` defaults — if the checkpoint was
  trained on a different config the results are meaningless.
- The strategy net loads in **eval mode**
  (`DeepCFRAgent.load` already does this).
- `regret_matching` inside the policy returns a valid distribution; a
  degenerate uniform over all legal actions shows up as "~random" in
  eval.
- Seat alternation is actually happening (it is, in
  `head_to_head_mbb`), but custom runners sometimes forget.

---

## 7. Target numbers for the nightly retrain gate

The gate in `scripts/evaluate_checkpoint.py` compares a new checkpoint
against the previous one. Reasonable thresholds to codify once the policy
is past smoke-run stage:

- **vs Fish:** positive and within noise of prior (or better).
- **vs StrengthHand:** positive and within noise of prior.
- **vs Random:** large positive — regression here means something broke
  at the feature / loading layer.
- **LBR:** within 1.5× of prior (LBR is noisy; don't gate too tightly).

A new checkpoint "passes" when every matchup is ≥ prior − 1σ and at least
one matchup is strictly better. That prevents silent drift while tolerating
noise.
