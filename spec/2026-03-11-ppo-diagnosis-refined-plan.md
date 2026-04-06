# Refined Plan: PPO Diagnosis & Paper-Faithful Reproduction

**Date:** 2026-03-11
**Status:** Ready for execution
**Prerequisite docs:** `spec/2026-03-10-ppo-diagnosis.md`, `spec/paper-faithful-benchmark.md`

## Context

A 10M-step PPO run (`--preset mac`, run `57e5c132`) reports 100% training solve rate but 0.2% greedy eval solve rate. The root cause is established in `spec/2026-03-10-ppo-diagnosis.md`: the policy never learned (entropy=2.45, clipfrac=0, approx_kl=0). Training "solved" presentations via stochastic random exploration, not a learned policy. The paper uses 100M steps with 28 envs; the mac preset uses 10M with 8.

**Goal:** Add targeted instrumentation to confirm the diagnosis, create a paper-faithful Mac preset at 100M steps, and verify the reproduction with both deterministic and stochastic evaluation.

---

## What Already Exists (Skip/Reuse)

| Item | Location | Action |
|------|----------|--------|
| Faithfulness audit | `spec/paper-faithful-benchmark.md` — complete mismatch table with file/line refs | **Skip** |
| PPO diagnosis | `spec/2026-03-10-ppo-diagnosis.md` — root cause analysis, metric tables | **Skip** |
| W&B logging (15 metrics) | `training.py:410-435` — policy_loss, value_loss, entropy, approx_kl, clipfrac, explained_var, advantages, LR, solved/unsolved, returns, lengths | **Reuse** |
| Checkpointing at thresholds | `training.py:449-461` + `args.py:145` — eval_at=[100K..10M] | **Extend** to 100M |
| Greedy (argmax) eval | `evaluate.py` — loads checkpoint, CSV output, standard schema | **Extend** with stochastic mode |
| PAPER_PRESET | `args.py:11-34` — [512,512], LR=1e-4, gamma=0.999, 100M steps, 28 envs | **Inherit** |
| MAC_PRESET | `args.py:36-41` — inherits paper, overrides num_envs=8, timesteps=10M | **Reference** |
| Run script | `scripts/run_ppo_reproduction.sh` — 3-seed mac preset pipeline | **Reference** |
| Classical baselines | `results/greedy_1M.csv` (554/1190), `results/bfs_1M.csv` (278/1190) | **Reuse** |
| Mac 10M run data | `out/57e5c132*/`, `ppo_out.log`, eval CSVs | **Reference only** |

---

## Execution Plan

### Phase 1: Targeted Instrumentation (Day 1)

#### Step 1.1 — Add missing training metrics

**File:** `ac_solver/agents/training.py`

**During rollout phase (lines 188-276),** collect:
- **Action counts:** Initialize `action_counts = torch.zeros(12, device=device)`. After line 202 (`actions[step] = action`), add `action_counts.scatter_add_(0, action, torch.ones_like(action, dtype=torch.float32))`.
- **No-op count:** After `envs.step()` at line 206, compare `next_obs` (numpy) to `obs[step]` (tensor on device) for non-terminal envs. Count instances where observation is unchanged. Accumulate into `noop_count`.
- **Episode outcome counters:** Inside the `_record_info` block (lines 221-272), count `episodes_this_update += 1` and `episodes_solved_this_update += 1` when `done[i]` is True.

**After rollout, before PPO update,** compute logit statistics efficiently:
```python
# Batch compute logits for all collected obs (no grad, single forward pass)
with torch.no_grad():
    all_logits = agent.actor(obs.reshape(-1, obs.shape[-1]))  # (num_steps*num_envs, 12)
    sorted_logits, _ = all_logits.sort(dim=-1, descending=True)
    logit_gap = (sorted_logits[:, 0] - sorted_logits[:, 1]).mean().item()
    top1_prob = all_logits.softmax(dim=-1).max(dim=-1).values.mean().item()
```

**Note:** This does NOT modify `get_action_and_value()` — avoids touching the agent API. The batch forward pass after rollout is cheaper than per-step computation and uses the already-collected `obs` tensor.

**In W&B logging block (after line 433),** add:
```python
"actions/frequency": wandb.Histogram(action_counts.cpu().numpy()),
"debug/noop_rate": noop_count / (args.num_steps * args.num_envs),
"debug/logit_gap_mean": logit_gap,
"debug/top1_prob_mean": top1_prob,
"charts/episodes_this_update": episodes_this_update,
"charts/episodes_solved_this_update": episodes_solved_this_update,
```

**Also add non-W&B print logging** (after the W&B block) so metrics appear in terminal/log even without W&B:
```python
if update % 10 == 0:
    print(f"[{global_step}] entropy={entropy_loss.item():.3f} clipfrac={np.mean(clipfracs):.4f} "
          f"logit_gap={logit_gap:.4f} noop={noop_count/(args.num_steps*args.num_envs):.3f} "
          f"solved={len(success_record['solved'])}")
```

**Justification:** These 6 metrics fill the diagnostic gap. Existing logging shows PPO is frozen but not *why*. Action frequency reveals uniform vs collapsed policy; logit gap quantifies policy sharpness; no-op rate reveals wasted computation from max_relator_length overflow; top-1 probability directly measures policy confidence.

#### Step 1.2 — Add stochastic evaluation mode

**File:** `ac_solver/agents/evaluate.py`

Add CLI arguments (after line 130):
```python
parser.add_argument("--stochastic", action="store_true",
    help="Use stochastic sampling instead of greedy argmax")
parser.add_argument("--num-rollouts", type=int, default=1,
    help="Number of stochastic rollouts per instance (report best)")
parser.add_argument("--eval-seed", type=int, default=42,
    help="RNG seed for stochastic evaluation")
```

Modify `evaluate_instance()` (lines 51-104):
- Add `stochastic: bool = False` and `eval_seed: int = 42` parameters
- When `stochastic=True`: replace `logits.argmax(dim=-1).item()` (line 75) with:
  ```python
  from torch.distributions import Categorical
  action = Categorical(logits=logits).sample().item()
  ```
- Set `algorithm = "ppo_stochastic"` in the return dict when stochastic
- For `--num-rollouts > 1`: run N rollouts per instance, keep the one that solved (or the shortest path if multiple solved, or the one with smallest final length if none solved)
- Seed `torch.manual_seed(eval_seed + instance_id)` before each instance for reproducibility

**Justification:** This is the KEY diagnostic test. If stochastic eval >> deterministic eval, the policy is diffuse (entropy too high for argmax to be useful). If both are near-zero, the policy genuinely hasn't learned.

#### Step 1.3 — Create eval checkpoint convenience script

**New file:** `scripts/eval_checkpoints.sh`

```bash
#!/bin/bash
# Usage: bash scripts/eval_checkpoints.sh <checkpoint_dir> [horizon]
set -e
CKPT_DIR="${1:?Usage: eval_checkpoints.sh <checkpoint_dir> [horizon]}"
HORIZON="${2:-200}"

for CKPT in "${CKPT_DIR}"/ckpt_*.pt; do
    [ -f "$CKPT" ] || continue
    STEPS=$(basename "$CKPT" .pt | sed 's/ckpt_//')
    echo "=== Evaluating checkpoint at ${STEPS} steps ==="

    # Deterministic (argmax) eval
    python -m ac_solver.agents.evaluate \
        --checkpoint "$CKPT" --horizon "$HORIZON" --full \
        --output "${CKPT_DIR}/eval_det_${STEPS}.csv"

    # Stochastic eval (single rollout)
    python -m ac_solver.agents.evaluate \
        --checkpoint "$CKPT" --horizon "$HORIZON" --full \
        --stochastic --output "${CKPT_DIR}/eval_sto_${STEPS}.csv"

    # Stochastic eval (best-of-5 rollouts)
    python -m ac_solver.agents.evaluate \
        --checkpoint "$CKPT" --horizon "$HORIZON" --full \
        --stochastic --num-rollouts 5 \
        --output "${CKPT_DIR}/eval_sto5_${STEPS}.csv"
done

echo "=== Summary ==="
for CSV in "${CKPT_DIR}"/eval_*.csv; do
    SOLVED=$(tail -n +2 "$CSV" | grep -c "True" || echo 0)
    TOTAL=$(tail -n +2 "$CSV" | wc -l | tr -d ' ')
    echo "$(basename $CSV): ${SOLVED}/${TOTAL} solved"
done
```

---

### Phase 2: Paper-Faithful Mac Preset (Day 1)

#### Step 2.1 — Add PAPER_FAITHFUL_MAC preset

**File:** `ac_solver/agents/args.py`

After `MAC_PRESET` (line 41), add:
```python
PAPER_FAITHFUL_MAC = {
    **PAPER_PRESET,
    "num_envs": 8,                  # Mac hardware: 28 → 8 parallel actors
    "total_timesteps": 100_000_000, # Full paper budget (not truncated)
    "device": "auto",               # MPS on Mac, CPU fallback
}
```

Update `PRESETS` dict (line 43):
```python
PRESETS = {
    "paper": PAPER_PRESET,
    "mac": MAC_PRESET,
    "paper_mac": PAPER_FAITHFUL_MAC,
}
```

Update argparser `choices` (lines 52, 62):
```python
choices=["paper", "mac", "paper_mac"]
```

**Key difference from MAC_PRESET:** `total_timesteps=100M` instead of 10M. All hyperparameters remain paper-faithful: [512,512] MLP, LR=1e-4 with linear decay, gamma=0.999, update_epochs=1, clip_coef=0.2, ent_coef=0.01, clip_rewards [-10, 1000].

**Batch size computation:** `batch_size = 8 * 200 = 1600`, `minibatch_size = 1600 / 4 = 400`. Paper uses `batch_size = 28 * 200 = 5600`, `minibatch_size = 1400`. The smaller batch/minibatch is the only algorithmic difference from the paper — this means more PPO updates per step (62,500 vs ~17,857) but with noisier gradients.

#### Step 2.2 — Extend eval_at for paper_mac

The default `eval_at` in `args.py:145` is `[100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]`. For 100M-step runs, need additional thresholds.

**Option A (preferred):** Add to `PAPER_FAITHFUL_MAC` preset:
```python
"eval_at": [1_000_000, 5_000_000, 10_000_000, 20_000_000, 50_000_000, 100_000_000],
```

**Note:** This requires `eval_at` to be handled by `parser.set_defaults(**preset_defaults)`. Currently `eval_at` is a `nargs="+"` argument with default `[100K, 500K, 1M, 5M, 10M]`. Preset defaults will override the default, and CLI `--eval-at` will override the preset. This works correctly with argparse.

#### Step 2.3 — Create paper-faithful run script

**New file:** `scripts/run_ppo_paper_faithful.sh`

```bash
#!/bin/bash
# Paper-faithful PPO reproduction on Mac hardware
# Usage: bash scripts/run_ppo_paper_faithful.sh [seed]
set -e
SEED="${1:-1}"
RUN_NAME="paper_faithful_mac_s${SEED}"

echo "=== Training: --preset paper_mac, seed=${SEED} ==="
echo "Expected runtime: ~5-15 hours depending on MPS/CPU"

python -m ac_solver.agents.ppo \
    --preset paper_mac \
    --seed "$SEED" \
    --wandb-log --wandb-mode offline \
    --exp-name "$RUN_NAME"

echo "=== Evaluating all checkpoints ==="
bash scripts/eval_checkpoints.sh "out/${RUN_NAME}"
```

#### Step 2.4 — Add preset validation test

**File:** `tests/agents/test_ppo_presets.py`

Add test verifying:
- `PAPER_FAITHFUL_MAC["total_timesteps"] == 100_000_000`
- `PAPER_FAITHFUL_MAC["num_envs"] == 8`
- All PAPER_PRESET keys present in PAPER_FAITHFUL_MAC
- For each key in PAPER_PRESET that isn't overridden: `PAPER_FAITHFUL_MAC[key] == PAPER_PRESET[key]`

---

### Phase 3: Run & Analysis (Day 2-5)

#### Step 3.1 — Smoke test (30 min)

```bash
# Verify instrumentation
python -m ac_solver.agents.ppo --preset paper_mac --total-timesteps 50000 \
    --wandb-log --wandb-mode offline --exp-name smoke_test

# Verify stochastic eval
python -m ac_solver.agents.evaluate --checkpoint out/smoke_test/ckpt.pt \
    --horizon 200 --stochastic --output /tmp/smoke_sto.csv

# Verify tests pass
poetry run pytest
```

Check:
- [ ] New W&B metrics (action frequency, noop_rate, logit_gap, top1_prob) appear in offline run
- [ ] Terminal prints show new metrics every 10 updates
- [ ] Stochastic eval CSV has `algorithm=ppo_stochastic`
- [ ] All existing tests + new preset test pass

#### Step 3.2 — Full paper-faithful run

```bash
bash scripts/run_ppo_paper_faithful.sh 1
```

**Runtime estimate:**
- MPS: ~5.5 hours (100M steps / ~5k steps/sec)
- CPU: ~14 hours (100M steps / ~2k steps/sec)

**Monitor at intermediate checkpoints** (check terminal output or W&B offline):
- Does entropy decrease from 2.45? (Expect: gradual decrease if learning)
- Does clipfrac become non-zero? (Expect: >0.05 if policy is updating)
- Does explained_variance improve from -1.75? (Expect: approach 0 then positive)
- Does logit_gap increase? (Expect: grows if policy becomes decisive)

#### Step 3.3 — Evaluate all checkpoints

After training completes, `eval_checkpoints.sh` runs automatically. Expected results:

| Checkpoint | Expected det. solve rate | Reasoning |
|---|---|---|
| 1M | ~0/1190 | Too early |
| 5M | ~0-5/1190 | Random baseline |
| 10M | ~2-10/1190 | Same as current mac run |
| 20M | >10/1190 if learning | First signal of improvement |
| 50M | Increasing | Learning curve inflection |
| 100M | ~431/1190 if paper reproduces | Paper's reported result |

**If results at 100M are still near-zero:** the batch size difference (1600 vs 5600) may matter more than expected, or there's an implementation issue beyond undertraining.

#### Step 3.4 — Comparison plot

**File:** `ac_solver/agents/plot_ppo_results.py`

Add a third plot function: `plot_det_vs_sto(det_csvs, sto_csvs, output_path)`:
- Grouped bar chart: x-axis = checkpoint step, y-axis = solve count
- Three bars per checkpoint: deterministic, stochastic (1 rollout), stochastic (best-of-5)
- Include horizontal lines for classical baselines (Greedy=554, BFS=278)

#### Step 3.5 — Update diagnosis document

**File:** `spec/2026-03-10-ppo-diagnosis.md`

Append "## 100M-Step Results" section with:
- Eval results table (det + sto) at each checkpoint
- New instrumentation metrics (logit gap curve, action distribution, no-op rate)
- Comparison with paper's 431/1190
- Final verdict: undertraining confirmed or other issues found

---

## Files Modified/Created Summary

| File | Change | Type |
|------|--------|------|
| `ac_solver/agents/training.py` | Add 6 new metrics: action frequency, noop_rate, logit_gap, top1_prob, episodes_this_update, episodes_solved_this_update; add periodic print logging | **Modify** |
| `ac_solver/agents/evaluate.py` | Add --stochastic, --num-rollouts, --eval-seed flags | **Modify** |
| `ac_solver/agents/args.py` | Add PAPER_FAITHFUL_MAC preset, update PRESETS dict and choices | **Modify** |
| `ac_solver/agents/plot_ppo_results.py` | Add det vs sto comparison plot | **Modify** |
| `tests/agents/test_ppo_presets.py` | Add test for paper_mac preset | **Modify** |
| `scripts/eval_checkpoints.sh` | Convenience script for batch eval | **New** |
| `scripts/run_ppo_paper_faithful.sh` | Run script for paper-faithful experiment | **New** |

**Not modified:** `ppo_agent.py` (logit gap computed via batch `agent.actor()` call after rollout, not by changing the agent API), `ac_env.py`, `ac_moves.py`, `environment.py`.

---

## Acceptance Criteria

1. **Stochastic vs deterministic gap:** If stochastic eval >> deterministic at any checkpoint → policy is diffuse, not a greedy solver. Quantify the ratio.
2. **PPO health recovery:** If clipfrac and approx_kl remain zero AND entropy stays near ln(12)=2.48 at 100M steps → PPO is fundamentally not updating (likely batch size issue, not just undertraining).
3. **100M-step improvement:** If 100M-step deterministic eval solves >100/1190 → undertraining hypothesis confirmed.
4. **Paper reproduction threshold:** If 100M-step deterministic eval achieves >350/1190 (within 20% of paper's 431/1190) → declare successful reproduction on Mac hardware.
5. **If reproduction fails at 100M:** the 8-env batch size (1600 vs paper's 5600) is the likely remaining culprit. Document and recommend next steps (e.g., increase num_envs, or accept the batch size difference).

## Deliberate Omissions

| Omitted | Reason |
|---------|--------|
| `audit/` directory | `spec/` already serves this role with existing docs |
| `mismatch_table.csv` | Already in `spec/paper-faithful-benchmark.md` |
| Re-run Mac 10M | Already complete (run 57e5c132), diagnosis conclusive |
| Variable-horizon experiment | Separate research question |
| Reward histogram plots | Low marginal value — diagnosis is clear |
| Per-instance heatmap | Nice-to-have, not needed for core question |
| Modify ppo_agent.py | Logit gap computed post-rollout via batch forward pass |
| 3 seeds initially | Validate 1 seed first, then extend |
