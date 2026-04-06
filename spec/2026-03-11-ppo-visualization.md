# PPO Training Visualization Plan

**Date:** 2026-03-11
**Status:** Implemented

## Problem

Training runs produce no usable visualization data:
- Wandb is broken (`ImportError` on protobuf mismatch)
- Even when working, offline runs store binary `.wandb` files, not readable JSONL
- Prior 10M-step run has eval CSVs (0/4/2 solved at 1M/5M/10M) but zero training curves
- Existing plot scripts (`plot_ppo_results.py`, `plot_results.py`) only consume eval CSVs

## Solution

### 1. CSV Metrics Logger in `training.py`

Every training run now writes `metrics.csv` to the output directory, independent of wandb. One row per update with all diagnostic metrics.

**Columns:** update, global_step, learning_rate, policy_loss, value_loss, entropy_loss, approx_kl, clipfrac, explained_variance, logit_gap, top1_prob, noop_rate, episodes_this_update, episodes_solved, total_solved, total_unsolved, advantages_mean, advantages_std, action_0..action_11

**File size:** ~2MB for a full 100M-step run (62,500 rows). Safe to commit to git.

### 2. Plotting Script: `plot_training_curves.py`

Produces two figures from `metrics.csv`:

#### Figure 1: `training_diagnostics.png` (2x3 grid)

| Panel | Metrics | What to look for |
|-------|---------|------------------|
| Losses | policy_loss, value_loss (dual axis) | value_loss should decrease; policy_loss oscillates |
| Entropy & KL | entropy_loss, approx_kl | Entropy decays gradually, KL stays small |
| Clipping | clipfrac, explained_variance | clipfrac 0.1-0.3 healthy; explained_var rises toward 1.0 |
| Policy sharpness | logit_gap, top1_prob | Rising = learning; flat at 1/12 = uniform = no learning |
| Outcomes | episodes_solved/update, total_solved | Any upward trend = progress |
| No-op rate | noop_rate | >0.5 = agent stuck |

#### Figure 2: `action_distribution.png`

Stacked area chart of action fractions over training. Uniform (all ~8.3%) = random. Concentrated = learned or collapsed.

### 3. Existing Plot Scripts (no changes needed)

| Script | Input | When to use |
|--------|-------|-------------|
| `plot_ppo_results.py` | Eval CSVs | After `eval_checkpoints.sh` — scaling curve |
| `plot_results.py` | Eval CSVs | Per-instance heatmaps, bar charts |
| **`plot_training_curves.py`** | **`metrics.csv`** | **During/after training — training dynamics** |

## Terminal Commands

### Smoke test
```bash
python -m ac_solver.agents.ppo --preset paper_mac --total-timesteps 50000 \
    --exp-name smoke_test
head -3 out/smoke_test_*/metrics.csv
python -m ac_solver.agents.plot_training_curves \
    --csv out/smoke_test_*/metrics.csv --output-dir /tmp/smoke_plots/
open /tmp/smoke_plots/training_diagnostics.png
```

### Full 100M run
```bash
bash scripts/run_ppo_paper_faithful.sh 1
```

### Plot training curves (can run during training)
```bash
python -m ac_solver.agents.plot_training_curves \
    --csv out/paper_faithful_mac_s1_*/metrics.csv \
    --output-dir results/plots/ --smooth 50
```

### Plot eval results (after training)
```bash
python -m ac_solver.agents.plot_ppo_results \
    --ppo-csvs out/paper_faithful_mac_s1_*/eval_det_*.csv \
    --classical-csvs results/greedy_1M.csv results/bfs_1M.csv \
    --output-dir results/plots/
```

### Compare det vs stochastic
```bash
for csv in out/paper_faithful_mac_s1_*/eval_*.csv; do
    solved=$(tail -n +2 "$csv" | grep -c "True" || echo 0)
    total=$(tail -n +2 "$csv" | wc -l | tr -d ' ')
    echo "$(basename $csv): ${solved}/${total}"
done
```

## Best Practices

1. **Check training curves before eval** — flat logit_gap + high entropy = no learning = don't bother with full eval
2. **Smooth aggressively for long runs** — `--smooth 50` or `--smooth 100` for 100M steps
3. **Compare det vs sto eval** — sto >> det means policy learned but is diffuse
4. **Check action distribution** — 1-2 actions dominating early = collapse; uniform at 10M = no learning
5. **Save metrics.csv to git** — small file, enables re-plotting without re-training
