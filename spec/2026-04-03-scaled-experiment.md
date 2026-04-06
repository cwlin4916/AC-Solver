# Scaled PPO Diagnostic Experiment

**Date:** 2026-04-03
**Goal:** Isolate why PPO achieved 32/1190 (2.7%) vs the paper's 431/1190 (36.2%), then test targeted fixes at 1/10 scale.

---

## 1. Systematic Diagnosis

Five root causes, ranked by estimated impact:

| # | Cause | Impact | Evidence |
|---|-------|--------|----------|
| 1 | **Reward ÷14400** (`training.py:332`) | Success signal +1000 → +0.069 (14,400× attenuation) | Paper §4.3 defines R ∈ [-10, +1000] with no division |
| 2 | **Batch 3.5× smaller** (1600 vs 5600) | Noisier gradients per update | 8 envs (Mac) vs 28 (paper) |
| 3 | **No variable horizon** (constant T=200) | Cannot solve long-path instances | Paper: T = 200→400→800→1200 at 10M/25M/50M |
| 4 | **LR→0 degradation** | 50M checkpoint outperforms 100M | Linear decay to exactly 0 wastes second half of training |
| 5 | **update_epochs=1** | Single pass over each batch | Standard PPO uses 3–10 epochs |

### 1.1 The Reward Attenuation Problem (Primary)

The paper defines (Section 4.3):
```
R(s_t, a_t, s_{t+1}) = {
  -min(10, length(s_{t+1}))   if length > 2
  +1000                        if solved
}
```
No further normalization. PPO sees rewards in [-10, +1000].

Our code applies an additional division **not present in the paper**:
```python
# training.py:332
rewards /= base_env.max_reward  # max_reward = 14,400
```

Result: PPO sees rewards in [-0.0007, +0.069]. The success signal is barely above noise level, making it extremely difficult for the critic to learn meaningful value estimates.

### 1.2 Training Evidence

From the 100M-step run (`paper_faithful_mac`):

| Metric | Start | End (100M) | Interpretation |
|--------|-------|------------|----------------|
| Entropy | 2.485 | 1.253 | Policy concentrated (learning occurred) |
| Top-1 prob | 8.4% | 60.1% | Strong action preferences |
| Explained var | -0.92 | +0.66 | Critic improved substantially |
| clipfrac | 0.0 | 0.0 | Expected with epochs=1 and tiny rewards |

The policy **did learn**, but only shallow structure. It solves only the easiest instances (low n, short paths). The 50M checkpoint (32/1190 best-of-5) outperforms 100M (29/1190) because the LR decayed to zero.

---

## 2. Scaled-Down Subset Design

### 2.1 Full Dataset Distribution

Each (n, |w|) cell has identical counts across all n values:

| |w| | Count per n | Cells |
|-----|-------------|-------|
| 1 | 2 | 7 |
| 2 | 2 | 7 |
| 3 | 2 | 7 |
| 4 | 6 | 7 |
| 5 | 18 | 7 |
| 6 | 42 | 7 |
| 7 | 98 | 7 |

Total: 7 × 170 = **1190 presentations**.

### 2.2 Stratified Subset (every_k=10)

Using `load_stratified_subset(instances, every_k=10)`:

| Cell size | Sampled | Formula |
|-----------|---------|---------|
| 2 (|w|≤3) | 1 | `[::10]` on 2 items → index 0 |
| 6 (|w|=4) | 1 | `[::10]` on 6 items → index 0 |
| 18 (|w|=5) | 2 | indices 0, 10 |
| 42 (|w|=6) | 5 | indices 0, 10, 20, 30, 40 |
| 98 (|w|=7) | 10 | indices 0, 10, ..., 90 |

Per n-row: 1+1+1+1+2+5+10 = **21 instances**.
Total: 7 × 21 = **147 instances** (12.4% of 1190).

### 2.3 Justification

- **Full difficulty coverage:** Every (n, |w|) cell retains ≥1 instance
- **Easy baseline preserved:** All w≤3 cells (100% greedy solve rate) have 1 instance each
- **Hard frontier sampled:** w=6,7 at high n retain 5–10 instances for statistical power
- **Reuses existing code:** `load_stratified_subset(instances, every_k=10)` — no custom selection logic
- **Expected greedy baseline:** ~68/147 (46.6%), matching the full-dataset rate

### 2.4 Greedy Solve Rates on Full Dataset (reference)

```
         w=1   w=2   w=3   w=4   w=5   w=6   w=7
n=1:   2/2   2/2   2/2   6/6  18/18  42/42  98/98   (100%)
n=2:   2/2   2/2   2/2   6/6  18/18  38/42  70/98   (81.2%)
n=3:   2/2   2/2   2/2   4/6  12/18  28/42  50/98   (58.8%)
n=4:   2/2   2/2   2/2   4/6  10/18  16/42  24/98   (35.3%)
n=5:   2/2   2/2   2/2   4/6   6/18   8/42  18/98   (24.7%)
n=6:   2/2   2/2   2/2   2/6   2/18   6/42  10/98   (15.3%)
n=7:   2/2   2/2   2/2   2/6   2/18   4/42   4/98   (10.6%)
```

---

## 3. Experiment Design: All Three Algorithms

This section specifies exactly what experiments we run for each algorithm, the hyperparameters used, and the justification for each choice.

### 3.1 Greedy Search Experiments

**What it does:** Min-heap priority search. Expands the state with the shortest total relator length first. No training — pure deterministic search.

**Hyperparameters:**

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `max_nodes_to_explore` | 1,000,000 | Matches paper (§4.2). The paper evaluates all classical methods at 10⁶ nodes. Increasing beyond this gives diminishing returns: greedy solves 554/1190 at 10⁶; the marginal gain per extra node drops sharply because the remaining unsolved instances require path-length increases that greedy's heuristic resists |
| `cyclically_reduce_after_moves` | True | Matches PPO's `ACEnv`, which applies cyclic reduction by default. Ensures fair comparison: both classical and PPO see the same state after each move. Cyclic reduction removes inverse pairs at relator boundaries (e.g., `x·x⁻¹` → ε), shrinking the state space |
| `use_fixed_lengths` | N/A | Greedy doesn't use this (BFS-only parameter) |

**What we run:**

| Experiment | Dataset | Purpose | CLI |
|------------|---------|---------|-----|
| GS-full (already done) | 1190 instances | Full baseline, already in `results/greedy_1M.csv` | — |
| GS-subset | 147 instances (every_k=10) | Establish ceiling for PPO ablation comparison | `--every-k 10` |

**Why not vary `max_nodes`?** The paper fixes this at 10⁶. Varying it (e.g., 10⁴, 10⁵, 10⁶) would be informative for understanding search scaling but is **not the goal of this experiment**, which is to diagnose PPO. We use the paper's value for apples-to-apples comparison.

**Why not vary `cyclic_reduce`?** The PPO environment uses cyclic reduction. If we ran greedy without it, we'd be comparing against a weaker baseline that sees a different state space. The comparison must be fair.

```bash
# GS-subset: greedy baseline on 147-instance subset
python -m ac_solver.search.miller_schupp.evaluate \
    --search-fn greedy --max-nodes 1000000 --workers 4 \
    --every-k 10 --output results/greedy_subset_k10.csv
```

### 3.2 BFS Experiments

**What it does:** FIFO queue search. Expands all states at depth d before depth d+1. Guarantees shortest solution path but explores more nodes per solved instance.

**Hyperparameters:**

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `max_nodes_to_explore` | 1,000,000 | Same as greedy, same paper reference (§4.2). BFS is exhaustive per level, so it hits the budget sooner — solves 278/1190 vs greedy's 554 |
| `cyclically_reduce_after_moves` | True | Same rationale as greedy: match PPO's `ACEnv` for fair comparison |
| `use_fixed_lengths` | True | Uses correct length recomputation from current state. The `False` option is a legacy bug preserved for backward compatibility with earlier results — we do not use it |

**What we run:**

| Experiment | Dataset | Purpose | CLI |
|------------|---------|---------|-----|
| BFS-full (already done) | 1190 instances | Full baseline, already in `results/bfs_1M.csv` | — |
| BFS-subset | 147 instances (every_k=10) | Secondary baseline for PPO comparison | `--every-k 10` |

**Why include BFS at all?** BFS provides the **shortest-path guarantee** — if PPO solves an instance that BFS also solves, we can compare path lengths. BFS also acts as a lower bound on classical search performance (greedy is the upper bound). The gap between BFS and greedy quantifies how much the greedy heuristic helps, which contextualises how much "intelligence" PPO needs to add.

```bash
# BFS-subset: BFS baseline on 147-instance subset
python -m ac_solver.search.miller_schupp.evaluate \
    --search-fn bfs --max-nodes 1000000 --workers 4 \
    --every-k 10 --output results/bfs_subset_k10.csv
```

### 3.3 PPO Ablation Experiments

**What it does:** Proximal Policy Optimization (actor-critic RL). Trains a policy network to select AC moves given a presentation state. Unlike GS/BFS, PPO requires training before evaluation.

#### 3.3.1 Fixed Hyperparameters (held constant across all runs)

These parameters are fixed to match the paper (Appendix A, Table 1) or constrained by hardware:

| Parameter | Value | Source | Justification |
|-----------|-------|--------|---------------|
| `num_envs` | 8 | Hardware | Mac hardware limit. Paper uses 28 but we cannot match this. The ablation isolates _other_ factors while holding this constant. Batch size difference (1600 vs 5600) is acknowledged as a confound but is not the variable under test |
| `num_steps` | 200 | Paper §4.4 | Rollout/horizon length. Paper uses T=200 for the constant-horizon experiments that achieved 431/1190. Variable horizon (200→1200) is a separate experiment not included in this round |
| `nodes_counts` | [512, 512] | Paper Appendix A | 2-layer MLP with 512 hidden units. Matches paper exactly. No reason to vary: the architecture is not a suspected failure mode |
| `gamma` | 0.999 | Paper Appendix A | High discount factor appropriate for long-horizon problems (T=200). With γ=0.999, the effective horizon is 1/(1-γ)=1000, well beyond the episode length |
| `gae_lambda` | 0.95 | Paper Appendix A | Standard GAE parameter. Balances bias-variance in advantage estimation |
| `clip_coef` | 0.2 | Paper Appendix A | PPO clipping threshold. Standard value; not suspected as a failure mode |
| `ent_coef` | 0.01 | Paper Appendix A | Entropy bonus. Encourages exploration. Paper value; no reason to vary |
| `vf_coef` | 0.5 | Paper Appendix A | Value function loss coefficient. Standard; paper value |
| `max_grad_norm` | 0.5 | Paper Appendix A | Gradient clipping. Prevents training instability from large gradients |
| `target_kl` | 0.01 | Paper Appendix A | Early stop within epoch if KL exceeds this. Critical safety net when `update_epochs > 1` |
| `epsilon` (Adam) | 1e-5 | Paper Appendix A | Adam optimizer epsilon. Matches paper; affects numerical stability of updates |
| `clip_rewards` | True | Paper §4.3 | Clip rewards to [-10, +1000]. This is the paper's reward function definition, not an additional processing step |
| `repeat_solved_prob` | 0.25 | Paper §4.4 | Curriculum: 25% chance of revisiting solved presentations. Prevents forgetting |
| `seed` | 1 | Convention | Fixed for reproducibility. Single seed is sufficient for this diagnostic round — we're looking for large effects (2×+), not small statistical differences |

#### 3.3.2 Varied Hyperparameters (ablation variables)

Three factors are varied in a partial factorial design:

**Factor 1: Reward scaling (`--skip-reward-division`)**

| Setting | Reward range seen by PPO | Justification |
|---------|--------------------------|---------------|
| Off (default) | [-0.0007, +0.069] | Control. Current broken behavior |
| **On** | **[-10, +1000]** | **Paper-faithful.** This is the primary hypothesis: the ÷14400 division attenuates the success signal by 14,400× and is the dominant cause of poor performance. Removing it restores the paper's intended reward scale |

Why this is the primary variable: The success reward drops from +1000 to +0.069. With `gamma=0.999` and a 200-step horizon, the discounted return for a success at step t is approximately `0.069 × 0.999^t`. At step 100, this is `0.069 × 0.905 = 0.062` — barely distinguishable from the per-step penalty of `0.0007`. The critic cannot learn meaningful value differences. With the paper's scale, the return at step 100 is `1000 × 0.905 = 905` vs per-step `-10` — a clear 90× signal-to-noise ratio.

**Factor 2: `update_epochs`**

| Setting | Justification |
|---------|---------------|
| 1 (default) | Paper value. Single pass over each batch. Control |
| **4** | **Standard PPO practice.** Schulman et al. (2017) recommend 3–10 epochs. With attenuated rewards, single-pass updates extract minimal signal. With restored rewards, 4 epochs allows the policy to extract more value from each batch. `target_kl=0.01` prevents overfitting by early-stopping within an epoch if the policy changes too much |

Why 4, not 3 or 10? 4 is conservative: enough to test the hypothesis (epochs help) without risking destructive overfitting on the small batch (1600 samples). If 4 helps, we can experiment with higher values in a follow-up. The paper's `update_epochs=1` may have been sufficient at batch_size=5600 but insufficient at 1600 — this tests that hypothesis.

**Factor 3: LR schedule**

| Setting | Schedule | Justification |
|---------|----------|---------------|
| linear→0 (default) | LR decays linearly from 1e-4 to 0 over all steps | Paper default. At 10M steps, LR reaches 0 at step 10M — fine for a diagnostic run where we evaluate at 10M. But if the best checkpoint is at 5M (as observed in previous runs), late training with near-zero LR is wasted |
| **cosine, min=0.1** | Cosine decay from 1e-4 to 1e-5 | **Prevents zero-LR plateau.** The 50M checkpoint outperforming 100M in our previous run (32 vs 29 best-of-5) proves that LR→0 causes late degradation. A cosine schedule with floor=10% maintains a minimum LR of 1e-5, allowing continued refinement throughout training. At 10M scale, this means the LR stays above 1e-5 for the entire run vs dropping below 1e-5 after ~9M with linear decay |

#### 3.3.3 Experiment Matrix

| Run | Label | skip_reward_div | update_epochs | LR schedule | Purpose |
|-----|-------|:---------------:|:-------------:|:-----------:|---------|
| 0 | `diag_baseline` | No | 1 | linear→0 | **Control.** Reproduces current broken behavior at 10M. Expected: ~2/147 solved (extrapolating from full dataset at 10M: 2/1190) |
| 1 | `diag_reward_fix` | **Yes** | 1 | linear→0 | **Isolates reward fix.** If this alone lifts performance significantly, reward attenuation is confirmed as the primary cause. All other settings identical to Run 0 |
| 2 | `diag_reward_epochs` | **Yes** | **4** | linear→0 | **Tests data reuse.** Do 4 epochs help given proper reward scale? Compares directly to Run 1 (only difference: epochs). If Run 2 >> Run 1, data reuse is a bottleneck |
| 3 | `diag_reward_cosine` | **Yes** | 1 | **cosine 0.1** | **Tests LR floor.** Does preventing LR→0 improve 10M performance? Compares directly to Run 1 (only difference: LR schedule). At 10M, linear and cosine diverge most in the final 2M steps |
| 4 | `diag_all_fixes` | **Yes** | **4** | **cosine 0.1** | **Best-case combination.** All three fixes applied. If Run 4 >> max(Run 1, 2, 3), there is positive interaction. If Run 4 ≈ max(Run 1, 2, 3), the fixes are roughly independent |

**Why not a full 2³ = 8 factorial?** The reward fix is the primary hypothesis — runs without it (epochs=4 alone, cosine alone) would compound a fix on top of a fundamentally broken signal. Testing epochs/cosine without the reward fix would be scientifically interesting but computationally wasteful: we already know the attenuated signal produces near-zero performance, so marginal improvements on near-zero are not actionable. The 5-run design tests the primary hypothesis (Run 0 vs 1) plus two secondary hypotheses (Run 1 vs 2, Run 1 vs 3) plus their combination (Run 4).

**Why seed=1 only?** This is a diagnostic experiment looking for large effects (2×+ improvement). If the reward fix is the primary cause, we expect Run 1 to solve ~5–10× more than Run 0 — an effect size that does not require multiple seeds to detect. Multi-seed experiments are appropriate for the follow-up 100M run once we've identified the winning configuration.

#### 3.3.4 Training Budget Justification

**10M timesteps per run** (`--preset mac`):

| Aspect | Value | Justification |
|--------|-------|---------------|
| Total steps | 10M | The 50M checkpoint outperformed 100M in previous runs → learning signal (or its absence) is visible well before 50M. At 10M, we've completed 10% of the paper's training budget. The previous 10M mac run (with broken rewards) solved 2/1190 greedy. If the reward fix works, 10M should be sufficient to see a clear lift |
| PPO updates | 6,250 | = 10M / 1600 batch_size. Enough for meaningful policy change: entropy dropped from 2.485 to 1.919 in our previous 10M run |
| Wall-clock | ~6h per run | On MPS (Mac). 5 runs = ~30h total, parallelizable |
| Total compute | 50M steps | = cost of a single half-run. Efficient for a diagnostic |

**Checkpoints saved at:** 1M, 5M, 10M steps (matching `MAC_PRESET`'s `eval_at` schedule). This gives three evaluation points to track the learning curve within each run.

**Why not 100M?** Cost: 5 × 100M = 500M steps = 5× the paper's full budget. The diagnostic goal is to _identify_ the right configuration, not to _complete_ the full training. If Run 4 shows strong signal at 10M, we extend that single configuration to 100M.

### 3.4 PPO Evaluation Hyperparameters

After training, each PPO checkpoint is evaluated. The evaluation has its own hyperparameters:

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `horizon` | 200 | Must match training horizon. The policy was trained with T=200; evaluating with a different horizon would test out-of-distribution behaviour, not policy quality |
| `max_relator_length` | 36 | From checkpoint config. Must match training environment |
| Deterministic (argmax) | Yes | Tests what the policy _learned_ — the mode of the action distribution. This is the strictest test: can the policy's best guess solve the instance? |
| Stochastic, best-of-5 | Yes | Tests policy + exploration. Multiple stochastic rollouts allow the policy to explore alternatives. best-of-5 is the same protocol used in our 100M evaluation, enabling direct comparison |
| `eval_seed` | 42 | Fixed for reproducibility across all runs. Same seed ensures same stochastic sequences, isolating the effect of the policy, not the randomness |

### 3.5 CLI Commands

```bash
# ── Classical baselines on 147-instance subset ──────────────────────────────

# GS-subset
python -m ac_solver.search.miller_schupp.evaluate \
    --search-fn greedy --max-nodes 1000000 --workers 4 \
    --every-k 10 --output results/greedy_subset_k10.csv

# BFS-subset
python -m ac_solver.search.miller_schupp.evaluate \
    --search-fn bfs --max-nodes 1000000 --workers 4 \
    --every-k 10 --output results/bfs_subset_k10.csv

# ── PPO ablation runs ───────────────────────────────────────────────────────

# Run 0: baseline (current broken behavior)
python -m ac_solver.agents.ppo --preset mac --seed 1 \
    --exp-name diag_baseline

# Run 1: reward fix only
python -m ac_solver.agents.ppo --preset mac --seed 1 \
    --skip-reward-division \
    --exp-name diag_reward_fix

# Run 2: reward fix + epochs=4
python -m ac_solver.agents.ppo --preset mac --seed 1 \
    --skip-reward-division --update-epochs 4 \
    --exp-name diag_reward_epochs

# Run 3: reward fix + cosine LR with floor
python -m ac_solver.agents.ppo --preset mac --seed 1 \
    --skip-reward-division --lr-decay cosine --min-lr-frac 0.1 \
    --exp-name diag_reward_cosine

# Run 4: all fixes combined
python -m ac_solver.agents.ppo --preset mac --seed 1 \
    --skip-reward-division --update-epochs 4 \
    --lr-decay cosine --min-lr-frac 0.1 \
    --exp-name diag_all_fixes

# ── PPO evaluation (repeat for each run × each checkpoint) ─────────────────

# Template: deterministic evaluation
python -m ac_solver.agents.evaluate \
    --checkpoint out/<run_label>/ckpt_<step>.pt \
    --horizon 200 --output results/<run_label>_det_<step>.csv

# Template: stochastic best-of-5 evaluation
python -m ac_solver.agents.evaluate \
    --checkpoint out/<run_label>/ckpt_<step>.pt \
    --horizon 200 --stochastic --num-rollouts 5 \
    --output results/<run_label>_sto5_<step>.csv
```

---

## 4. Evaluation Methodology

### 4.1 Metrics per Algorithm

**Greedy / BFS (per instance):**
- `solved`: bool
- `visited_nodes`: unique states explored (budget utilization)
- `path_length`: number of AC moves to solution (if solved)
- `max_intermediate_length`: peak relator length during solution path
- `length_increase`: max_intermediate - initial (measures how much the search "inflated" relators)
- `wallclock_seconds`: wall-clock time

**PPO (per instance, per checkpoint, per eval mode):**
- Same fields as above, plus:
- `algorithm`: `ppo` (deterministic) or `ppo_stochastic`
- Training curves per run: entropy, explained_variance, top1_prob, clipfrac, noop_rate

### 4.2 Comparisons

| Comparison | What it tests | Key metric |
|------------|---------------|------------|
| Run 0 vs GS-subset | How broken is current PPO vs classical? | Solve count ratio |
| **Run 1 vs Run 0** | **Does removing ÷14400 help?** (primary) | Solve count, per-(n,|w|) |
| Run 2 vs Run 1 | Does epochs=4 add value with fixed rewards? | Solve count delta |
| Run 3 vs Run 1 | Does cosine LR floor help at 10M? | Solve count at 10M vs 5M |
| Run 4 vs all | Do fixes combine superlinearly? | Solve count vs max of 1,2,3 |
| Run 4 vs GS-subset | How close to classical ceiling? | Ratio: PPO/greedy |
| PPO solved ∩ BFS solved | Do PPO solutions overlap with BFS-accessible instances? | Overlap count, path length comparison |

### 4.3 Per-(n, |w|) Breakdown

For each PPO run's best checkpoint, compute a 7×7 solve-rate matrix (same format as §2.4). This reveals:
- Which difficulty regions improve with the reward fix
- Whether PPO gains are concentrated in easy cells (low n, low |w|) or spread across the difficulty spectrum
- Whether PPO can solve any instance that greedy cannot (would be a novel result)

---

## 5. Success Criteria

| Criterion | Threshold | Interpretation |
|-----------|-----------|----------------|
| Reward fix is primary cause | Run 1 > 2× Run 0 at 10M | The ÷14400 division was indeed the dominant failure mode |
| Epochs help with fixed reward | Run 2 > Run 1 by >30% | Data reuse matters once the reward signal is adequate |
| Cosine LR prevents degradation | Run 3 at 10M > Run 3 at 5M | LR floor eliminates the late-training regression observed previously |
| Combined is viable | Run 4 > 10% solve rate (~15/147) | Strong enough signal to justify extending to 100M |
| Approaching paper | Run 4 > 36% of greedy ceiling (~25/147) | Suggests 100M run could approach the paper's 431/1190 |
| Classical baselines hold | GS-subset ≈ 46.6%, BFS-subset ≈ 23.4% | Confirms the subset is representative of the full dataset |

---

## 6. Potential Risks

1. **Critic instability with larger rewards:** Removing ÷14400 increases reward magnitude by 14,400×. The critic (initialized with orthogonal, gain=1.0) may produce large value prediction errors initially. Expected to self-correct within a few hundred updates as the critic recalibrates. If instability persists, consider adding `--norm-adv True` (already the default) as the primary stabiliser.

2. **update_epochs=4 with small batches:** 1,600 samples seen 4× may overfit. Mitigated by `clip_coef=0.2` (PPO clipping) and `target_kl=0.01` (early stopping per epoch). If KL spikes, the epoch loop terminates early — this is the designed safety mechanism.

3. **Cosine LR at 10M scale:** The cosine curve decays faster at 10M than at 100M. With `min_lr_frac=0.1`, the floor is 1e-5 — still meaningful for fine-tuning. The comparison Run 3 vs Run 1 at 10M may understate the benefit if the main advantage is at 50M–100M.

4. **Subset representativeness:** 147 instances with 1 instance per small cell (|w|≤4) means any cell-level analysis for those cells is binary (solved/unsolved, no rates). This is acceptable for the diagnostic goal but limits fine-grained difficulty analysis.

5. **Single-seed variance:** With seed=1 only, PPO runs could be lucky or unlucky. However, the expected effect sizes (2×+ for reward fix) are well above typical seed variance (~20%). If results are ambiguous (e.g., Run 1 is 1.3× Run 0), we add seeds 2, 3 for the ambiguous comparison.

---

## 7. Code Changes Summary

| File | Change | Lines |
|------|--------|-------|
| `ac_solver/agents/args.py` | Add `--skip-reward-division` flag | After line 382 |
| `ac_solver/agents/training.py` | Guard `rewards /= max_reward` with the flag | Lines 325–337 |
| `ac_solver/search/miller_schupp/evaluate.py` | Add `--every-k` CLI flag, pass to `run_evaluation` | Lines 246, 153 |

---

## 8. Next Steps After This Experiment

1. **If reward fix works (Run 1 >> Run 0):** Run the best configuration at 100M steps on full 1190 dataset
2. **If still far from paper:** Investigate batch size (try 16 envs) and variable horizon (200→400→800→1200)
3. **If all fixes combined approach paper performance:** Document as successful reproduction
4. **If no configuration helps:** The remaining gap may be due to batch size (3.5× smaller) or implementation details not captured in the paper. Consider requesting access to the paper's original codebase
