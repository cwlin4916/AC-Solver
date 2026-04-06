# PPO Training Diagnosis

**Run:** `57e5c132` — 10M steps, `--preset mac` (8 envs, 200 horizon, [512,512] MLP)
**Date:** 2026-03-10

## Summary

| Metric | Training | Evaluation (greedy argmax) |
|--------|----------|---------------------------|
| Presentations solved | 1190/1190 (100%) | 2/1190 (0.2%) |

The agent "solved" all presentations during training via stochastic random exploration, but the actual learned policy is near-uniform random and useless at eval time.

## Conclusion

**The code is correct.** The reward formula, normalization, and `max_relator_length=36` all match the original codebase design:
- `max_relator_length=36` is hardcoded in `ac_solver/agents/environment.py:87` for the Miller-Schupp dataset, derived from `max(4n+2)` for `n=1..7`
- The reward normalization (`rewards /= max_reward`) is the original design, not a bug
- The reward formula `max_reward * done - sum(lengths) * (1 - done)` is from the original `ac_env.py`

**The failure is due to insufficient training.** The `--preset mac` uses 10M steps — the paper uses 100M steps. The LR anneals linearly to 0 over all updates; with only 6,250 updates (vs the paper's ~17,857), the policy has far less time at a meaningful learning rate. The agent never had enough gradient signal to overcome the sparse reward and learn a useful policy.

**To reproduce the paper:** run with `--preset paper` (100M steps, 28 envs).

## Key Metrics at 10M Steps

| Metric | Value | Healthy Range | Verdict |
|--------|-------|---------------|---------|
| `policy_loss` | -7.15e-09 | should fluctuate | Policy not updating |
| `clipfrac` | 0.0 | 0.1-0.3 | No policy changes |
| `approx_kl` | -1.43e-08 | 0.001-0.01 | Policy frozen |
| `entropy` | 2.45 | should decrease | Near-uniform random (~12 actions) |
| `explained_var` | -1.75 | 0.5-0.9 | Critic worse than mean predictor |
| `mean_return` | -0.125 | should be positive | Negligible learning signal |

At 5M steps the metrics were already dead: `clipfrac=0.0`, `approx_kl=3.27e-07`, `entropy=2.47`.

## Eval Results Across Checkpoints

| Checkpoint | Solved |
|-----------|--------|
| 1M steps  | 0/1190 |
| 5M steps  | 4/1190 |
| 10M steps | 2/1190 |

Classical baselines: Greedy search = 554/1190, BFS = 278/1190.

## Root Causes

### 1. Insufficient training (primary cause)

| Setting | Paper | Mac Preset | Ratio |
|---------|-------|------------|-------|
| `total_timesteps` | 100M | 10M | 10x fewer |
| `num_envs` | 28 | 8 | 3.5x fewer |
| PPO updates | ~17,857 | 6,250 | 2.9x fewer |

The paper uses 100M steps with 28 parallel envs. The mac preset has 10M with 8 envs. With LR annealing to 0 over all updates, the mac preset gives the policy far fewer updates at a meaningful learning rate before the LR decays to near-zero.

### 2. LR anneals to 0 too quickly at 10M scale

**File:** `ac_solver/agents/training.py:176-185`

Config: `anneal_lr=True`, `lr_decay=linear`, `min_lr_frac=0.0`, `learning_rate=1e-4`

The LR decays linearly from 1e-4 to 0 over 6250 updates. By mid-training it's already ~5e-5. At the paper's 100M scale this is spread over ~17,857 updates — 3x more time at a useful LR. This is not a bug (same schedule as paper), but it means the mac preset doesn't give enough time.

### 3. Reward normalization makes signal small (by design, not a bug)

**File:** `ac_solver/agents/training.py:280-286`

```python
if not args.norm_rewards:
    base_env = envs.envs[0].unwrapped if hasattr(envs.envs[0], 'unwrapped') else envs.envs[0]
    rewards /= base_env.max_reward  # max_reward = horizon * max_relator_length * 2 = 200 * 36 * 2 = 14400
```

Raw rewards are clipped to [-10, 1000]. After dividing by 14400, the terminal success reward (+1000) becomes ~0.069 and per-step penalties become ~-0.0007. This is the original design — at 100M steps there's enough experience for the small signal to accumulate. At 10M steps, there isn't.

### 4. Training "solved" via random walk, not learned policy

With entropy=2.45 (near-uniform over ~12 actions), the training loop's stochastic sampling is essentially random search. Over 200-step episodes with 8 parallel envs and 52,801 episodes, many easy presentations get solved by chance. But the policy weights never captured this — eval's argmax picks from a flat distribution.

## Config Dump

```
preset: mac
seed: 1
num_envs: 8
num_steps: 200
horizon_length: 200
total_timesteps: 10,000,000
batch_size: 1600
minibatch_size: 400
learning_rate: 1e-4
lr_decay: linear
min_lr_frac: 0.0
anneal_lr: True
gamma: 0.999
gae_lambda: 0.95
update_epochs: 1
clip_coef: 0.2
ent_coef: 0.01
vf_coef: 0.5
max_grad_norm: 0.5
target_kl: 0.01
clip_rewards: True (range [-10, 1000])
nodes_counts: [512, 512]
max_relator_length: 36
states_type: all
repeat_solved_prob: 0.25
```

## Codebase Reference

### `ac_solver/agents/args.py` — CLI arguments and presets

Defines two presets:
- **PAPER_PRESET**: 100M timesteps, 28 envs, LR=1e-4 with linear decay to 0, horizon=200, [512,512] MLP, update_epochs=1, clip_coef=0.2, ent_coef=0.01, gamma=0.999, target_kl=0.01, clip_rewards to [-10, 1000].
- **MAC_PRESET**: inherits all PAPER_PRESET values but overrides `num_envs=8` and `total_timesteps=10M`.

`parse_args()` uses a two-pass approach: first extracts `--preset`, then sets those values as defaults so CLI flags can override them. Computes `batch_size = num_envs * num_steps` (paper: 5600, mac: 1600) and `minibatch_size = batch_size / num_minibatches`.

Note: `max_relator_length` default is 7 in args, but gets overridden to 36 in `environment.py` when using the Miller-Schupp dataset (see below).

### `ac_solver/agents/ppo.py` — Entry point

The `train_ppo()` function: sets seeds, resolves device (auto-detects MPS on Mac), calls `get_env(args)` to create vectorized environments, creates the Agent and Adam optimizer, optionally loads a checkpoint via `--resume` (restoring actor/critic/optimizer state + training metadata), then calls `ppo_training_loop()`.

### `ac_solver/agents/training.py` — PPO training loop

The core training loop with these key sections:

**Rollout phase (lines 188-276):** For each of `num_steps` steps across `num_envs` parallel envs, samples actions stochastically from the policy (`agent.get_action_and_value(next_obs)` returns action from `Categorical.sample()`), executes in envs, collects obs/actions/rewards/dones. When an episode ends (done or truncated), picks the next presentation from the curriculum: cycles through all 1190 presentations first, then with probability `repeat_solved_prob=0.25` picks a solved one, otherwise picks an unsolved one.

**Reward normalization (lines 278-290):** When `norm_rewards=False` (the default), divides all rewards by `base_env.max_reward`. For the Miller-Schupp dataset, `max_reward = horizon * max_relator_length * 2 = 200 * 36 * 2 = 14400`. This shrinks the clipped reward range from [-10, 1000] to approximately [-0.0007, 0.069].

**GAE computation (lines 292-310):** Standard Generalized Advantage Estimation with `gamma=0.999`, `gae_lambda=0.95`.

**PPO update (lines 322-404):** For `update_epochs=1` epoch, shuffles batch, splits into minibatches of size 400. Computes new log-probs and values, calculates clipped policy loss (`clip_coef=0.2`), clipped value loss, and entropy bonus (`ent_coef=0.01`). Total loss = `pg_loss - ent_coef * entropy + vf_coef * v_loss`. Gradient clipped to `max_grad_norm=0.5`. Early-stops if `approx_kl > target_kl (0.01)`.

**LR annealing (lines 176-185):** Calls `get_curr_lr()` which linearly decays LR from `1e-4` to `min_lr_frac * 1e-4 = 0` over all updates. With mac preset (6250 updates), LR reaches 0 at the end. With paper preset (~17,857 updates), the same decay is spread over 3x more updates.

**Checkpointing (lines 437-461):** Saves `ckpt.pt` every 100 updates. Saves named checkpoints (`ckpt_1000000.pt`, etc.) when `global_step` crosses thresholds in `eval_at`. Checkpoint includes: actor/critic weights, optimizer state, update number, global_step, episode count, success_record (which presentations solved/unsolved), curriculum state, training metrics.

### `ac_solver/agents/ppo_agent.py` — Actor-critic network

`Agent` is an `nn.Module` with separate actor and critic networks:
- **Actor**: `input_dim → 512 → Tanh → 512 → Tanh → 12` (12 = number of AC moves). Output is logits for a `Categorical` distribution. Final layer initialized with std=0.01 (near-uniform initial policy).
- **Critic**: `input_dim → 512 → Tanh → 512 → Tanh → 1`. Final layer initialized with std=1.0.
- `input_dim = max_relator_length * 2 = 72` (the flattened presentation).

`get_action_and_value(x, action=None)`: computes logits and value. If `action` is None, samples from `Categorical(logits=logits)`. Returns (action, log_prob, entropy, value).

During **training**: actions are sampled stochastically (`probs.sample()`).
During **evaluation**: actions are chosen greedily (`logits.argmax()`). This is the key gap — a near-uniform policy samples diverse actions (enabling accidental solves) but argmax picks a fixed (arbitrary) action per state.

### `ac_solver/agents/environment.py` — Environment setup

`get_env(args)` handles two modes:
- **Fixed init state**: uses a single presentation specified by `--relator1` and `--relator2`.
- **Miller-Schupp dataset** (default, `fixed_init_state=False`): loads 1190 presentations from `all_presentations.txt`, **hardcodes `args.max_relator_length = 36`** (line 87, derived from `max(4n+2)` for `n=1..7`), pads all presentations to length 72 (36 per relator), creates `SyncVectorEnv` with `num_envs` parallel instances.

`make_env(presentation, args)` creates a thunk that initializes `ACEnv` and optionally wraps it with `gym.wrappers.NormalizeReward` (if `norm_rewards=True`) and `gym.wrappers.TransformReward` for reward clipping to `[min_rew, max_rew]` = `[-10, 1000]`.

The reward clipping happens **before** the `/max_reward` normalization in `training.py`. So the flow is: raw reward (from `ac_env.py`) → clip to [-10, 1000] → divide by 14400.

### `ac_solver/agents/evaluate.py` — Greedy evaluation

Loads a checkpoint, reconstructs the Agent from saved config (obs_dim, nodes_counts), loads actor weights, sets to eval mode. Runs greedy (argmax) rollouts on the Miller-Schupp dataset (1190 instances or 1/5 stratified subset).

Per instance: resets env with that presentation, runs up to `horizon=200` steps, at each step picks `action = logits.argmax()`. Records whether solved, path length, max intermediate length, wallclock time. Writes CSV with same schema as classical search evaluation for direct comparison.

The **critical difference** from training: training uses `Categorical.sample()` (stochastic), evaluation uses `logits.argmax()` (deterministic). With a near-uniform policy (entropy=2.45, ~12 actions), stochastic sampling explores randomly and can stumble on solutions, while argmax deterministically picks the same (arbitrary) action every time and gets stuck.

### `ac_solver/envs/ac_env.py` — AC Environment

Gymnasium environment where:
- **State**: a numpy array of shape `(max_relator_length * 2,)` = `(72,)` representing a balanced presentation `<r_0, r_1>` with 2 generators. Each relator is zero-padded to `max_relator_length=36`. Values are in `{-2, -1, 0, 1, 2}` where `0` = padding, `±1` = generator x, `±2` = generator y.
- **Action space**: `Discrete(12)` — the 12 AC moves.
- **Reward**: `max_reward * done - sum(lengths) * (1 - done)` where `max_reward = horizon * max_relator_length * n_gen = 200 * 36 * 2 = 14400`. If the presentation is trivialized (`sum(lengths) == 2`, meaning both relators are single generators), reward = +14400. Otherwise, reward = `-sum(lengths)` (negative of total relator length, typically -5 to -70).
- **Done**: when `sum(lengths) == 2` (trivial presentation).
- **Truncated**: after `horizon_length=200` steps.

### `ac_solver/envs/ac_moves.py` — The 12 AC moves

`ACMove(move_id, presentation, max_relator_length, lengths)` applies one of 12 moves:
- **Moves 0-3 (concatenations)**: replace `r_i` with `r_i * r_j^{±1}` (multiply one relator by the other or its inverse). Includes free reduction (cancellation of adjacent inverse generators).
- **Moves 4-11 (conjugations)**: replace `r_i` with `x_j^{±1} * r_i * x_j^{∓1}` (conjugate a relator by a generator). Includes cancellation at boundaries.

All moves check if the result exceeds `max_relator_length`; if so, the presentation is returned unchanged (move is a no-op). After each move, `simplify_presentation()` applies free and cyclic reductions.

### Data flow summary

```
args.py (--preset mac → 10M steps, 8 envs)
  → ppo.py (create envs, agent, optimizer)
    → environment.py (load 1190 presentations, pad to length 72, set max_relator_length=36)
      → training.py:
          rollout: sample actions stochastically from actor
            → ac_env.py: apply ACMove, compute reward = 14400*done - sum(lengths)*(1-done)
            → clip reward to [-10, 1000]
          normalize: reward /= 14400
          compute GAE advantages
          PPO update: policy loss + value loss + entropy bonus
          anneal LR linearly toward 0
          save checkpoints
  → evaluate.py (load checkpoint, run argmax policy → 2/1190 solved)
```
