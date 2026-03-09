# AlphaZero Adaptation Audit: AC-Solver Problem Characterization

**Date:** 2026-03-08
**Purpose:** Comprehensive audit of the Andrews-Curtis (AC) problem characteristics and the AlphaZero_PP framework, to inform an LLM-assisted execution plan for an AlphaZero-style solver.

---

## 1. Problem Definition

The **Andrews-Curtis conjecture** asks whether every balanced presentation of the trivial group can be reduced to the trivial presentation via a sequence of AC moves. The AC-Solver attempts to find such reduction sequences computationally.

**Mathematical Setup:**
- A *balanced presentation* has 2 generators (x, y) and 2 relators (r₀, r₁)
- The *trivial presentation* has both relators of length 1 (e.g., `[x, y]`)
- AC moves transform relators while preserving the presented group
- The 5 types of AC moves are: (1) replace rᵢ with rᵢ · rⱼ, (2) replace rᵢ with rⱼ · rᵢ, (3) replace rᵢ with rᵢ⁻¹, (4) conjugate rᵢ by a generator g (i.e., rᵢ → g⁻¹ · rᵢ · g), (5) swap r₀ and r₁. The 12 discrete actions in the implementation are specific instantiations of moves (1), (2), and (4) for 2 generators.

**Key algebraic operations applied after every move:**
- **Free reduction:** Cancel adjacent inverse pairs in a word (e.g., `x · x⁻¹ → ε`, `y⁻¹ · y → ε`). Applied iteratively until no more cancellations exist.
- **Cyclic reduction:** Remove matching inverse pairs at the start and end of a word (e.g., `x · w · x⁻¹ → w`). Applied because relators represent conjugacy classes.

---

## 2. Environment Specification

### 2.1 State Space

| Property | Value |
|---|---|
| **Representation** | 1-D NumPy array of int8, length `2 * max_relator_length` |
| **Default shape** | `(72,)` — two relators concatenated, each padded to length 36 |
| **Alphabet** | `{-2, -1, 0, 1, 2}` where ±1 = generator x/x⁻¹, ±2 = generator y/y⁻¹, 0 = padding |
| **Constraints** | Zeros must be right-padded; each relator has ≥1 non-zero element |
| **Observation space** | `Box(low=-2, high=2, shape=(72,), dtype=int8)` |
| **Theoretical state space** | ~5^72 ≈ 10^50 (practical reachable space much smaller) |

**Source:** `ac_solver/envs/ac_env.py` (lines 56-92), `ac_solver/envs/utils.py` (lines 13-54)

### 2.2 Action Space

**Discrete(12)** — 4 concatenation moves + 8 conjugation moves, constant branching factor.

| Action | Type | Operation |
|--------|------|-----------|
| 0 | Concatenation | r₀ → r₀ · r₁ |
| 1 | Concatenation | r₁ → r₁ · r₀⁻¹ |
| 2 | Concatenation | r₀ → r₀ · r₁⁻¹ |
| 3 | Concatenation | r₁ → r₁ · r₀ |
| 4 | Conjugation | r₁ → x⁻¹ · r₁ · x |
| 5 | Conjugation | r₀ → y⁻¹ · r₀ · y |
| 6 | Conjugation | r₁ → y⁻¹ · r₁ · y |
| 7 | Conjugation | r₀ → x · r₀ · x⁻¹ |
| 8 | Conjugation | r₁ → x · r₁ · x⁻¹ |
| 9 | Conjugation | r₀ → y · r₀ · y⁻¹ |
| 10 | Conjugation | r₁ → y · r₁ · y⁻¹ |
| 11 | Conjugation | r₀ → x⁻¹ · r₀ · x |

- All 12 actions are always "legal" but may be no-ops if result exceeds `max_relator_length`
- After each move: free reduction (cancel adjacent inverses) + cyclic reduction applied automatically
- **Source:** `ac_solver/envs/ac_moves.py` (lines 159-280)

### 2.3 Reward Structure

```
if done (sum_of_lengths == 2):
    reward = horizon_length * max_relator_length * 2  (e.g., 72,000)
else:
    reward = -(r₀_length + r₁_length)
```

- **Dense negative signal** each step (penalizes total word length)
- **Sparse large positive signal** on success
- Optional clipping to `[-10, 1000]` and/or running normalization
- **Source:** `ac_solver/envs/ac_env.py` (lines 95-113)

### 2.4 Episode Termination

| Condition | Trigger |
|---|---|
| **Success (done)** | `sum(relator_lengths) == 2` (trivial presentation) |
| **Failure (truncated)** | `step_count >= horizon_length` (default: 1000) |

---

## 3. Problem Characteristics Relevant to AlphaZero

### 3.1 Favorable Properties

| Property | Value | Impact on AlphaZero |
|---|---|---|
| **Action space size** | 12 | Very small — MCTS branching is highly manageable (cf. Go: 361, Chess: ~35) |
| **Deterministic dynamics** | Yes | Perfect for MCTS — no stochastic transitions to model |
| **Fully observable** | Yes | No hidden information; state fully determines future |
| **Single-player** | Yes | No adversary; simplifies MCTS (no minimax needed) |
| **Clear terminal condition** | Yes | Unambiguous goal state detection |
| **Fixed action set** | Yes | No variable-size action masking needed |

### 3.2 Challenging Properties

| Property | Value | Concern |
|---|---|---|
| **Episode length** | Up to 1000-2000 steps | MCTS at every step is expensive; need efficient simulation budgets |
| **Sparse terminal reward** | Large bonus only at goal | Value network must learn long-horizon goal reachability |
| **Non-monotonic landscape** | Solutions may require lengthening before shortening | Greedy heuristics fail; value network must learn this |
| **Large state space** | ~10^50 theoretical | Generalization critical; cannot memorize states |
| **Solution length** | 20-100+ moves | Credit assignment over long sequences is hard |
| **No self-play** | Single-agent problem | Must adapt training loop (no opponent modeling) |

### 3.3 Why Current Approaches Fail on Hard Instances

1. **Greedy search** gets trapped in local minima — commits to locally minimal word lengths but optimal paths often require temporary length increases
2. **BFS** explores exhaustively but hits the 10,000 node budget before finding solutions that require 50+ moves at branching factor 12
3. **PPO (greedy inference)** follows a single path with no lookahead or backtracking — one bad move derails the entire episode

---

## 4. Dataset: Miller-Schupp Presentations

**Definition:** `MS(n, w) = <x, y | x⁻¹yⁿx = yⁿ⁺¹, x = w>` where `exponent_sum(x in w) = 0`

| Property | Value |
|---|---|
| **Total instances** | 1,190 |
| **Parameter n** | 1 to 7 |
| **Word length(w)** | 1 to 7 |
| **Max relator length** | 36 (from `max(4n+2)` for n∈[1,7]) |
| **Difficulty** | Increases with n and length(w) |

**Source:** `ac_solver/search/miller_schupp/miller_schupp.py`, dataset at `ac_solver/search/miller_schupp/data/all_presentations.txt`

### 4.1 Concrete Example

**AK(2) presentation** (Akbulut-Kirby series, n=2):
- Initial state array: `[1, 1, -2, -2, -2, 0, 0, 1, 2, 1, -2, -1, -2, 0]` (padded to full length)
- This encodes r₀ = `x x y⁻¹ y⁻¹ y⁻¹` (length 5) and r₁ = `x y x y⁻¹ x⁻¹ y⁻¹` (length 6)
- Total initial length = 11
- Known solution: 23 AC moves to reach trivial presentation `[x, y]`

---

## 5. Baseline Results

### 5.1 Greedy Search (1M node budget)
- **545 / 1,044 instances solved** (52.2%)
- Evaluated on 1,044 of the 1,190 Miller-Schupp instances
- Fails on hard instances where word length must temporarily increase

### 5.2 BFS (1M node budget)
- **50 / 57 instances solved** (87.7%)
- Only evaluated on a small subset (57 instances) — too slow for full dataset
- Finds optimal (shortest) paths but computationally prohibitive at scale

### 5.3 PPO
- Trained for up to 100M timesteps
- Uses greedy argmax inference (no search)
- Exact solve rate on full dataset not yet benchmarked in results/

**Takeaway:** ~48% of instances remain unsolved by greedy search at 1M nodes. An AlphaZero approach that combines learned heuristics with MCTS lookahead could potentially solve more of these hard instances.

---

## 6. Current PPO Architecture

### 6.1 Architecture

```
Actor:  state(72) → Linear(512) → Tanh → Linear(512) → Tanh → Linear(12) → Categorical
Critic: state(72) → Linear(512) → Tanh → Linear(512) → Tanh → Linear(1) → scalar V(s)
```

- Orthogonal weight initialization
- Separate actor/critic (no shared trunk)
- **Source:** `ac_solver/agents/ppo_agent.py`

### 6.2 Training Configuration (Paper Preset)

| Hyperparameter | Value |
|---|---|
| Parallel envs | 28 |
| Rollout steps | 200 |
| Total timesteps | 100M |
| Learning rate | 1e-4 (linear decay) |
| Discount (gamma) | 0.999 |
| GAE lambda | 0.95 |
| PPO clip | 0.2 |
| Entropy coef | 0.01 |
| Value coef | 0.5 |
| Update epochs | 1 |
| Minibatches | 4 |

### 6.3 Inference

- **Greedy argmax** over actor logits — no sampling, no search
- Single deterministic path per problem
- Value network (critic) is discarded at inference time

---

## 7. AlphaZero_PP Framework Assessment

### 7.1 Framework Overview

**Repository:** `/Users/miltonlin/Documents/GitHub/AlphaZero_PP`
**Design:** Modular single-player AlphaZero implementation with pluggable games.

**Core Components:**
- `src/alphazeropp/core/game.py` — Abstract game interface (Gymnasium-compatible)
- `src/alphazeropp/core/mcts.py` — Full MCTS with UCB, Dirichlet noise, action masking, tree reuse
- `src/alphazeropp/core/policy_value_net.py` — Abstract NN interface (PyTorch)
- `src/alphazeropp/core/agent.py` — Agent combining MCTS + NN for self-play
- `src/alphazeropp/core/config.py` — Dataclass-based configuration
- `src/alphazeropp/training/trainer.py` — Training loop with multiprocessing

### 7.2 Game Interface Requirements

To plug AC-Solver into AlphaZero_PP, implement the `Game` abstract class:

```python
class Game[ObsType, ActType]:
    action_space: gymnasium.Space       # Discrete(12)
    observation_space: gymnasium.Space   # Box(-2, 2, (72,), int8)

    def reset() -> (obs, info)
    def step(action) -> (obs, reward, terminated, truncated, info)
    def get_action_mask() -> np.ndarray  # shape (12,), boolean
    def clone() -> Game                  # deep copy for MCTS simulation
    def stash_state() / unstash_state()  # save/restore for MCTS

    @property
    def hashable_obs -> Hashable         # for MCTS node dedup (default: .tobytes())
```

### 7.3 MCTS Capabilities

| Feature | Supported | Notes |
|---|---|---|
| UCB exploration | Yes | Configurable `c_exploration` |
| Dirichlet noise | Yes | `alpha`, `epsilon` parameters |
| Action masking | Yes | Boolean mask per state |
| Tree reuse | Yes | `perform_simulations_reuse()` + `advance_to(action)` |
| Backup rules | Multiple | mean, max, topk, softmax |
| Rollout evaluation | Yes | Configurable depth and blend with NN value |
| Min-max Q normalization | Yes | For UCB calculation |

### 7.4 Training Loop

1. **Self-play:** Play `n_games_per_train` episodes using MCTS-guided policy
2. **Data collection:** Store `(state, MCTS_policy_probs, discounted_return)` tuples
3. **Network training:** Train on collected data with replay buffer (`n_past_iterations_to_train`)
4. **Loss:** `MSE(value) + weight * CrossEntropy(policy)`
5. **Parallelization:** Multiprocessing across episodes

### 7.5 Existing Game Implementations

| Game | State Dim | Actions | Episode Length | Type |
|---|---|---|---|---|
| BitString | n_sites | n_sites | 2×n_sites | Combinatorial optimization |
| CartPole | 4 | 2 | 100-500 | Control |
| Doors | Variable | Variable | Variable | Planning (PDDL) |

---

## 8. Compatibility Assessment

### 8.1 Direct Compatibility (Works Out-of-Box)

- Discrete action space (12 actions)
- Fixed observation shape (72-dim)
- Deterministic transitions
- Single-player (no opponent)
- Clear terminal detection
- State hashing via `.tobytes()`

### 8.2 Adaptations Required

| Component | What's Needed | Complexity |
|---|---|---|
| **ACGame class** | Wrap `ACEnv` into `Game` interface; implement `clone()`, `stash/unstash_state()` | Low |
| **Network** | Use existing `PolicyValueNetModel` or custom; input=72, output=12 | Low |
| **Config** | Create `ACConfig` dataclass with tuned hyperparameters | Low |
| **Reward shaping** | Current dense reward (-word_length) should work; may need tuning | Medium |
| **Episode length** | 1000 steps with MCTS at each = 1000 × n_simulations NN forward passes per episode | Medium |
| **Training efficiency** | Long episodes mean fewer completed episodes per training iteration | Medium |

### 8.3 Key Open Questions for Execution Plan

1. **Simulation budget:** How many MCTS simulations per move? (12 actions suggests 25-100 is reasonable)
2. **Tree reuse:** Should we reuse trees across steps? (Yes — critical for long episodes)
3. **Reward design:** Keep dense -length reward, use only sparse terminal, or potential-based shaping?
4. **Network architecture:** Keep 2-layer MLP or upgrade to deeper/residual network?
5. **Curriculum:** Train on easy instances first (small n) then progress to harder ones?
6. **Discount factor:** gamma=1.0 (AlphaZero default) vs gamma=0.999 (PPO default)?
7. **Evaluation protocol:** Compare against PPO baseline, greedy search, and BFS on same 1,190 instances?
8. **Compute budget:** How many training iterations / self-play games are feasible?

---

## 9. Key File Paths

### AC-Solver
| Component | Path |
|---|---|
| AC Environment | `ac_solver/envs/ac_env.py` |
| AC Moves | `ac_solver/envs/ac_moves.py` |
| Env Utilities | `ac_solver/envs/utils.py` |
| PPO Agent | `ac_solver/agents/ppo_agent.py` |
| PPO Training | `ac_solver/agents/training.py` |
| PPO Entry Point | `ac_solver/agents/ppo.py` |
| PPO Args | `ac_solver/agents/args.py` |
| Env Wrapper | `ac_solver/agents/environment.py` |
| Evaluation | `ac_solver/agents/evaluate.py` |
| Greedy Search | `ac_solver/search/greedy.py` |
| BFS Search | `ac_solver/search/breadth_first.py` |
| Miller-Schupp Generator | `ac_solver/search/miller_schupp/miller_schupp.py` |
| Dataset | `ac_solver/search/miller_schupp/data/all_presentations.txt` |

### AlphaZero_PP
| Component | Path |
|---|---|
| Game Interface | `src/alphazeropp/core/game.py` |
| MCTS | `src/alphazeropp/core/mcts.py` |
| Policy-Value Net | `src/alphazeropp/core/policy_value_net.py` |
| Agent | `src/alphazeropp/core/agent.py` |
| Config | `src/alphazeropp/core/config.py` |
| Trainer | `src/alphazeropp/training/trainer.py` |
| BitString Example | `src/alphazeropp/instances/bitstring/` |
| CartPole Example | `src/alphazeropp/instances/cartpole/` |
