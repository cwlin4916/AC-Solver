# LLM Prompt: AlphaZero Architecture Design & Staged Roadmap for AC-Solver

---

## Instructions for the LLM

You are a research engineer specializing in reinforcement learning, Monte Carlo Tree Search (MCTS), and combinatorial optimization. You have deep expertise in AlphaZero-style algorithms and their adaptation to single-agent search problems.

I am building an AlphaZero-style solver for the Andrews-Curtis (AC) conjecture — a problem in combinatorial group theory where the goal is to reduce a balanced group presentation to the trivial presentation via a sequence of algebraic moves. I have an existing PPO baseline and an existing AlphaZero framework (AlphaZero_PP) that I want to adapt.

Below is a comprehensive audit of the problem, environment, current baselines, and the AlphaZero_PP framework. Read it carefully before answering.

---

## Context: Problem Audit

<PASTE THE FULL CONTENTS OF spec/alphazero-audit-2026-03-08.md HERE>

---

## Your Task

I need you to produce three things: (1) a literature review situating this problem, (2) a thorough theoretical analysis of the proposed approach, and (3) a staged implementation roadmap. Do not restrict yourself to a rigid template — use your judgment on how to best structure the response, but ensure all of the following areas are covered.

### Part 1: Literature Review

Situate the AC-Solver problem within the broader landscape of related work. Specifically:

1. **AlphaZero and its single-agent adaptations** — Survey how AlphaZero has been adapted beyond two-player games. Cover DeepCube (Rubik's Cube), puzzle solvers, theorem provers, and any combinatorial search applications. What worked, what didn't, and why?

2. **RL for combinatorial and algebraic problems** — Are there prior applications of RL or MCTS to problems in group theory, topology, knot theory, or related algebraic domains? What about program synthesis, automated reasoning, or symbolic manipulation?

3. **Search algorithms for the Andrews-Curtis conjecture** — What computational approaches have been tried for AC? (e.g., Havas & Ramsay, Miasnikov, Lisitsa & Potapov, Bowman & McCaul). How do they compare to RL-based approaches?

4. **Relevant algorithmic innovations** — Identify specific techniques from the literature that are particularly relevant to this problem (e.g., hindsight experience replay for sparse rewards, curriculum learning for variable-difficulty instances, potential-based reward shaping, warm-starting MCTS with a pre-trained policy).

For each paper or technique discussed, explain its relevance to the AC problem and whether/how it should inform our design.

### Part 2: Theoretical Analysis & Architecture Design

Provide a detailed, worked-through theoretical discussion of how an AlphaZero-style approach should be designed for this problem. For each design decision:
- Present the options explicitly
- Work through the reasoning (not just state conclusions)
- Discuss benefits AND potential drawbacks/failure modes
- Justify with references from Part 1

Cover at minimum these design dimensions:

1. **State representation**: Raw int8 array vs one-hot encoding vs learned embeddings vs sequence model input. Work through the information-theoretic and computational trade-offs.

2. **Network architecture**: MLP vs residual network vs 1D-CNN vs transformer. Consider the sequential/symbolic nature of relators, the observation size (72 dims), the need to learn non-monotonic value landscapes, and compute constraints (single GPU). Work through why certain architectures succeed or fail on similar problems.

3. **Shared vs separate trunk**: AlphaZero-style (shared) vs PPO-style (separate). What does the literature say about when sharing helps vs hurts?

4. **MCTS configuration**: Work through the reasoning for specific values of:
   - Number of simulations per move (given branching factor 12, episode length up to 1000)
   - Exploration constant (c_puct)
   - Dirichlet noise parameters
   - Backup rule (mean vs max vs other)
   - Tree reuse across steps (cost-benefit given deterministic dynamics)
   - Discount factor (gamma = 1.0 vs 0.999 vs other)

5. **Reward design**: Work through the interaction between reward signal and MCTS value estimation. Consider sparse terminal, dense -word_length, potential-based shaping (Ng et al., 1999), and combinations. What are the failure modes of each? How does the choice affect the value network's learning dynamics?

6. **Training data generation**: Curriculum vs uniform sampling. Replay buffer sizing. How to handle the wide difficulty distribution (5-move vs 100+-move solutions). What does the self-play loop look like concretely?

7. **Potential failure modes and mitigations**: What are the most likely ways this approach could fail? (e.g., value network fails to generalize, MCTS too slow for long episodes, catastrophic forgetting across difficulty levels). For each failure mode, propose a concrete mitigation strategy.

### Part 3: Staged Implementation Roadmap

Break the full implementation into **sequential stages** where each stage is independently testable and produces measurable results before proceeding to the next. For each stage, provide:

1. **Stage title and objective** (1 sentence)
2. **Theoretical justification** — why this stage is necessary and what it validates, connecting back to the analysis in Part 2
3. **Success criteria** — specific, measurable conditions to pass before moving to the next stage (e.g., "solve X% of n=1 instances", "value loss < Y on held-out set")
4. **Claude Code prompt** — an explicit, self-contained prompt I can paste into Claude Code (the Anthropic CLI tool) to implement that stage. The prompt must:
   - Reference specific files to read and modify (use paths from the audit document)
   - Specify what code to write, where to put it, and what tests to create
   - Include the architectural decisions from Part 2
   - Be concrete enough that Claude Code can execute it without ambiguity
   - Not assume knowledge from previous stages — each prompt should be self-contained with necessary context

### Part 4: References

Provide a BibTeX bibliography of **all** papers cited in Parts 1-3. Include at minimum:
- Silver et al. (2018) — AlphaZero
- Agostinelli et al. (2019) — DeepCube / Solving the Rubik's Cube with deep RL
- Ng et al. (1999) — Policy invariance under reward transformations
- Schrittwieser et al. (2020) — MuZero
- Any papers on AC conjecture computation (Havas, Miasnikov, Lisitsa, Bowman, etc.)
- Any other papers you cite for architectural choices, MCTS theory, curriculum learning, or combinatorial RL

---

## Output Format Requirements

- Use markdown with clear headers and subheaders
- Number all stages sequentially (Stage 1, Stage 2, ...)
- Each Claude Code prompt must be in a fenced code block marked as ```prompt
- BibTeX entries must be in a fenced code block marked as ```bibtex
- Be precise — avoid vague advice like "tune hyperparameters". Give specific values with justification.
- Total expected stages: 4-8 (not fewer than 4, not more than 8)
- Each stage should be completable in 1-3 coding sessions

---

## Constraints

- The AlphaZero_PP framework already exists and should be reused — do not propose writing MCTS from scratch
- The AC environment (ac_solver/envs/) already exists and should be wrapped, not rewritten
- Target hardware: Apple M-series (MPS) or single NVIDIA GPU
- Python 3.10+, PyTorch, Gymnasium
- The PPO baseline solves ~52% of instances with greedy search at 1M nodes. The goal is to significantly exceed this.
