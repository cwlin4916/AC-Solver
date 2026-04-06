"""
Evaluate a trained PPO checkpoint on Miller-Schupp presentations.

Runs greedy (argmax) policy rollouts and writes per-instance metrics to CSV
using the same schema as the classical search evaluate.py for compatibility
with plot_results.py.

Usage:
    python -m ac_solver.agents.evaluate \
        --checkpoint results/ppo_checkpoints/ckpt_1000000.pt \
        --horizon 200 \
        --output results/ppo_eval_1e6.csv

    Add --full to evaluate all 1190 instances (default: 1/5 stratified subset).
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
from torch.distributions import Categorical
from tqdm import tqdm

from ac_solver.agents.ppo_agent import Agent
from ac_solver.envs.ac_env import ACEnv, ACEnvConfig
from ac_solver.envs.utils import change_max_relator_length_of_presentation
from ac_solver.search.miller_schupp.dataset import (
    load_tagged_dataset,
    load_stratified_subset,
)

# Same schema as ac_solver/search/miller_schupp/evaluate.py
CSV_FIELDS = [
    "instance_id",
    "n",
    "lenw",
    "algorithm",
    "max_nodes",
    "solved",
    "visited_nodes",
    "path_length",
    "initial_total_length",
    "max_intermediate_length",
    "length_increase",
    "wallclock_seconds",
]


def evaluate_instance(agent, device, instance, horizon, max_relator_length,
                      stochastic=False):
    """Run a single rollout on an instance. Greedy (argmax) or stochastic."""
    presentation = instance["presentation"]
    presentation = change_max_relator_length_of_presentation(
        presentation, max_relator_length
    )

    config = ACEnvConfig(
        initial_state=np.array(presentation, dtype=np.int8),
        horizon_length=horizon,
    )
    env = ACEnv(config)
    obs, _ = env.reset()

    initial_total_length = sum(env.lengths)
    max_intermediate_length = initial_total_length
    solved = False
    path_length = 0

    t_start = time.time()
    for step in range(horizon):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = agent.actor(obs_tensor)
        if stochastic:
            action = Categorical(logits=logits).sample().item()
        else:
            action = logits.argmax(dim=-1).item()

        obs, reward, done, truncated, info = env.step(action)
        path_length = step + 1
        current_length = sum(env.lengths)
        max_intermediate_length = max(max_intermediate_length, current_length)

        if done:
            solved = True
            break
        if truncated:
            break

    wallclock = time.time() - t_start
    length_increase = max_intermediate_length - initial_total_length
    algorithm = "ppo_stochastic" if stochastic else "ppo"

    return {
        "instance_id": instance["instance_id"],
        "n": instance["n"],
        "lenw": instance["lenw"],
        "algorithm": algorithm,
        "max_nodes": horizon,
        "solved": solved,
        "visited_nodes": path_length,
        "path_length": path_length if solved else "",
        "initial_total_length": initial_total_length,
        "max_intermediate_length": max_intermediate_length if solved else "",
        "length_increase": length_increase if solved else "",
        "wallclock_seconds": round(wallclock, 6),
    }


def evaluate_instance_best_of_n(agent, device, instance, horizon,
                                max_relator_length, num_rollouts, eval_seed):
    """Run N stochastic rollouts and return the best result."""
    best = None
    for rollout in range(num_rollouts):
        torch.manual_seed(eval_seed + instance["instance_id"] * num_rollouts + rollout)
        result = evaluate_instance(
            agent, device, instance, horizon, max_relator_length, stochastic=True
        )
        if best is None:
            best = result
        elif result["solved"] and not best["solved"]:
            best = result
        elif result["solved"] and best["solved"]:
            if result["path_length"] < best["path_length"]:
                best = result
    return best


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PPO checkpoint on Miller-Schupp dataset"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to checkpoint .pt file",
    )
    parser.add_argument(
        "--horizon", type=int, default=200,
        help="Maximum steps per episode (default: 200)",
    )
    parser.add_argument(
        "--output", type=str, default="results/ppo_eval.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Evaluate all 1190 instances (default: 1/5 stratified subset = 238)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: 'auto', 'cpu', 'mps', 'cuda'",
    )
    parser.add_argument(
        "--stochastic", action="store_true",
        help="Use stochastic sampling instead of greedy argmax",
    )
    parser.add_argument(
        "--num-rollouts", type=int, default=1,
        help="Number of stochastic rollouts per instance, report best (requires --stochastic)",
    )
    parser.add_argument(
        "--eval-seed", type=int, default=42,
        help="RNG seed for stochastic evaluation reproducibility",
    )
    parser.add_argument(
        "--every-k", type=int, default=5,
        help="Stratified sampling stride (default: 5 → 238 instances, 10 → 147)",
    )
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt["config"]
    global_step = ckpt.get("global_step", "?")
    print(f"Checkpoint from global_step={global_step}, update={ckpt.get('update', '?')}")

    # Load dataset
    instances = load_tagged_dataset()
    if not args.full:
        instances = load_stratified_subset(instances, every_k=args.every_k)
    print(f"Evaluating {len(instances)} instances")

    # Reconstruct agent
    max_relator_length = config.get("max_relator_length", 36)
    obs_dim = max_relator_length * 2
    nodes_counts = config.get("nodes_counts", [256, 256])

    # Create a dummy env to initialize the agent with correct obs/action space dims
    trivial = np.zeros(obs_dim, dtype=np.int8)
    trivial[0] = 1
    trivial[max_relator_length] = 2
    dummy_env = ACEnv(ACEnvConfig(initial_state=trivial, horizon_length=args.horizon))

    # Wrap in a minimal object that has single_observation_space and single_action_space
    class _EnvShim:
        def __init__(self, env):
            self.single_observation_space = env.observation_space
            self.single_action_space = env.action_space

    agent = Agent(_EnvShim(dummy_env), nodes_counts).to(device)
    agent.actor.load_state_dict(ckpt["actor"])
    agent.critic.load_state_dict(ckpt["critic"])
    agent.eval()

    # Run evaluation
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    solved_count = 0
    mode_str = "stochastic" if args.stochastic else "deterministic"
    if args.stochastic and args.num_rollouts > 1:
        mode_str = f"stochastic (best-of-{args.num_rollouts})"
    print(f"Evaluation mode: {mode_str}")

    if args.stochastic:
        torch.manual_seed(args.eval_seed)

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for inst in tqdm(instances, desc=f"Evaluating ({mode_str})", unit=" instances"):
            if args.stochastic and args.num_rollouts > 1:
                result = evaluate_instance_best_of_n(
                    agent, device, inst, args.horizon, max_relator_length,
                    args.num_rollouts, args.eval_seed,
                )
            else:
                if args.stochastic:
                    torch.manual_seed(args.eval_seed + inst["instance_id"])
                result = evaluate_instance(
                    agent, device, inst, args.horizon, max_relator_length,
                    stochastic=args.stochastic,
                )
            writer.writerow(result)
            f.flush()
            if result["solved"]:
                solved_count += 1

    print(f"\nResults: {solved_count}/{len(instances)} solved ({mode_str})")
    print(f"CSV written to: {args.output}")

    # Print summary
    print(f"\nTo plot alongside classical results:")
    print(f"  python -m ac_solver.search.miller_schupp.plot_results \\")
    print(f"      --csvs results/greedy_1M.csv results/bfs_1M.csv {args.output} \\")
    print(f"      --output-dir results/figures/")


if __name__ == "__main__":
    main()
