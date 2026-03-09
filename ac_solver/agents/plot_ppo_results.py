"""
PPO-specific comparison plots that require cross-checkpoint aggregation
or cross-referencing with classical search results.

Plot 1: ppo_scaling.png — Solved count vs environment interactions
Plot 2: ppo_vs_gs_pathlengths.png — Greedy path lengths for PPO-solved vs PPO-unsolved

For per-algorithm figures (bar charts, heatmaps, etc.), use
    python -m ac_solver.search.miller_schupp.plot_results
which accepts PPO CSVs directly (same schema).

Usage:
    python -m ac_solver.agents.plot_ppo_results \
        --ppo-csvs results/ppo_eval_seed1_100000.csv results/ppo_eval_seed1_1000000.csv \
        --classical-csvs results/greedy_1M.csv results/bfs_1M.csv \
        --output-dir results/plots/
"""

import argparse
import csv
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def load_csv(path):
    """Load CSV into list of dicts with type coercion."""
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["instance_id"] = int(row["instance_id"])
            row["solved"] = row["solved"] == "True"
            row["path_length"] = int(row["path_length"]) if row["path_length"] else None
            rows.append(row)
    return rows


def extract_steps_from_filename(filename):
    """Extract step count from filenames like ppo_eval_seed1_1000000.csv or ckpt_1000000."""
    # Try pattern: *_<digits>.csv
    m = re.search(r"_(\d+)\.csv$", filename)
    if m:
        return int(m.group(1))
    return None


def extract_seed_from_filename(filename):
    """Extract seed from filenames like ppo_eval_seed1_1000000.csv."""
    m = re.search(r"seed(\d+)", filename)
    if m:
        return int(m.group(1))
    return None


def fig_scaling(ppo_data, classical_data, subset_ids, output_dir):
    """
    Plot 1: Solved count vs environment interactions.

    ppo_data: dict of {steps: [rows]} aggregated across seeds
    classical_data: dict of {algo_name: [rows]}
    subset_ids: set of instance_ids in the evaluation subset
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("PPO Scaling: Solved Instances vs Training Interactions")

    # Group PPO results by seed and steps
    seed_step_solved = defaultdict(dict)  # {seed: {steps: solved_count}}
    step_solved_all = defaultdict(list)    # {steps: [solved_count_per_seed]}

    for steps, rows_list in sorted(ppo_data.items()):
        for seed, rows in rows_list:
            solved = sum(1 for r in rows if r["solved"])
            seed_step_solved[seed][steps] = solved
            step_solved_all[steps].append(solved)

    steps_sorted = sorted(step_solved_all.keys())

    # Plot per-seed lines (thin)
    for seed, step_dict in sorted(seed_step_solved.items()):
        s = sorted(step_dict.keys())
        v = [step_dict[st] for st in s]
        ax.plot(s, v, "o-", alpha=0.3, linewidth=1, label=f"seed {seed}")

    # Plot mean line (thick)
    if steps_sorted:
        means = [np.mean(step_solved_all[st]) for st in steps_sorted]
        ax.plot(steps_sorted, means, "s-", color="tab:green", linewidth=2.5,
                markersize=8, label="PPO mean", zorder=5)

    # Classical reference lines (filtered to same subset)
    colors = {"greedy": "tab:blue", "bfs": "tab:orange"}
    for algo_name, rows in classical_data.items():
        if subset_ids:
            rows = [r for r in rows if r["instance_id"] in subset_ids]
        solved = sum(1 for r in rows if r["solved"])
        color = colors.get(algo_name, "tab:gray")
        ax.axhline(y=solved, color=color, linestyle="--", linewidth=1.5,
                    label=f"{algo_name} ({solved})")

    ax.set_xscale("log")
    ax.set_xlabel("Environment interactions")
    ax.set_ylabel(f"Instances solved (out of {len(subset_ids) if subset_ids else '?'})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(output_dir, "ppo_scaling.png"), dpi=300)
    fig.savefig(os.path.join(output_dir, "ppo_scaling.pdf"))
    plt.close(fig)
    print(f"Saved ppo_scaling.png")


def fig_pathlengths(ppo_rows, classical_gs_rows, output_dir):
    """
    Plot 2: Greedy search path lengths for PPO-solved vs PPO-unsolved instances.

    Cross-references PPO eval results with greedy search results.
    """
    # Build lookup: instance_id -> greedy path_length
    gs_path = {}
    for r in classical_gs_rows:
        if r["solved"] and r["path_length"] is not None:
            gs_path[r["instance_id"]] = r["path_length"]

    ppo_solved_ids = set(r["instance_id"] for r in ppo_rows if r["solved"])
    ppo_unsolved_ids = set(r["instance_id"] for r in ppo_rows if not r["solved"])

    # Greedy path lengths for PPO-solved instances (that were also GS-solved)
    gs_paths_ppo_solved = [gs_path[iid] for iid in ppo_solved_ids if iid in gs_path]
    gs_paths_ppo_unsolved = [gs_path[iid] for iid in ppo_unsolved_ids if iid in gs_path]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Greedy Search Path Lengths: PPO-Solved vs PPO-Unsolved")

    if gs_paths_ppo_solved:
        ax.hist(gs_paths_ppo_solved, bins=50, alpha=0.6, color="tab:green",
                label=f"PPO-solved ({len(gs_paths_ppo_solved)})", edgecolor="black",
                linewidth=0.5)
    if gs_paths_ppo_unsolved:
        ax.hist(gs_paths_ppo_unsolved, bins=50, alpha=0.6, color="tab:red",
                label=f"PPO-unsolved ({len(gs_paths_ppo_unsolved)})", edgecolor="black",
                linewidth=0.5)

    ax.axvline(x=200, color="black", linestyle="--", linewidth=1.5, label="Horizon T=200")
    ax.set_xlabel("Greedy search path length (AC moves)")
    ax.set_ylabel("Number of instances")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(output_dir, "ppo_vs_gs_pathlengths.png"), dpi=300)
    fig.savefig(os.path.join(output_dir, "ppo_vs_gs_pathlengths.pdf"))
    plt.close(fig)
    print(f"Saved ppo_vs_gs_pathlengths.png")


def main():
    parser = argparse.ArgumentParser(description="PPO comparison plots")
    parser.add_argument(
        "--ppo-csvs", nargs="+", required=True,
        help="PPO evaluation CSV files (from agents/evaluate.py)",
    )
    parser.add_argument(
        "--classical-csvs", nargs="+", default=[],
        help="Classical search CSV files (from search/miller_schupp/evaluate.py)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/plots/",
        help="Directory for output figures",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load PPO results, grouped by (steps, seed)
    ppo_data = defaultdict(list)  # {steps: [(seed, rows), ...]}
    latest_ppo_rows = None
    latest_steps = 0

    for path in args.ppo_csvs:
        rows = load_csv(path)
        basename = os.path.basename(path)
        steps = extract_steps_from_filename(basename)
        seed = extract_seed_from_filename(basename) or 0
        if steps is None:
            print(f"Warning: could not extract step count from {basename}, skipping for scaling plot")
            steps = 0
        ppo_data[steps].append((seed, rows))
        if steps >= latest_steps:
            latest_steps = steps
            latest_ppo_rows = rows

    # Get the subset of instance_ids from the PPO eval
    subset_ids = set()
    for steps, rows_list in ppo_data.items():
        for seed, rows in rows_list:
            subset_ids.update(r["instance_id"] for r in rows)

    # Load classical results
    classical_data = {}
    classical_gs_rows = None
    for path in args.classical_csvs:
        rows = load_csv(path)
        label = os.path.splitext(os.path.basename(path))[0]
        classical_data[label] = rows
        # Detect greedy results for path-length comparison
        algos = set(r.get("algorithm", "") for r in rows)
        if "greedy" in algos:
            classical_gs_rows = rows

    # Plot 1: Scaling curve
    if ppo_data:
        fig_scaling(ppo_data, classical_data, subset_ids, args.output_dir)

    # Plot 2: Path-length comparison (needs greedy results and PPO results)
    if latest_ppo_rows and classical_gs_rows:
        fig_pathlengths(latest_ppo_rows, classical_gs_rows, args.output_dir)
    elif latest_ppo_rows:
        print("Skipping path-length plot: no greedy CSV provided via --classical-csvs")

    print(f"All figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
