"""
Plot PPO training curves from metrics.csv.

Produces two figures:
  1. training_diagnostics.png — 2x3 grid of key training metrics
  2. action_distribution.png — action frequency over training

Usage:
    python -m ac_solver.agents.plot_training_curves \
        --csv out/run_name/metrics.csv \
        --output-dir results/plots/ \
        --smooth 20
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(path):
    """Load metrics.csv into dict of lists."""
    data = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                if key not in data:
                    data[key] = []
                try:
                    data[key].append(float(val))
                except (ValueError, TypeError):
                    data[key].append(float("nan"))
    return data


def smooth(values, window):
    """Rolling mean smoothing."""
    if window <= 1 or len(values) <= window:
        return np.array(values)
    kernel = np.ones(window) / window
    # Pad to avoid shrinking
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def fig_diagnostics(data, output_dir, window):
    """2x3 grid: losses, entropy/KL, clipping, policy sharpness, outcomes, noop."""
    steps = np.array(data["global_step"])

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("PPO Training Diagnostics", fontsize=14, fontweight="bold")

    # (0,0) Losses
    ax = axes[0, 0]
    ax.plot(steps, smooth(data["policy_loss"], window), label="policy_loss", alpha=0.8)
    ax.set_ylabel("policy_loss")
    ax2 = ax.twinx()
    ax2.plot(steps, smooth(data["value_loss"], window), color="tab:orange",
             label="value_loss", alpha=0.8)
    ax2.set_ylabel("value_loss", color="tab:orange")
    ax.set_title("Losses")
    ax.set_xlabel("steps")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    # (0,1) Entropy & KL
    ax = axes[0, 1]
    ax.plot(steps, smooth(data["entropy_loss"], window), label="entropy", alpha=0.8)
    ax.set_ylabel("entropy")
    ax2 = ax.twinx()
    ax2.plot(steps, smooth(data["approx_kl"], window), color="tab:red",
             label="approx_kl", alpha=0.8)
    ax2.set_ylabel("approx_kl", color="tab:red")
    ax.set_title("Entropy & KL")
    ax.set_xlabel("steps")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    # (0,2) Clipping & Explained Variance
    ax = axes[0, 2]
    ax.plot(steps, smooth(data["clipfrac"], window), label="clipfrac", alpha=0.8)
    ax.plot(steps, smooth(data["explained_variance"], window),
            label="explained_var", alpha=0.8)
    ax.set_title("Clipping & Explained Variance")
    ax.set_xlabel("steps")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.1, 1.1)

    # (1,0) Policy Sharpness
    ax = axes[1, 0]
    ax.plot(steps, smooth(data["logit_gap"], window), label="logit_gap", alpha=0.8)
    ax.plot(steps, smooth(data["top1_prob"], window), label="top1_prob", alpha=0.8)
    ax.axhline(y=1.0 / 12, color="gray", linestyle="--", linewidth=1,
               label="uniform (1/12)")
    ax.set_title("Policy Sharpness")
    ax.set_xlabel("steps")
    ax.legend(fontsize=8)

    # (1,1) Outcomes
    ax = axes[1, 1]
    ax.plot(steps, smooth(data["episodes_solved"], window),
            label="solved/update", alpha=0.8)
    ax2 = ax.twinx()
    ax2.plot(steps, data["total_solved"], color="tab:green",
             label="total_solved (cumulative)", alpha=0.8)
    ax2.set_ylabel("cumulative solved", color="tab:green")
    ax.set_title("Episode Outcomes")
    ax.set_xlabel("steps")
    ax.set_ylabel("solved per update")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    # (1,2) Noop Rate
    ax = axes[1, 2]
    ax.plot(steps, smooth(data["noop_rate"], window), label="noop_rate",
            color="tab:purple", alpha=0.8)
    ax.axhline(y=0.5, color="red", linestyle="--", linewidth=1, label="warning (0.5)")
    ax.set_title("No-op Rate")
    ax.set_xlabel("steps")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(output_dir, "training_diagnostics.png"), dpi=300)
    fig.savefig(os.path.join(output_dir, "training_diagnostics.pdf"))
    plt.close(fig)
    print("Saved training_diagnostics.png")


def fig_actions(data, output_dir, window):
    """Stacked area chart of action frequencies."""
    steps = np.array(data["global_step"])

    # Collect action columns
    action_keys = sorted([k for k in data if k.startswith("action_")],
                         key=lambda k: int(k.split("_")[1]))
    if not action_keys:
        print("No action columns found, skipping action distribution plot")
        return

    # Normalize to fractions
    raw = np.array([data[k] for k in action_keys])  # (num_actions, num_steps)
    totals = raw.sum(axis=0)
    totals[totals == 0] = 1  # avoid division by zero
    fractions = raw / totals

    # Smooth each action
    smoothed = np.array([smooth(fractions[i], window) for i in range(len(action_keys))])

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Action Distribution Over Training", fontsize=14, fontweight="bold")

    ax.stackplot(steps, smoothed, labels=action_keys, alpha=0.8)
    ax.axhline(y=1.0 / len(action_keys), color="black", linestyle="--",
               linewidth=1, alpha=0.5, label=f"uniform ({1/len(action_keys):.3f})")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Action fraction")
    ax.set_ylim(0, 1)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)

    fig.tight_layout(rect=[0, 0, 0.85, 0.95])
    fig.savefig(os.path.join(output_dir, "action_distribution.png"), dpi=300)
    fig.savefig(os.path.join(output_dir, "action_distribution.pdf"))
    plt.close(fig)
    print("Saved action_distribution.png")


def print_interpretation(data):
    """Print quick interpretation hints."""
    n = len(data["global_step"])
    if n == 0:
        print("No data points found.")
        return

    last_idx = n - 1
    quarter_idx = n // 4

    print(f"\n--- Quick Interpretation ({n} updates, "
          f"{int(data['global_step'][last_idx]):,} steps) ---")

    # Policy sharpness
    early_top1 = np.mean(data["top1_prob"][:max(1, quarter_idx)])
    late_top1 = np.mean(data["top1_prob"][-(quarter_idx or 1):])
    if late_top1 > early_top1 * 1.5:
        print(f"  top1_prob: {early_top1:.3f} -> {late_top1:.3f} (IMPROVING)")
    elif abs(late_top1 - early_top1) < 0.01:
        print(f"  top1_prob: {early_top1:.3f} -> {late_top1:.3f} (FLAT - may not be learning)")
    else:
        print(f"  top1_prob: {early_top1:.3f} -> {late_top1:.3f}")

    # Solved count
    total_solved = int(data["total_solved"][last_idx])
    print(f"  total_solved: {total_solved}")

    # Noop rate
    late_noop = np.mean(data["noop_rate"][-(quarter_idx or 1):])
    if late_noop > 0.5:
        print(f"  noop_rate: {late_noop:.3f} (HIGH - agent may be stuck)")
    else:
        print(f"  noop_rate: {late_noop:.3f}")

    # Entropy
    late_entropy = np.mean(data["entropy_loss"][-(quarter_idx or 1):])
    uniform_entropy = np.log(12)  # ~2.485 for 12 actions
    if late_entropy > uniform_entropy * 0.95:
        print(f"  entropy: {late_entropy:.3f} (near-uniform {uniform_entropy:.3f} - no learning)")
    elif late_entropy < 0.5:
        print(f"  entropy: {late_entropy:.3f} (LOW - policy may have collapsed)")
    else:
        print(f"  entropy: {late_entropy:.3f} (healthy range)")


def main():
    parser = argparse.ArgumentParser(
        description="Plot PPO training curves from metrics.csv"
    )
    parser.add_argument(
        "--csv", type=str, required=True,
        help="Path to metrics.csv from training run",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/plots/",
        help="Directory for output figures",
    )
    parser.add_argument(
        "--smooth", type=int, default=10,
        help="Rolling mean window size for smoothing (default: 10)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.csv}")
    data = load_metrics(args.csv)

    if not data or "global_step" not in data:
        print("Error: metrics.csv is empty or missing global_step column")
        return

    print(f"Loaded {len(data['global_step'])} data points")
    print(f"Smoothing window: {args.smooth}")

    fig_diagnostics(data, args.output_dir, args.smooth)
    fig_actions(data, args.output_dir, args.smooth)
    print_interpretation(data)

    print(f"\nAll figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
