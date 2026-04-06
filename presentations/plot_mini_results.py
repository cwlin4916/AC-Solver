"""
Generate plots for the mini reproduction experiment presentation.

Produces:
  1. Bar chart: solve counts across algorithms and checkpoints
  2. Per-n solve rate comparison
  3. Per-(n,|w|) heatmaps for best PPO checkpoints vs classical baselines

Usage:
    python presentations/plot_mini_results.py
"""

import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "results"
MINI_DIR = "results/mini_experiment"
OUTPUT_DIR = "presentations/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_INSTANCES = 147
N_VALUES = list(range(1, 8))
W_VALUES = list(range(1, 8))


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def count_solved(rows):
    return sum(1 for r in rows if r["solved"] == "True")


def solved_by_n(rows):
    """Return dict {n: (solved, total)} for n=1..7."""
    result = {}
    for n in N_VALUES:
        n_rows = [r for r in rows if int(r["n"]) == n]
        n_solved = sum(1 for r in n_rows if r["solved"] == "True")
        result[n] = (n_solved, len(n_rows))
    return result


def solved_by_nw(rows):
    """Return 7x7 array of solve rates for (n, |w|)."""
    grid = np.full((7, 7), np.nan)
    for i, n in enumerate(N_VALUES):
        for j, w in enumerate(W_VALUES):
            cell = [r for r in rows if int(r["n"]) == n and int(r["lenw"]) == w]
            if cell:
                grid[i, j] = sum(1 for r in cell if r["solved"] == "True") / len(cell)
    return grid


# ── Load all results ──────────────────────────────────────────────────────────

greedy = load_csv(f"{RESULTS_DIR}/greedy_subset_k10.csv")
bfs = load_csv(f"{RESULTS_DIR}/bfs_subset_k10.csv")

# PPO results: {run_label: {step_label: {mode: rows}}}
ppo = {}
for run in ["run0_baseline", "run1_reward_fix", "run2_reward_epochs"]:
    ppo[run] = {}
    for step in ["1M", "5M", "10M"]:
        ppo[run][step] = {}
        for mode in ["det", "sto5"]:
            path = f"{MINI_DIR}/{run}_{mode}_{step}.csv"
            if os.path.exists(path):
                ppo[run][step][mode] = load_csv(path)


# ── Plot 1: Solve count bar chart ────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 5))

labels = []
counts = []
colors = []

# Classical baselines
labels.append("Greedy\n(10⁶ nodes)")
counts.append(count_solved(greedy))
colors.append("#2E7D32")

labels.append("BFS\n(10⁶ nodes)")
counts.append(count_solved(bfs))
colors.append("#1565C0")

# PPO runs
for run, run_label, base_color in [
    ("run0_baseline", "Run 0\n(baseline)", "#C62828"),
    ("run1_reward_fix", "Run 1\n(reward fix)", "#E65100"),
    ("run2_reward_epochs", "Run 2\n(rwd+ep4)", "#1565C0"),
]:
    for step in ["1M", "5M", "10M"]:
        for mode, mode_label in [("det", "det"), ("sto5", "sto×5")]:
            if step in ppo.get(run, {}) and mode in ppo[run][step]:
                labels.append(f"{run_label}\n{step} {mode_label}")
                counts.append(count_solved(ppo[run][step][mode]))
                alpha = {"1M": 0.4, "5M": 0.7, "10M": 1.0}[step]
                if mode == "sto5":
                    colors.append(base_color)
                else:
                    colors.append(base_color + "99")  # lighter for det

x = np.arange(len(labels))
bars = ax.bar(x, counts, color=colors, edgecolor="black", linewidth=0.5)

# Add count labels
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            str(count), ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_ylabel("Instances Solved (out of 147)", fontsize=11)
ax.set_title("Mini Experiment: Solve Counts by Algorithm and Checkpoint", fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7, ha="center")
ax.axhline(y=N_INSTANCES, color="gray", linestyle="--", alpha=0.3)
ax.set_ylim(0, max(counts) * 1.15)

# Add a reference line for greedy
ax.axhline(y=count_solved(greedy), color="#2E7D32", linestyle=":", alpha=0.5, label=f"Greedy ceiling ({count_solved(greedy)})")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/mini_solve_counts.pdf", dpi=150)
plt.savefig(f"{OUTPUT_DIR}/mini_solve_counts.png", dpi=150)
print(f"Saved: {OUTPUT_DIR}/mini_solve_counts.{{pdf,png}}")


# ── Plot 2: Solve rate by n ──────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))

# Classical
for data, label, color, ls in [
    (greedy, "Greedy", "#2E7D32", "-"),
    (bfs, "BFS", "#1565C0", "--"),
]:
    by_n = solved_by_n(data)
    rates = [by_n[n][0] / by_n[n][1] * 100 for n in N_VALUES]
    ax.plot(N_VALUES, rates, color=color, linestyle=ls, marker="o", linewidth=2, label=label)

# PPO: best checkpoint for each run (10M sto5)
for run, label, color in [
    ("run0_baseline", "Run 0 (baseline, 10M sto×5)", "#C62828"),
    ("run1_reward_fix", "Run 1 (reward fix, 10M sto×5)", "#E65100"),
    ("run2_reward_epochs", "Run 2 (rwd+ep4, 10M sto×5)", "#1565C0"),
]:
    if "10M" in ppo.get(run, {}) and "sto5" in ppo[run]["10M"]:
        by_n = solved_by_n(ppo[run]["10M"]["sto5"])
        rates = [by_n[n][0] / by_n[n][1] * 100 for n in N_VALUES]
        ax.plot(N_VALUES, rates, color=color, linestyle="-.", marker="s", linewidth=2, label=label)

ax.set_xlabel("n (first relator complexity)", fontsize=11)
ax.set_ylabel("Solve Rate (%)", fontsize=11)
ax.set_title("Solve Rate by n: Classical vs PPO on Subset (147 instances)", fontsize=12)
ax.set_xticks(N_VALUES)
ax.set_ylim(-5, 105)
ax.legend(fontsize=9, loc="upper right")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/mini_solve_rate_by_n.pdf", dpi=150)
plt.savefig(f"{OUTPUT_DIR}/mini_solve_rate_by_n.png", dpi=150)
print(f"Saved: {OUTPUT_DIR}/mini_solve_rate_by_n.{{pdf,png}}")


# ── Plot 3: Heatmaps ─────────────────────────────────────────────────────────

datasets = [
    ("Greedy", greedy),
    ("BFS", bfs),
]
# Add best PPO checkpoints
for run, label in [
    ("run0_baseline", "Run 0 (baseline)\n10M sto×5"),
    ("run1_reward_fix", "Run 1 (reward fix)\n10M sto×5"),
    ("run2_reward_epochs", "Run 2 (rwd+ep4)\n10M sto×5"),
]:
    if "10M" in ppo.get(run, {}) and "sto5" in ppo[run]["10M"]:
        datasets.append((label, ppo[run]["10M"]["sto5"]))

ncols = len(datasets)
fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 5))
if ncols == 1:
    axes = [axes]

for ax, (title, data) in zip(axes, datasets):
    grid = solved_by_nw(data)
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    # Annotate cells
    for i in range(7):
        for j in range(7):
            if not np.isnan(grid[i, j]):
                text_color = "white" if grid[i, j] < 0.3 or grid[i, j] > 0.8 else "black"
                ax.text(j, i, f"{grid[i,j]:.0%}", ha="center", va="center",
                        fontsize=7, color=text_color, fontweight="bold")

    ax.set_xticks(range(7))
    ax.set_xticklabels(W_VALUES)
    ax.set_yticks(range(7))
    ax.set_yticklabels(N_VALUES)
    ax.set_xlabel("|w|")
    ax.set_ylabel("n")
    ax.set_title(title, fontsize=10)

fig.suptitle("Solve Rate by (n, |w|) on 147-Instance Subset", fontsize=13, y=1.02)
fig.colorbar(im, ax=axes, shrink=0.8, label="Solve Rate")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/mini_heatmaps.pdf", dpi=150, bbox_inches="tight")
plt.savefig(f"{OUTPUT_DIR}/mini_heatmaps.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT_DIR}/mini_heatmaps.{{pdf,png}}")


# ── Plot 4: Learning curve — PPO solve count vs checkpoint ────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

steps = [1, 5, 10]  # in millions

for run, label, color, marker in [
    ("run0_baseline", "Run 0 (baseline)", "#C62828", "o"),
    ("run1_reward_fix", "Run 1 (reward fix)", "#E65100", "s"),
    ("run2_reward_epochs", "Run 2 (rwd+ep4)", "#1565C0", "D"),
]:
    for mode, mode_label, ls in [("det", "deterministic", "--"), ("sto5", "stochastic ×5", "-")]:
        counts_by_step = []
        valid_steps = []
        for step_m, step_label in zip(steps, ["1M", "5M", "10M"]):
            if step_label in ppo.get(run, {}) and mode in ppo[run][step_label]:
                counts_by_step.append(count_solved(ppo[run][step_label][mode]))
                valid_steps.append(step_m)
        if valid_steps:
            ax_target = ax1 if mode == "det" else ax2
            ax_target.plot(valid_steps, counts_by_step, color=color, linestyle=ls,
                          marker=marker, linewidth=2, markersize=8,
                          label=f"{label}")

for ax, title in [(ax1, "Deterministic (argmax)"), (ax2, "Stochastic (best-of-5)")]:
    ax.axhline(y=count_solved(greedy), color="#2E7D32", linestyle=":", alpha=0.5,
               label=f"Greedy ceiling ({count_solved(greedy)})")
    ax.axhline(y=count_solved(bfs), color="#1565C0", linestyle=":", alpha=0.5,
               label=f"BFS ceiling ({count_solved(bfs)})")
    ax.set_xlabel("Training Steps (millions)", fontsize=11)
    ax.set_ylabel("Instances Solved (out of 147)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_xticks(steps)
    ax.set_ylim(-2, max(count_solved(greedy) + 10, 20))
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

fig.suptitle("PPO Learning Curves: Run 0 vs Run 1", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/mini_learning_curves.pdf", dpi=150)
plt.savefig(f"{OUTPUT_DIR}/mini_learning_curves.png", dpi=150)
print(f"Saved: {OUTPUT_DIR}/mini_learning_curves.{{pdf,png}}")


# ── Print summary table ──────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("EVALUATION SUMMARY")
print("=" * 70)
print(f"{'Algorithm':<35} {'Solved':>8} {'Rate':>8}")
print("-" * 55)
print(f"{'Greedy (10^6 nodes)':<35} {count_solved(greedy):>5}/147 {count_solved(greedy)/147*100:>6.1f}%")
print(f"{'BFS (10^6 nodes)':<35} {count_solved(bfs):>5}/147 {count_solved(bfs)/147*100:>6.1f}%")
print("-" * 55)

for run, run_label in [("run0_baseline", "Run 0 (baseline)"), ("run1_reward_fix", "Run 1 (reward fix)"), ("run2_reward_epochs", "Run 2 (rwd+ep4)")]:
    for step in ["1M", "5M", "10M"]:
        for mode, mode_label in [("det", "det"), ("sto5", "sto×5")]:
            if step in ppo.get(run, {}) and mode in ppo[run][step]:
                s = count_solved(ppo[run][step][mode])
                print(f"{run_label + ' ' + step + ' ' + mode_label:<35} {s:>5}/147 {s/147*100:>6.1f}%")
    print()
