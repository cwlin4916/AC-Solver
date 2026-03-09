"""
Plotting script for classical search reproduction results.

Reads CSV files produced by evaluate.py and generates 5 multi-panel figures
plus JSON split files and a discrepancy report.

Usage:
    python -m ac_solver.search.miller_schupp.plot_results \
        --csvs results/greedy_1M.csv results/bfs_1M.csv \
        --output-dir results/figures/
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def load_csv(path):
    """Load a CSV file into a list of dicts with type coercion."""
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["instance_id"] = int(row["instance_id"])
            row["n"] = int(row["n"])
            row["lenw"] = int(row["lenw"])
            row["max_nodes"] = int(row["max_nodes"])
            row["solved"] = row["solved"] == "True"
            row["visited_nodes"] = int(row["visited_nodes"]) if row["visited_nodes"] else -1
            row["path_length"] = int(row["path_length"]) if row["path_length"] else None
            row["initial_total_length"] = (
                int(row["initial_total_length"]) if row["initial_total_length"] else None
            )
            row["max_intermediate_length"] = (
                int(row["max_intermediate_length"]) if row["max_intermediate_length"] else None
            )
            row["length_increase"] = (
                int(row["length_increase"]) if row["length_increase"] else None
            )
            row["wallclock_seconds"] = float(row["wallclock_seconds"])
            rows.append(row)
    return rows


def get_algo_name(rows):
    """Extract algorithm name from rows."""
    algos = set(r["algorithm"] for r in rows)
    if len(algos) == 1:
        return algos.pop()
    return "mixed"


def fig1_solve_rate(datasets, output_dir):
    """Figure 1: Solve-rate overview — 2-panel bar chart by n and lenw."""
    fig, (ax_n, ax_w) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("AC Trivialization of Miller-Schupp Presentations: Solve Rate (1M node cap)")

    ns = sorted(set(r["n"] for ds in datasets.values() for r in ds))
    ws = sorted(set(r["lenw"] for ds in datasets.values() for r in ds))
    colors = {"greedy": "tab:blue", "bfs": "tab:orange", "ppo": "tab:green"}

    width = 0.8 / len(datasets)
    for idx, (label, rows) in enumerate(datasets.items()):
        algo = get_algo_name(rows)
        color = colors.get(algo, f"C{idx}")

        solved_by_n = defaultdict(int)
        for r in rows:
            if r["solved"]:
                solved_by_n[r["n"]] += 1
        x = np.arange(len(ns))
        vals = [solved_by_n.get(n, 0) for n in ns]
        bars = ax_n.bar(x + idx * width - 0.4 + width / 2, vals, width, label=label, color=color)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax_n.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                          str(v), ha="center", va="bottom", fontsize=7)

        solved_by_w = defaultdict(int)
        for r in rows:
            if r["solved"]:
                solved_by_w[r["lenw"]] += 1
        vals_w = [solved_by_w.get(w, 0) for w in ws]
        bars_w = ax_w.bar(np.arange(len(ws)) + idx * width - 0.4 + width / 2, vals_w, width,
                          label=label, color=color)
        for bar, v in zip(bars_w, vals_w):
            if v > 0:
                ax_w.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                          str(v), ha="center", va="bottom", fontsize=7)

    ax_n.set_xlabel("Miller-Schupp parameter n")
    ax_n.set_ylabel("Instances trivialized")
    ax_n.set_title("Instances Trivialized by n")
    ax_n.set_xticks(np.arange(len(ns)))
    ax_n.set_xticklabels(ns)
    ax_n.legend()

    ax_w.set_xlabel("Word length |w|")
    ax_w.set_ylabel("Instances trivialized")
    ax_w.set_title("Instances Trivialized by |w|")
    ax_w.set_xticks(np.arange(len(ws)))
    ax_w.set_xticklabels(ws)
    ax_w.legend()

    totals = []
    for label, rows in datasets.items():
        total_solved = sum(1 for r in rows if r["solved"])
        total_all = len(rows)
        totals.append(f"{label}: {total_solved}/{total_all}")
    fig.text(0.5, 0.01, " | ".join(totals) + " (targets: GS≈533/1190, BFS≈278/1190)",
             ha="center", fontsize=9)

    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(os.path.join(output_dir, "ac_trivialization_solve_rate.pdf"))
    fig.savefig(os.path.join(output_dir, "ac_trivialization_solve_rate.png"), dpi=300)
    plt.close(fig)


def fig2_heatmap(datasets, output_dir):
    """Figure 2: Solve-rate heatmap — 2-panel, faceted by algorithm."""
    n_panels = len(datasets)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5), sharey=True, squeeze=False)
    axes = axes[0]
    fig.suptitle("AC Trivialization Rate by Presentation Complexity (n, |w|)")

    for idx, (label, rows) in enumerate(datasets.items()):
        ax = axes[idx]
        ns = sorted(set(r["n"] for r in rows))
        ws = sorted(set(r["lenw"] for r in rows))

        total_grid = defaultdict(int)
        solved_grid = defaultdict(int)
        for r in rows:
            total_grid[(r["n"], r["lenw"])] += 1
            if r["solved"]:
                solved_grid[(r["n"], r["lenw"])] += 1

        matrix = np.zeros((len(ns), len(ws)))
        annots = [['' for _ in ws] for _ in ns]
        for i, n in enumerate(ns):
            for j, w in enumerate(ws):
                total = total_grid.get((n, w), 0)
                solved = solved_grid.get((n, w), 0)
                matrix[i, j] = solved / total if total > 0 else 0
                annots[i][j] = f"{solved}/{total}" if total > 0 else ""

        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(np.arange(len(ws)))
        ax.set_xticklabels(ws)
        ax.set_yticks(np.arange(len(ns)))
        ax.set_yticklabels(ns)
        ax.set_xlabel("Word length |w|")
        if idx == 0:
            ax.set_ylabel("Parameter n")
        ax.set_title(label)

        for i in range(len(ns)):
            for j in range(len(ws)):
                ax.text(j, i, annots[i][j], ha="center", va="center", fontsize=7,
                        color="black" if matrix[i, j] > 0.4 else "white")

        fig.colorbar(im, ax=ax, label="Fraction solved", shrink=0.8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(output_dir, "ac_trivialization_heatmap_n_vs_w.pdf"))
    fig.savefig(os.path.join(output_dir, "ac_trivialization_heatmap_n_vs_w.png"), dpi=300)
    plt.close(fig)


def fig3_length_increase(datasets, output_dir):
    """Figure 3: Length-increase analysis — 2x2 faceted (histograms + boxplots)."""
    labels = list(datasets.keys())
    n_cols = len(labels)
    fig, axes = plt.subplots(2, n_cols, figsize=(6 * n_cols, 10), squeeze=False)
    fig.suptitle("Presentation Length Increase During AC Trivialization")

    for col, label in enumerate(labels):
        rows = datasets[label]
        solved = [r for r in rows if r["solved"] and r["length_increase"] is not None]
        increases = [r["length_increase"] for r in solved]

        ax_hist = axes[0, col]
        if increases:
            max_inc = max(increases)
            ax_hist.hist(increases, bins=range(0, max_inc + 2), alpha=0.7, color="tab:blue",
                         edgecolor="black", linewidth=0.5)
            ax_hist.axvline(x=5, color="red", linestyle="--", linewidth=1, label="Paper max=5")
            ax_hist.legend()
        ax_hist.set_xlabel("Max length increase")
        ax_hist.set_ylabel("Number of instances")
        ax_hist.set_title(f"{label}: {len(solved)} trivialized")

        ax_box = axes[1, col]
        ns = sorted(set(r["n"] for r in solved)) if solved else []
        box_data = [[r["length_increase"] for r in solved if r["n"] == n] for n in ns]
        box_data = [d for d in box_data if d]
        if box_data:
            ax_box.boxplot(box_data, labels=[str(n) for n in ns if any(r["n"] == n for r in solved)])
        ax_box.set_xlabel("Parameter n")
        ax_box.set_ylabel("Max presentation length increase")
        ax_box.set_title(f"{label}: by n")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(output_dir, "ac_path_length_increase.pdf"))
    fig.savefig(os.path.join(output_dir, "ac_path_length_increase.png"), dpi=300)
    plt.close(fig)


def fig4_path_length(datasets, output_dir):
    """Figure 4: Path-length analysis — 2x2 faceted (histograms + scatter)."""
    labels = list(datasets.keys())
    n_cols = len(labels)
    fig, axes = plt.subplots(2, n_cols, figsize=(6 * n_cols, 10), squeeze=False)
    fig.suptitle("Number of AC Moves to Trivialize")

    for col, label in enumerate(labels):
        rows = datasets[label]
        solved = [r for r in rows if r["solved"] and r["path_length"] is not None]
        path_lengths = [r["path_length"] for r in solved]

        ax_hist = axes[0, col]
        if path_lengths:
            ax_hist.hist(path_lengths, bins=50, alpha=0.7, color="tab:green",
                         edgecolor="black", linewidth=0.5)
            ax_hist.axvline(x=344, color="red", linestyle="--", linewidth=1, label="Paper max=344")
            ax_hist.legend()
        ax_hist.set_xlabel("AC moves to trivialize")
        ax_hist.set_ylabel("Number of instances")
        ax_hist.set_title(f"{label}: {len(solved)} trivialized")

        ax_scatter = axes[1, col]
        if solved:
            ns_vals = [r["n"] for r in solved]
            pl_vals = [r["path_length"] for r in solved]
            li_vals = [r["length_increase"] for r in solved]
            sc = ax_scatter.scatter(pl_vals, li_vals, c=ns_vals, cmap="viridis",
                                    alpha=0.6, s=15, edgecolors="none")
            fig.colorbar(sc, ax=ax_scatter, label="Parameter n", shrink=0.8)
        ax_scatter.set_xlabel("Path length")
        ax_scatter.set_ylabel("Max length increase")
        ax_scatter.set_title(f"{label}: AC moves vs. length increase")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(output_dir, "ac_path_steps_to_trivialization.pdf"))
    fig.savefig(os.path.join(output_dir, "ac_path_steps_to_trivialization.png"), dpi=300)
    plt.close(fig)


def fig5_effort(datasets, output_dir):
    """Figure 5: Computational effort — visited nodes histogram + wallclock scatter."""
    fig, (ax_hist, ax_scatter) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Search Effort: Nodes Explored vs. Wallclock Time")

    colors = {"greedy": "tab:blue", "bfs": "tab:orange", "ppo": "tab:green"}

    for label, rows in datasets.items():
        algo = get_algo_name(rows)
        color = colors.get(algo, "tab:gray")
        visited = [r["visited_nodes"] for r in rows if r["visited_nodes"] > 0]
        wall = [r["wallclock_seconds"] for r in rows if r["visited_nodes"] > 0]

        if visited:
            ax_hist.hist(visited, bins=50, alpha=0.5, label=label, color=color, log=False)
            ax_scatter.scatter(visited, wall, alpha=0.3, s=10, label=label, color=color,
                               edgecolors="none")

    ax_hist.set_xscale("log")
    ax_hist.axvline(x=1_000_000, color="red", linestyle="--", linewidth=1, label="1M cap")
    ax_hist.set_xlabel("Visited nodes (log scale)")
    ax_hist.set_ylabel("Number of instances")
    ax_hist.set_title("Nodes explored per instance")
    ax_hist.legend()

    ax_scatter.set_xscale("log")
    ax_scatter.set_xlabel("Visited nodes (log scale)")
    ax_scatter.set_ylabel("Wallclock seconds")
    ax_scatter.set_title("Nodes explored vs. wallclock time")
    ax_scatter.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(output_dir, "ac_search_computational_effort.pdf"))
    fig.savefig(os.path.join(output_dir, "ac_search_computational_effort.png"), dpi=300)
    plt.close(fig)


def write_splits(datasets, output_dir):
    """Write gs_solved.json and gs_unsolved.json from greedy results."""
    for label, rows in datasets.items():
        algo = get_algo_name(rows)
        if algo == "greedy":
            solved_ids = [r["instance_id"] for r in rows if r["solved"]]
            unsolved_ids = [r["instance_id"] for r in rows if not r["solved"]]
            splits_dir = os.path.join(os.path.dirname(output_dir.rstrip("/")), "splits")
            os.makedirs(splits_dir, exist_ok=True)
            with open(os.path.join(splits_dir, "gs_solved.json"), "w") as f:
                json.dump(solved_ids, f)
            with open(os.path.join(splits_dir, "gs_unsolved.json"), "w") as f:
                json.dump(unsolved_ids, f)
            print(f"Splits written: {len(solved_ids)} solved, {len(unsolved_ids)} unsolved")
            break


def write_discrepancy_report(datasets, output_dir):
    """Write a markdown table comparing reproduced vs target counts."""
    targets = {"greedy": 533, "bfs": 278}
    lines = ["# Discrepancy Report\n", "| Dataset | Algorithm | Solved | Target | Match |",
             "|---|---|---|---|---|"]
    for label, rows in datasets.items():
        algo = get_algo_name(rows)
        solved = sum(1 for r in rows if r["solved"])
        total = len(rows)
        target = targets.get(algo, "?")
        match = "YES" if solved == target else f"NO (diff={solved - target if isinstance(target, int) else '?'})"
        lines.append(f"| {label} ({total} instances) | {algo} | {solved} | {target} | {match} |")

    report_path = os.path.join(output_dir, "discrepancy_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Discrepancy report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot classical search reproduction results")
    parser.add_argument(
        "--csvs", nargs="+", required=True,
        help="CSV files from evaluate.py (one per algorithm/config)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/figures/",
        help="Directory for output figures",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    datasets = {}
    for path in args.csvs:
        rows = load_csv(path)
        label = os.path.splitext(os.path.basename(path))[0]
        datasets[label] = rows
        algo = get_algo_name(rows)
        solved = sum(1 for r in rows if r["solved"])
        print(f"Loaded {path}: {len(rows)} instances, {solved} solved ({algo})")

    fig1_solve_rate(datasets, args.output_dir)
    fig2_heatmap(datasets, args.output_dir)
    fig3_length_increase(datasets, args.output_dir)
    fig4_path_length(datasets, args.output_dir)
    fig5_effort(datasets, args.output_dir)

    write_splits(datasets, args.output_dir)
    write_discrepancy_report(datasets, args.output_dir)

    print(f"All figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
