"""
Evaluation harness for classical search on Miller-Schupp presentations.

Runs greedy or BFS search over the dataset with multiprocessing support,
writing per-instance metrics to a resumable CSV.

Usage:
    python -m ac_solver.search.miller_schupp.evaluate \
        --search-fn greedy --max-nodes 1000000 --workers 4 --output results/greedy_1M.csv

    python -m ac_solver.search.miller_schupp.evaluate \
        --search-fn bfs --max-nodes 1000000 --workers 4 --output results/bfs_1M.csv

    Add --full to run all 1190 instances (default: 1/5 stratified subset).
    Add --resume to skip already-completed instances.
    For BFS, add --no-use-fixed-lengths to use legacy (buggy) length computation.
"""

import argparse
import csv
import os
import time
from multiprocessing import Pool

from tqdm import tqdm

from ac_solver.search.miller_schupp.dataset import (
    load_tagged_dataset,
    load_stratified_subset,
)

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


def _run_single_instance(args):
    """Worker function for multiprocessing. Must be top-level for pickling."""
    instance, search_fn_name, max_nodes, use_fixed_lengths, cyclic_reduce = args

    if search_fn_name == "greedy":
        from ac_solver.search.greedy import greedy_search

        search_fn = greedy_search
    else:
        from ac_solver.search.breadth_first import bfs

        search_fn = bfs

    presentation = instance["presentation"]
    kwargs = {
        "presentation": presentation,
        "max_nodes_to_explore": max_nodes,
        "verbose": False,
        "cyclically_reduce_after_moves": cyclic_reduce,
        "show_progress": False,
    }
    if search_fn_name == "bfs":
        kwargs["use_fixed_lengths"] = use_fixed_lengths

    t0 = time.time()
    try:
        solved, path, visited_nodes = search_fn(**kwargs)
    except MemoryError:
        return {
            "instance_id": instance["instance_id"],
            "n": instance["n"],
            "lenw": instance["lenw"],
            "algorithm": search_fn_name,
            "max_nodes": max_nodes,
            "solved": False,
            "visited_nodes": -1,
            "path_length": "",
            "initial_total_length": "",
            "max_intermediate_length": "",
            "length_increase": "",
            "wallclock_seconds": round(time.time() - t0, 3),
        }
    elapsed = round(time.time() - t0, 3)

    if solved and path:
        initial_length = path[0][1]
        max_intermediate = max(l for _, l in path)
        result = {
            "instance_id": instance["instance_id"],
            "n": instance["n"],
            "lenw": instance["lenw"],
            "algorithm": search_fn_name,
            "max_nodes": max_nodes,
            "solved": True,
            "visited_nodes": visited_nodes,
            "path_length": len(path) - 1,
            "initial_total_length": initial_length,
            "max_intermediate_length": max_intermediate,
            "length_increase": max_intermediate - initial_length,
            "wallclock_seconds": elapsed,
        }
    else:
        initial_length = ""
        if path:
            initial_length = path[0][1]
        result = {
            "instance_id": instance["instance_id"],
            "n": instance["n"],
            "lenw": instance["lenw"],
            "algorithm": search_fn_name,
            "max_nodes": max_nodes,
            "solved": False,
            "visited_nodes": visited_nodes,
            "path_length": "",
            "initial_total_length": initial_length,
            "max_intermediate_length": "",
            "length_increase": "",
            "wallclock_seconds": elapsed,
        }
    return result


def load_completed_ids(csv_path):
    """Read existing CSV and return set of completed instance_ids."""
    completed = set()
    if os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add(int(row["instance_id"]))
    return completed


def run_evaluation(
    search_fn_name,
    max_nodes,
    output_path,
    workers,
    resume,
    full,
    use_fixed_lengths,
    cyclic_reduce=True,
):
    instances = load_tagged_dataset()
    if not full:
        instances = load_stratified_subset(instances, every_k=5)

    print(f"Dataset: {len(instances)} instances ({'full' if full else '1/5 subset'})")
    print(f"Algorithm: {search_fn_name}, max_nodes: {max_nodes}, workers: {workers}")

    completed_ids = set()
    if resume:
        completed_ids = load_completed_ids(output_path)
        if completed_ids:
            print(f"Resuming: {len(completed_ids)} instances already completed")

    remaining = [i for i in instances if i["instance_id"] not in completed_ids]
    if not remaining:
        print("All instances already completed.")
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    write_header = not os.path.exists(output_path) or not completed_ids
    mode = "a" if completed_ids else "w"

    with open(output_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        work_args = [
            (inst, search_fn_name, max_nodes, use_fixed_lengths, cyclic_reduce) for inst in remaining
        ]

        solved_count = 0
        if workers > 1:
            with Pool(processes=workers) as pool:
                for result in tqdm(
                    pool.imap_unordered(_run_single_instance, work_args),
                    total=len(remaining),
                    desc=f"{search_fn_name} search",
                    unit=" instances",
                ):
                    writer.writerow(result)
                    f.flush()
                    if result["solved"]:
                        solved_count += 1
        else:
            for args in tqdm(work_args, desc=f"{search_fn_name} search", unit=" instances"):
                result = _run_single_instance(args)
                writer.writerow(result)
                f.flush()
                if result["solved"]:
                    solved_count += 1

    total = len(remaining)
    print(f"Done: {solved_count}/{total} solved ({total - solved_count} unsolved)")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate classical search on Miller-Schupp presentations"
    )
    parser.add_argument(
        "--search-fn",
        type=str,
        required=True,
        choices=["greedy", "bfs"],
        help="Search algorithm to use",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=1_000_000,
        help="Maximum nodes to explore per instance (default: 1000000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/classical_reproduction.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing CSV, skipping completed instances",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run all 1190 instances (default: 1/5 stratified subset)",
    )
    parser.add_argument(
        "--no-use-fixed-lengths",
        action="store_true",
        help="BFS only: use legacy (buggy) length computation for comparison",
    )
    parser.add_argument(
        "--no-cyclic-reduce",
        action="store_true",
        help="Disable cyclic reduction after moves (default: ON, matching PPO's ACEnv)",
    )
    args = parser.parse_args()

    use_fixed_lengths = not args.no_use_fixed_lengths
    cyclic_reduce = not args.no_cyclic_reduce

    run_evaluation(
        search_fn_name=args.search_fn,
        max_nodes=args.max_nodes,
        output_path=args.output,
        workers=args.workers,
        resume=args.resume,
        full=args.full,
        use_fixed_lengths=use_fixed_lengths,
        cyclic_reduce=cyclic_reduce,
    )


if __name__ == "__main__":
    main()
