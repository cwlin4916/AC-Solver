import csv
import os
import tempfile

import numpy as np

from ac_solver.search.miller_schupp.dataset import (
    load_tagged_dataset,
    load_stratified_subset,
)
from ac_solver.search.miller_schupp.evaluate import _run_single_instance


def test_dataset_loader_count():
    instances = load_tagged_dataset()
    assert len(instances) == 1190, f"Expected 1190, got {len(instances)}"
    ns = set(i["n"] for i in instances)
    assert ns == {1, 2, 3, 4, 5, 6, 7}
    for inst in instances:
        assert 1 <= inst["lenw"] <= 7
        assert isinstance(inst["presentation"], list)


def test_stratified_subset():
    instances = load_tagged_dataset()
    subset = load_stratified_subset(instances, every_k=5)
    assert len(subset) < len(instances)
    assert len(subset) > 0
    # Every (n, lenw) cell should be represented
    full_cells = set((i["n"], i["lenw"]) for i in instances)
    subset_cells = set((i["n"], i["lenw"]) for i in subset)
    assert subset_cells == full_cells


def test_evaluate_single_instance_greedy():
    instances = load_tagged_dataset()
    # Use first instance — should be a small, easy one (n=1, lenw=1 or 2)
    inst = instances[0]
    result = _run_single_instance((inst, "greedy", 10000, True, True))
    assert result["instance_id"] == inst["instance_id"]
    assert result["algorithm"] == "greedy"
    assert isinstance(result["solved"], bool)
    assert isinstance(result["visited_nodes"], int)
    assert isinstance(result["wallclock_seconds"], float)


def test_evaluate_single_instance_bfs():
    instances = load_tagged_dataset()
    inst = instances[0]
    result = _run_single_instance((inst, "bfs", 10000, True, True))
    assert result["algorithm"] == "bfs"
    assert isinstance(result["solved"], bool)


def test_bfs_fixed_vs_legacy():
    # AK(2) should be solvable under both flag settings
    presentation = list(np.array([1, 1, -2, -2, -2, 0, 0, 1, 2, 1, -2, -1, -2, 0]))
    inst = {"instance_id": 9999, "n": 2, "lenw": 5, "presentation": presentation}

    result_fixed = _run_single_instance((inst, "bfs", int(1e6), True, True))
    result_legacy = _run_single_instance((inst, "bfs", int(1e6), False, True))

    assert result_fixed["solved"], "BFS fixed should solve AK(2)"
    assert result_legacy["solved"], "BFS legacy should solve AK(2)"
