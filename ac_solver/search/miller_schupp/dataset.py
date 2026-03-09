"""
Tagged dataset loader for Miller-Schupp presentations.

Provides instance-level metadata (instance_id, n, lenw) by regenerating
presentations from the same parameters used to create all_presentations.txt.
"""

from ac_solver.search.miller_schupp.miller_schupp import (
    generate_miller_schupp_presentations,
)


def load_tagged_dataset(min_n=1, max_n=7, min_w_len=1, max_w_len=7):
    """
    Generate all Miller-Schupp presentations with (n, lenw) tags and stable IDs.

    Returns list of dicts with keys: instance_id, n, lenw, presentation.
    Iteration order: n ascending, then lenw ascending, then generation order.
    """
    instances = []
    instance_id = 0
    for n in range(min_n, max_n + 1):
        lenw_to_pres = generate_miller_schupp_presentations(n, max_w_len)
        for lenw in range(min_w_len, max_w_len + 1):
            if lenw not in lenw_to_pres:
                continue
            for pres in lenw_to_pres[lenw]:
                instances.append(
                    {
                        "instance_id": instance_id,
                        "n": n,
                        "lenw": lenw,
                        "presentation": pres,
                    }
                )
                instance_id += 1
    return instances


def load_stratified_subset(instances, every_k=5):
    """
    Take every k-th instance within each (n, lenw) cell for stratified sampling.

    Preserves proportional representation across all (n, lenw) cells.
    """
    from collections import defaultdict

    cells = defaultdict(list)
    for inst in instances:
        cells[(inst["n"], inst["lenw"])].append(inst)

    subset = []
    for key in sorted(cells.keys()):
        cell_instances = cells[key]
        subset.extend(cell_instances[::every_k])
    return subset
