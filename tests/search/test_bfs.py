import numpy as np
from ac_solver.search.breadth_first import bfs


def test_bfs_on_AK2_legacy():
    # Tests legacy (buggy) length computation for backwards compatibility
    presentation = np.array([1, 1, -2, -2, -2, 0, 0, 1, 2, 1, -2, -1, -2, 0])

    expected_path = [
        # fmt: off
        (-1, 11), (4, 11), (11, 11), (2, 12), (4, 12), (11, 12), (9, 12), (0, 11), (5, 11), (7, 11), (3, 13), (11, 13), (9, 13), (2, 12), (8, 12), (9, 12), (3, 7), (0, 5), (0, 3), (3, 2),
        # fmt: on
    ]

    solved, path, visited = bfs(
        presentation=presentation,
        max_nodes_to_explore=int(1e6),
        verbose=False,
        use_fixed_lengths=False,
    )

    assert solved
    assert path == expected_path
    assert visited > 0


def test_bfs_on_AK2_fixed():
    # Tests fixed length computation on AK(2) — should still solve it
    presentation = np.array([1, 1, -2, -2, -2, 0, 0, 1, 2, 1, -2, -1, -2, 0])

    solved, path, visited = bfs(
        presentation=presentation,
        max_nodes_to_explore=int(1e6),
        verbose=False,
        use_fixed_lengths=True,
    )

    assert solved, "BFS with fixed lengths should solve AK(2)"
    assert path[-1][1] == 2, "Final presentation should be trivial (length 2)"
    assert visited > 0


def test_bfs_max_nodes_reached():
    # Test on AK(2) but with maximum of 10 nodes.
    presentation = np.array([1, 1, -2, -2, -2, 0, 0, 1, 2, 1, -2, -1, -2, 0])
    solved, path, visited = bfs(
        presentation=presentation, max_nodes_to_explore=10, verbose=False
    )
    assert not solved
    assert path is None
    assert visited > 0
