import numpy as np
from ac_solver.search.greedy import greedy_search


def test_gs_on_AK2():
    # Input setup: n=2 case of Akbulut-Kirby series
    presentation = np.array([1, 1, -2, -2, -2, 0, 0, 1, 2, 1, -2, -1, -2, 0])

    expected_path = [
        # fmt: off
        (-1, 11), (11, 11), (4, 11), (2, 12), (11, 12), (4, 12), (9, 12), (0, 11), (1, 13), (8, 13), (6, 13), (9, 13), (0, 12), (1, 11), (6, 11), (8, 11), (3, 8), (6, 8), (2, 5), (5, 5), (1, 3), (8, 3), (2, 2),
        # fmt: on
    ]

    solved, path, visited = greedy_search(
        presentation=presentation, max_nodes_to_explore=int(1e6), verbose=False
    )

    assert solved
    assert path == expected_path
    assert visited > 0
