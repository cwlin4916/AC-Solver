"""
Implementation of greedy search for AC graph.

Example:
Trivialize Akbulut-Kirby series n=2 case "AK(2)" through greedy search as 
python greedy.py
"""

import numpy as np
import heapq
from tqdm import tqdm
from ac_solver.envs.utils import is_presentation_trivial
from ac_solver.envs.ac_moves import ACMove


def greedy_search(
    presentation,
    max_nodes_to_explore=10000,
    verbose=False,
    cyclically_reduce_after_moves=False,
    show_progress=True,
):
    """
    Performs a greedy search on an AC graph starting from the given presentation.

    Parameters:
        presentation (np.ndarray): Initial presentation as a NumPy array.
        max_nodes_to_explore (int, optional): Max nodes to explore before termination (default: 10000).
        verbose (bool, optional): Print updates when shorter presentations are found (default: False).
        cyclically_reduce_after_moves (bool, optional): Apply cyclic reduction after each move (default: False).

    Returns:
        tuple: (is_search_successful, path, visited_nodes)
            - is_search_successful (bool): Whether a trivial state was found.
            - path (list of tuple): Sequence of (action, presentation_length).
            - visited_nodes (int): Number of unique states explored.
    """

    presentation = np.array(
        presentation, dtype=np.int8
    )  # so that input may be a list or a tuple

    # set initial state for search and maximum relator length allowed
    # if we encounter a presentation with a relator of length greater than max_relator_length,
    initial_state = np.array(
        presentation, dtype=np.int8
    )  # so that input may be a list or a tuple
    max_relator_length = len(presentation) // 2

    # we keep track of word lengths
    first_word_length = np.count_nonzero(presentation[:max_relator_length])
    second_word_length = np.count_nonzero(presentation[max_relator_length:])
    word_lengths = [first_word_length, second_word_length]
    total_initial_length = sum(word_lengths)

    # add to a priority queue, keeping track of path length to initial state
    path_length = 0
    to_explore = [
        (
            total_initial_length,
            path_length,
            tuple(initial_state),
            tuple(word_lengths),
            [(-1, total_initial_length)],
        )
    ]
    heapq.heapify(to_explore)

    # a set containing states that have already been seen
    tree_nodes = set()
    tree_nodes.add(tuple(initial_state))
    min_length = total_initial_length

    pbar = tqdm(total=max_nodes_to_explore, desc="Searching", unit=" nodes", disable=not show_progress)

    while to_explore:
        _, path_length, state_tuple, word_lengths, path = heapq.heappop(to_explore)
        state = np.array(state_tuple, dtype=np.int8)  # convert tuple to state
        word_lengths = list(word_lengths)

        for action in range(0, 12):
            new_state, new_lengths = ACMove(
                action,
                state,
                max_relator_length,
                word_lengths,
                cyclical=cyclically_reduce_after_moves,
            )
            state_tup, new_length = tuple(new_state), sum(new_lengths)

            if new_length < min_length:
                min_length = new_length
                if verbose:
                    tqdm.write(f"New minimal length found: {min_length}")

            if new_length == 2:
                pbar.update(len(tree_nodes) - pbar.n)
                pbar.close()
                return True, path + [(action, new_length)], len(tree_nodes)

            if state_tup not in tree_nodes:
                tree_nodes.add(state_tup)
                heapq.heappush(
                    to_explore,
                    (
                        new_length,
                        path_length + 1,
                        state_tup,
                        tuple(new_lengths),
                        path + [(action, new_length)],
                    ),
                )
                pbar.update(1)

        if len(tree_nodes) >= max_nodes_to_explore:
            pbar.close()
            print(
                f"Exiting search as number of explored nodes = {len(tree_nodes)} has exceeded the limit {max_nodes_to_explore}"
            )
            break

    pbar.close()
    return False, path + [(action, new_length)], len(tree_nodes)


if __name__ == "__main__":
    from ac_solver.search.cli_utils import parse_args, load_presentation, print_path

    args = parse_args(description="Run greedy search on an AC presentation")
    presentation = load_presentation(args)

    ans, path, _ = greedy_search(
        presentation=presentation,
        max_nodes_to_explore=args.max_nodes,
        verbose=True,
        cyclically_reduce_after_moves=args.cyclic_reduce,
    )

    if ans and path:
        print_path(presentation, path)
    else:
        print(f"\nFailed to solve presentation {presentation}.")
