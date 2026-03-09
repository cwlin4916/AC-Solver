"""
Implementation of BFS for AC graph.

Example:
Trivialize Akbulut-Kirby series n=2 case "AK(2)" through BFS as 
python breadth_first.py
"""

import numpy as np
from collections import deque
from tqdm import tqdm
from ac_solver.envs.utils import is_array_valid_presentation, is_presentation_trivial
from ac_solver.envs.ac_moves import ACMove


def bfs(
    presentation,
    max_nodes_to_explore=10000,
    verbose=False,
    cyclically_reduce_after_moves=False,
    use_fixed_lengths=True,
    show_progress=True,
):
    """
    Performs a breadth-first search on an AC graph starting from the given presentation.

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

    assert is_array_valid_presentation(
        presentation
    ), f"{presentation} is not a valid presentation"

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

    # add to a queue, keeping track of path length to initial state
    # a set containing states that have already been seen
    state_tup = tuple(initial_state)
    tree_nodes = {state_tup}
    init_path = [(-1, total_initial_length)]
    to_explore = deque([(state_tup, init_path)])  #
    min_length = sum(word_lengths)

    pbar = tqdm(total=max_nodes_to_explore, desc="Searching", unit=" nodes", disable=not show_progress)

    while to_explore:
        state_tuple, path = to_explore.popleft()
        state = np.array(state_tuple, dtype=np.int8)  # convert tuple to state
        if use_fixed_lengths:
            word_lengths = [
                np.count_nonzero(state[:max_relator_length]),
                np.count_nonzero(state[max_relator_length:]),
            ]
        else:
            # Legacy behavior (bug): uses original presentation instead of current state
            word_lengths = [
                np.count_nonzero(presentation[:max_relator_length]),
                np.count_nonzero(presentation[max_relator_length:]),
            ]

        for action in range(0, 12):
            new_state, new_word_lengths = ACMove(
                move_id=action,
                presentation=state,
                max_relator_length=max_relator_length,
                lengths=word_lengths,
                cyclical=cyclically_reduce_after_moves,
            )
            state_tup, new_length = tuple(new_state), sum(new_word_lengths)

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
                to_explore.append((state_tup, path + [(action, new_length)]))
                pbar.update(1)

        if len(tree_nodes) >= max_nodes_to_explore:
            pbar.close()
            print(
                f"Exiting search as number of explored nodes = {len(tree_nodes)} has exceeded the limit {max_nodes_to_explore}"
            )
            break

    pbar.close()
    return False, None, len(tree_nodes)


if __name__ == "__main__":
    from ac_solver.search.cli_utils import parse_args, load_presentation, print_path

    args = parse_args(description="Run BFS on an AC presentation")
    presentation = load_presentation(args)

    ans, path, _ = bfs(
        presentation=presentation,
        max_nodes_to_explore=args.max_nodes,
        verbose=True,
        cyclically_reduce_after_moves=args.cyclic_reduce,
    )

    if ans and path:
        print_path(presentation, path)
    else:
        print(f"\nFailed to solve presentation {presentation}.")
