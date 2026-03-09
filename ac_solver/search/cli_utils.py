"""Shared CLI parsing, move name lookup, and path display for search scripts."""

import argparse
import ast
import numpy as np
from ac_solver.envs.utils import is_presentation_trivial
from ac_solver.envs.ac_moves import ACMove


GEN_NAMES = {1: 'x', 2: 'y', 3: 'z', 4: 'w'}


def _gen_name(index):
    """Map generator index (1,2,3,...) to a display name."""
    return GEN_NAMES.get(index, f'x_{index}')


def _relator_to_str(relator):
    """Convert a relator array (non-zero portion) to a human-readable string."""
    elements = [int(v) for v in relator if v != 0]
    if not elements:
        return '1'
    # group consecutive identical elements into powers
    groups = []
    i = 0
    while i < len(elements):
        val = elements[i]
        count = 1
        while i + count < len(elements) and elements[i + count] == val:
            count += 1
        base = abs(val)
        sign = 1 if val > 0 else -1
        power = sign * count
        name = _gen_name(base)
        if power == 1:
            groups.append(name)
        elif power == -1:
            groups.append(f'{name}^{{-1}}')
        else:
            groups.append(f'{name}^{{{power}}}')
        i += count
    return ''.join(groups)


def state_to_presentation_str(state):
    """Convert a state array to a human-readable presentation string like <x,y | r0, r1>."""
    max_relator_length = len(state) // 2
    r0 = state[:max_relator_length]
    r1 = state[max_relator_length:]
    # determine generators present
    gen_indices = sorted(set(abs(int(v)) for v in state if v != 0))
    gen_str = ','.join(_gen_name(g) for g in gen_indices)
    return f'<{gen_str} | {_relator_to_str(r0)}, {_relator_to_str(r1)}>'


AC_MOVE_NAMES = [
    "r_1 -> r_1 r_0",
    "r_0 -> r_0 r_1^{-1}",
    "r_1 -> r_1 r_0^{-1}",
    "r_0 -> r_0 r_1",
    "r_1 -> x_1^{-1} r_1 x_1",
    "r_0 -> x_2^{-1} r_0 x_2",
    "r_1 -> x_2^{-1} r_1 x_2",
    "r_0 -> x_1 r_0 x_1^{-1}",
    "r_1 -> x_1 r_1 x_1^{-1}",
    "r_0 -> x_2 r_0 x_2^{-1}",
    "r_1 -> x_2 r_1 x_2^{-1}",
    "r_0 -> x_1^{-1} r_0 x_1",
]

PRESETS = {
    "AK2": [1, 1, -2, -2, -2, 0, 0, 1, 2, 1, -2, -1, -2, 0],
    "AK3": [1, 1, 1, -2, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 2, 1, -2, -1, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}


def parse_args(description="Run search on an AC presentation"):
    parser = argparse.ArgumentParser(description=description)

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--presentation",
        type=str,
        default=None,
        help="Comma-separated integers, e.g. '1,1,-2,-2,-2,0,0,1,2,1,-2,-1,-2,0'",
    )
    group.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=list(PRESETS.keys()),
        help="Named preset presentation (default: AK2)",
    )
    group.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to a file containing presentations (one per line)",
    )

    parser.add_argument(
        "--line",
        type=int,
        default=1,
        help="Line number to read from --file (1-indexed, default: 1)",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=1000000,
        help="Max nodes to explore (default: 1000000)",
    )
    parser.add_argument(
        "--cyclic-reduce",
        action="store_true",
        help="Apply cyclic reduction after each move",
    )

    return parser.parse_args()


def load_presentation(args):
    if args.presentation is not None:
        ints = [int(x.strip()) for x in args.presentation.split(",")]
        return np.array(ints, dtype=np.int8)
    elif args.file is not None:
        with open(args.file, "r") as f:
            lines = f.readlines()
        if args.line < 1 or args.line > len(lines):
            raise ValueError(f"Line {args.line} out of range (file has {len(lines)} lines)")
        return np.array(ast.literal_eval(lines[args.line - 1].strip()), dtype=np.int8)
    else:
        # Default to preset (AK2 if --preset not specified)
        preset_name = args.preset if args.preset else "AK2"
        return np.array(PRESETS[preset_name], dtype=np.int8)


def print_path(initial_presentation, path):
    presentation = initial_presentation.copy()
    max_relator_length = len(presentation) // 2

    first_len = np.count_nonzero(presentation[:max_relator_length])
    second_len = np.count_nonzero(presentation[max_relator_length:])
    word_lengths = [first_len, second_len]

    print(f"\nPresentation {initial_presentation} solved!")
    print(f"Path length: {len(path) - 1}")
    print("\nStep-by-step path:")

    state = presentation.copy()
    for step, (action, length) in enumerate(path):
        if action == -1:
            move_str = "[initial state]"
        else:
            state, word_lengths = ACMove(
                move_id=action,
                presentation=state,
                max_relator_length=max_relator_length,
                lengths=word_lengths,
                cyclical=False,
            )
            move_str = AC_MOVE_NAMES[action]
        pres_str = state_to_presentation_str(state)
        print(f"  Step {step:2d}: {move_str:<30s} length={length:2d}  state={state}  {pres_str}")

    print(f"\nIs trivial? {is_presentation_trivial(state)}")
