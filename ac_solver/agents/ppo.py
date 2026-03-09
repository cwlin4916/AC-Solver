"""
This file trains a PPO (Proximal Policy Optimization) agent on AC Environment.
It sets up the training environment, initializes the agent, and runs the PPO training loop.
Run this script directly to start training the PPO agent, as simply as

```
python -m ac_solver.agents.ppo
```

To see the entire list of command line arguments you may pass, check args.py
"""

import numpy as np
import torch
import random
from torch.optim import Adam
from ac_solver.agents.ppo_agent import Agent
from ac_solver.agents.args import parse_args
from ac_solver.agents.environment import get_env
from ac_solver.agents.training import ppo_training_loop


def resolve_device(device_str):
    """Resolve device string to torch.device, with MPS support for Mac."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def train_ppo():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    (
        envs,
        initial_states,
        curr_states,
        success_record,
        ACMoves_hist,
        states_processed,
    ) = get_env(args)

    agent = Agent(envs, args.nodes_counts).to(device)
    optimizer = Adam(agent.parameters(), lr=args.learning_rate, eps=args.epsilon)

    start_update = 1
    resumed_state = {}
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        agent.actor.load_state_dict(ckpt["actor"])
        agent.critic.load_state_dict(ckpt["critic"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_update = ckpt["update"] + 1
        resumed_state = {
            "success_record": ckpt["success_record"],
            "ACMoves_hist": ckpt["ACMoves_hist"],
            "curr_states": ckpt["curr_states"],
            "states_processed": ckpt["states_processed"],
            "global_step": ckpt["global_step"],
            "episode": ckpt.get("episode", 0),
            "round1_complete": ckpt.get("round1_complete", False),
        }
        print(f"Resuming from update {start_update}, global_step {resumed_state['global_step']}")

    ppo_training_loop(
        envs,
        args,
        device,
        optimizer,
        agent,
        curr_states,
        success_record,
        ACMoves_hist,
        states_processed,
        initial_states,
        start_update=start_update,
        resumed_state=resumed_state,
    )

    envs.close()


if __name__ == "__main__":
    train_ppo()
