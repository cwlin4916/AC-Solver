"""Tests for PPO presets, device resolution, and new args."""

import sys
import torch
import pytest
from unittest.mock import patch

from ac_solver.agents.args import parse_args, PAPER_PRESET, MAC_PRESET, PRESETS
from ac_solver.agents.ppo import resolve_device


class TestPresets:
    def test_paper_preset_applies_defaults(self):
        test_args = ["script", "--preset", "paper"]
        with patch.object(sys, "argv", test_args):
            args = parse_args()
        assert args.horizon_length == 200
        assert args.num_envs == 28
        assert args.num_steps == 200
        assert args.total_timesteps == 100_000_000
        assert args.update_epochs == 1
        assert args.repeat_solved_prob == 0.25

    def test_mac_preset_applies_defaults(self):
        test_args = ["script", "--preset", "mac"]
        with patch.object(sys, "argv", test_args):
            args = parse_args()
        assert args.horizon_length == 200
        assert args.num_envs == 8
        assert args.total_timesteps == 10_000_000

    def test_cli_overrides_preset(self):
        test_args = ["script", "--preset", "mac", "--num-envs", "16"]
        with patch.object(sys, "argv", test_args):
            args = parse_args()
        assert args.num_envs == 16  # CLI override
        assert args.horizon_length == 200  # from preset

    def test_no_preset_uses_original_defaults(self):
        test_args = ["script"]
        with patch.object(sys, "argv", test_args):
            args = parse_args()
        assert args.horizon_length == 2000  # original default
        assert args.num_envs == 4  # original default
        assert args.total_timesteps == 200000  # original default

    def test_mac_preset_inherits_paper(self):
        for key, value in PAPER_PRESET.items():
            if key not in ("num_envs", "total_timesteps", "device"):
                assert MAC_PRESET[key] == value, f"mac_preset[{key}] differs from paper"


class TestDeviceResolution:
    def test_cpu_explicit(self):
        device = resolve_device("cpu")
        assert device == torch.device("cpu")

    def test_auto_returns_valid_device(self):
        device = resolve_device("auto")
        assert device.type in ("cpu", "cuda", "mps")

    def test_cuda_explicit(self):
        device = resolve_device("cuda")
        assert device == torch.device("cuda")


class TestNewArgs:
    def test_device_arg(self):
        test_args = ["script", "--device", "cpu"]
        with patch.object(sys, "argv", test_args):
            args = parse_args()
        assert args.device == "cpu"

    def test_wandb_mode_default(self):
        test_args = ["script"]
        with patch.object(sys, "argv", test_args):
            args = parse_args()
        assert args.wandb_mode == "offline"

    def test_wandb_mode_online(self):
        test_args = ["script", "--wandb-mode", "online"]
        with patch.object(sys, "argv", test_args):
            args = parse_args()
        assert args.wandb_mode == "online"

    def test_resume_default_none(self):
        test_args = ["script"]
        with patch.object(sys, "argv", test_args):
            args = parse_args()
        assert args.resume is None

    def test_checkpoint_args(self):
        test_args = ["script", "--checkpoint-dir", "/tmp/ckpts", "--checkpoint-interval", "50"]
        with patch.object(sys, "argv", test_args):
            args = parse_args()
        assert args.checkpoint_dir == "/tmp/ckpts"
        assert args.checkpoint_interval == 50

    def test_eval_at_default(self):
        test_args = ["script"]
        with patch.object(sys, "argv", test_args):
            args = parse_args()
        assert args.eval_at == [100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]

    def test_eval_at_custom(self):
        test_args = ["script", "--eval-at", "10000", "50000"]
        with patch.object(sys, "argv", test_args):
            args = parse_args()
        assert args.eval_at == [10000, 50000]

    def test_batch_size_computed(self):
        test_args = ["script", "--preset", "mac"]
        with patch.object(sys, "argv", test_args):
            args = parse_args()
        assert args.batch_size == args.num_envs * args.num_steps
        assert args.minibatch_size == args.batch_size // args.num_minibatches
