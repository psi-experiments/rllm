import pytest
import torch
from omegaconf import OmegaConf

from rllm.trainer.verl.verl_backend import (
    VerlBackend,
    compute_support_aware_rollout_correction_weights,
)


def _make_config(*, remote_enabled: bool, partial_rollout: bool):
    return OmegaConf.create(
        {
            "actor_rollout_ref": {"rollout": {"mode": "async"}},
            "rllm": {
                "async_training": {
                    "enable": True,
                    "mini_batch_size": 8,
                    "fwd_bwd_group_size": 8,
                    "partial_rollout": partial_rollout,
                },
                "remote_runtime": {"enabled": remote_enabled},
                "stepwise_advantage": {"mode": "broadcast"},
                "algorithm": {"rollout_correction": {}},
            },
            "algorithm": {},
            "reward": {"reward_model": {"enable": False}},
        }
    )


def _backend(config) -> VerlBackend:
    # Bypass __init__ (heavy: builds dataloaders, worker groups) — validate_config
    # only reads self.config and self.is_separated.
    be = VerlBackend.__new__(VerlBackend)
    be.is_separated = True
    be.config = config
    return be


def test_partial_rollout_with_remote_runtime_raises():
    be = _backend(_make_config(remote_enabled=True, partial_rollout=True))
    with pytest.raises(ValueError, match="partial_rollout"):
        be.validate_config()


def test_partial_rollout_disabled_with_remote_runtime_ok():
    be = _backend(_make_config(remote_enabled=True, partial_rollout=False))
    be.validate_config()


def test_partial_rollout_without_remote_runtime_ok():
    be = _backend(_make_config(remote_enabled=False, partial_rollout=True))
    be.validate_config()


def test_old_policy_out_of_support_actions_are_excluded_before_is_statistics():
    weights, metrics = compute_support_aware_rollout_correction_weights(
        torch.tensor([[0.0, 10.0, 0.0, -10.0]]),
        torch.tensor([[1, 1, 1, 0]]),
        torch.tensor([[True, False, True, False]]),
        rollout_is="token",
        rollout_is_threshold=2.0,
    )

    torch.testing.assert_close(weights, torch.tensor([[1.0, 0.0, 1.0, 0.0]]))
    assert metrics["old_policy_out_of_support_fraction"] == pytest.approx(1 / 3)
    assert metrics["rollout_is_ratio_fraction_high"] == 0.0
    assert metrics["rollout_is_mean"] == 1.0
    assert metrics["rollout_is_eff_sample_size"] == pytest.approx(1.0)
    assert metrics["rollout_is_metrics_defined"] == 1.0


def test_fully_unsupported_old_policy_batch_has_zero_weights_and_defined_telemetry():
    weights, metrics = compute_support_aware_rollout_correction_weights(
        torch.tensor([[10.0, -10.0]]),
        torch.tensor([[1, 1]]),
        torch.tensor([[False, False]]),
        rollout_is="token",
        rollout_is_threshold=2.0,
    )

    torch.testing.assert_close(weights, torch.zeros_like(weights))
    assert metrics["old_policy_out_of_support_fraction"] == 1.0
    assert metrics["rollout_is_eff_sample_size"] == 0.0
    assert metrics["rollout_is_metrics_defined"] == 0.0
