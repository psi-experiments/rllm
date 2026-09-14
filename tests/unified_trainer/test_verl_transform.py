"""
Tests for the verl transform pipeline, focusing on rollout log probs propagation.

Verifies that log probs from ModelOutput are correctly carried through
_process_trajectory → AccumulatedData → _batch_tensors_and_build_data_proto → DataProto,
so that downstream importance sampling and bypass mode work.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine.rollout import ModelOutput
from rllm.parser import QwenChatTemplateParser
from rllm.trainer.algorithms.config import CompactFilteringConfig, TransformConfig
from rllm.trainer.algorithms.transform import transform_episodes_to_trajectory_groups
from rllm.trainer.verl.dataclass import AccumulatedData
from rllm.trainer.verl.verl_backend import VerlBackend
from rllm.trainer.verl.transform import (
    _align_qwen_cumulative_behavior_logprobs,
    _batch_tensors_and_build_data_proto,
    transform_episodes_to_dataproto,
    transform_trajectory_groups_to_dataproto,
    update_dataproto_with_advantages,
)
from rllm.types import TrajectoryGroup
from rllm.workflows.workflow import TerminationReason

_EXPECTED_BEHAVIOR_SAMPLING = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "repetition_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
}


def _make_mock_rollout_engine(
    pad_token_id: int = 0, *, behavior_correction: bool = False
):
    """Create a mock VerlEngine with a tokenizer."""
    engine = MagicMock()
    engine.tokenizer.pad_token_id = pad_token_id
    engine.processor = None  # No multimodal processor
    engine.train_sampling_params = dict(_EXPECTED_BEHAVIOR_SAMPLING)
    engine.config = {
        "rllm": {
            "rollout": {
                "train": dict(_EXPECTED_BEHAVIOR_SAMPLING),
            }
        },
        "actor_rollout_ref": {
            "actor": {
                "behavior_logprobs_mode": (
                    "processed_logprobs"
                    if behavior_correction
                    else "temperature_only"
                )
            }
        },
    }
    return engine


def _make_episode(
    prompt_ids: list[int],
    completion_ids: list[int],
    logprobs: list[float] | None = None,
    reward: float = 1.0,
    episode_id: str = "task_0:0",
) -> Episode:
    """Create a single-step episode with optional logprobs."""
    model_output = ModelOutput(
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        logprobs=logprobs,
    )
    step = Step(
        prompt_ids=prompt_ids,
        response_ids=completion_ids,
        model_output=model_output,
        reward=reward,
    )
    trajectory = Trajectory(steps=[step], reward=reward)
    return Episode(id=episode_id, trajectories=[trajectory], is_correct=reward > 0)


def _make_transform_only_backend(engine, *, behavior_correction: bool):
    backend = VerlBackend.__new__(VerlBackend)
    backend.rollout_engine = engine
    backend.config = OmegaConf.create(
        {
            "data": {
                "max_prompt_length": 28_672,
                "max_response_length": 4_096,
            },
            "actor_rollout_ref": {
                "actor": {
                    "behavior_logprobs_mode": (
                        "processed_logprobs"
                        if behavior_correction
                        else "temperature_only"
                    )
                }
            },
        }
    )
    return backend


def test_transform_metrics_handle_all_filtered_groups():
    episodes = [Episode(id=f"task:{i}", trajectories=[], termination_reason=TerminationReason.TIMEOUT) for i in range(2)]
    cf_config = CompactFilteringConfig(enable=True, mask_timeout=True)

    groups, metrics = transform_episodes_to_trajectory_groups(
        episodes,
        TransformConfig(),
        cf_config,
    )

    assert groups == []
    assert metrics["groups/num_groups"] == 0
    assert metrics["groups/num_trajs_after_filter"] == 0
    assert metrics["groups/avg_group_size"] == 0.0
    assert metrics["groups/max_group_size"] == 0
    assert metrics["groups/min_group_size"] == 0


class TestRolloutLogProbsPropagation:
    """Tests that rollout log probs flow through the verl transform pipeline."""

    def test_logprobs_included_in_dataproto(self):
        """When steps have logprobs, DataProto should contain rollout_log_probs."""
        episodes = [
            _make_episode(
                prompt_ids=[1, 2, 3],
                completion_ids=[4, 5, 6],
                logprobs=[-0.5, -0.3, -0.1],
            ),
            _make_episode(
                prompt_ids=[10, 11],
                completion_ids=[12, 13, 14, 15],
                logprobs=[-0.2, -0.4, -0.6, -0.8],
                episode_id="task_1:0",
            ),
        ]
        engine = _make_mock_rollout_engine()

        batch = transform_episodes_to_dataproto(episodes, engine, max_prompt_length=8, max_response_length=8)

        assert "rollout_log_probs" in batch.batch, "rollout_log_probs should be present when logprobs are available"
        rollout_lp = batch.batch["rollout_log_probs"]
        assert rollout_lp.shape[0] == 2, "Batch size should be 2"
        assert rollout_lp.shape[1] == 8, "Should be padded to max_response_length"

        # First episode: 3 completion tokens, right-padded with 0
        # The actual logprob values should be present in the first positions
        assert torch.isclose(rollout_lp[0, 0], torch.tensor(-0.5))
        assert torch.isclose(rollout_lp[0, 1], torch.tensor(-0.3))
        assert torch.isclose(rollout_lp[0, 2], torch.tensor(-0.1))
        assert rollout_lp[0, 3] == 0.0  # padding

        # Second episode: 4 completion tokens
        assert torch.isclose(rollout_lp[1, 0], torch.tensor(-0.2))
        assert torch.isclose(rollout_lp[1, 1], torch.tensor(-0.4))
        assert torch.isclose(rollout_lp[1, 2], torch.tensor(-0.6))
        assert torch.isclose(rollout_lp[1, 3], torch.tensor(-0.8))
        assert rollout_lp[1, 4] == 0.0  # padding

    def test_no_logprobs_no_rollout_log_probs_key(self):
        """When steps have no logprobs, DataProto should NOT contain rollout_log_probs."""
        episodes = [
            _make_episode(
                prompt_ids=[1, 2, 3],
                completion_ids=[4, 5, 6],
                logprobs=None,
            ),
        ]
        engine = _make_mock_rollout_engine()

        batch = transform_episodes_to_dataproto(episodes, engine, max_prompt_length=8, max_response_length=8)

        assert "rollout_log_probs" not in batch.batch, "rollout_log_probs should be absent when logprobs are None"

    def test_empty_logprobs_no_rollout_log_probs_key(self):
        """When steps have empty logprobs list, DataProto should NOT contain rollout_log_probs."""
        episodes = [
            _make_episode(
                prompt_ids=[1, 2, 3],
                completion_ids=[4, 5, 6],
                logprobs=[],
            ),
        ]
        engine = _make_mock_rollout_engine()

        batch = transform_episodes_to_dataproto(episodes, engine, max_prompt_length=8, max_response_length=8)

        assert "rollout_log_probs" not in batch.batch, "rollout_log_probs should be absent when logprobs are empty"

    def test_mixed_logprobs_no_rollout_log_probs_key(self):
        """When some steps have logprobs and others don't, rollout_log_probs should be absent (length mismatch guard)."""
        ep_with = _make_episode(
            prompt_ids=[1, 2, 3],
            completion_ids=[4, 5, 6],
            logprobs=[-0.5, -0.3, -0.1],
            episode_id="task_0:0",
        )
        ep_without = _make_episode(
            prompt_ids=[10, 11],
            completion_ids=[12, 13],
            logprobs=None,
            episode_id="task_1:0",
        )
        engine = _make_mock_rollout_engine()

        batch = transform_episodes_to_dataproto([ep_with, ep_without], engine, max_prompt_length=8, max_response_length=8)

        # Length mismatch: 1 logprob tensor but 2 responses → should not include
        assert "rollout_log_probs" not in batch.batch

    def test_multi_step_trajectory_logprobs(self):
        """Cumulative-prefix multi-step trajectories merge into a single row.

        Step 2's prompt [1,2,3,4,5] prefix-extends step 1's full sequence
        [1,2,3,4], so they merge. The resulting row's response is
        [3, 4, 5, 6, 7, 8] (action₀, delta_obs, action₁) with mask
        [1, 1, 0, 1, 1, 1] and logprobs [-0.1, -0.2, 0, -0.3, -0.4, -0.5].
        """
        model_output_1 = ModelOutput(prompt_ids=[1, 2], completion_ids=[3, 4], logprobs=[-0.1, -0.2])
        model_output_2 = ModelOutput(prompt_ids=[1, 2, 3, 4, 5], completion_ids=[6, 7, 8], logprobs=[-0.3, -0.4, -0.5])
        step1 = Step(prompt_ids=[1, 2], response_ids=[3, 4], model_output=model_output_1, reward=0.0)
        step2 = Step(prompt_ids=[1, 2, 3, 4, 5], response_ids=[6, 7, 8], model_output=model_output_2, reward=1.0)
        trajectory = Trajectory(steps=[step1, step2], reward=1.0)
        episode = Episode(id="task_0:0", trajectories=[trajectory], is_correct=True)

        engine = _make_mock_rollout_engine()
        batch = transform_episodes_to_dataproto([episode], engine, max_prompt_length=8, max_response_length=8)

        assert "rollout_log_probs" in batch.batch
        rollout_lp = batch.batch["rollout_log_probs"]
        # Cumulative-prefix merge → 1 row
        assert rollout_lp.shape[0] == 1

        # Merged logprobs: action₀ (real), observation delta (0.0 placeholder),
        # action₁ (real), then right-padded.
        expected_prefix = [-0.1, -0.2, 0.0, -0.3, -0.4, -0.5]
        for i, exp in enumerate(expected_prefix):
            assert torch.isclose(rollout_lp[0, i], torch.tensor(exp)), (i, rollout_lp[0, i].item())

        # Mask follows the same shape: [1, 1, 0, 1, 1, 1]
        response_mask = batch.batch["response_mask"][0]
        assert response_mask[:6].tolist() == [1, 1, 0, 1, 1, 1]

    def test_nonprefix_cumulative_chat_uses_rllm_parser_once(self):
        """Chat-cumulative Qwen turns become one assistant-masked row.

        A chat template may rewrite historical turns, so the second prompt is
        not necessarily a literal extension of the first prompt plus
        completion. The workflow can opt into rLLM's canonical cumulative
        conversion rather than implementing a recipe-specific mask.
        """
        first_messages = [
            {"role": "system", "content": "Use tools."},
            {"role": "user", "content": "Solve this."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "execute",
                            "arguments": '{"code":"result = 1"}',
                        },
                    }
                ],
            },
        ]
        final_messages = first_messages + [
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "execute",
                "content": "1",
            },
            {"role": "assistant", "content": "ANSWER: 1"},
        ]
        output_1 = ModelOutput(
            prompt_ids=[1, 2],
            completion_ids=[3, 4],
            logprobs=[-0.1, -0.2],
            logprobs_mode="processed_logprobs",
            behavior_sampling_params=dict(_EXPECTED_BEHAVIOR_SAMPLING),
        )
        # Deliberately not prefixed by [1, 2, 3, 4].
        output_2 = ModelOutput(
            prompt_ids=[1, 2, 9, 4, 5],
            completion_ids=[6, 7],
            logprobs=[-0.3, -0.4],
            logprobs_mode="processed_logprobs",
            behavior_sampling_params=dict(_EXPECTED_BEHAVIOR_SAMPLING),
        )
        trajectory = Trajectory(
            steps=[
                Step(
                    chat_completions=first_messages,
                    model_output=output_1,
                ),
                Step(
                    chat_completions=final_messages,
                    model_output=output_2,
                ),
            ],
            reward=1.0,
            metadata={
                "rllm_cumulative_chat": {
                    "tools": [{"type": "function", "function": {"name": "execute"}}],
                    "max_total_length": 8,
                }
            },
        )
        episode = Episode(
            id="task_0:0",
            trajectories=[trajectory],
            is_correct=True,
            artifacts={
                "generation_config": {
                    **_EXPECTED_BEHAVIOR_SAMPLING,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "context_budget_reductions": [],
                    "model_context_tokens": 32768,
                    "model_request_max_tokens": [4096, 2048],
                },
            },
        )
        non_qwen_engine = _make_mock_rollout_engine()
        non_qwen_batch = transform_episodes_to_dataproto(
            [episode],
            non_qwen_engine,
            max_prompt_length=8,
            max_response_length=8,
            max_total_length=8,
        )
        assert non_qwen_batch.batch["responses"].shape[0] == 2
        non_qwen_engine.chat_parser.tokenize_and_mask_cumulative.assert_not_called()

        engine = _make_mock_rollout_engine()
        engine.chat_parser = MagicMock(spec=QwenChatTemplateParser)
        engine.chat_parser.tokenize_and_mask_cumulative.return_value = (
            torch.tensor([10, 11]),
            torch.tensor([12, 13, 14]),
            torch.tensor([1, 0, 1]),
        )

        batch = transform_episodes_to_dataproto(
            [episode],
            engine,
            max_prompt_length=8,
            max_response_length=8,
            max_total_length=8,
        )

        engine.chat_parser.tokenize_and_mask_cumulative.assert_called_once_with(
            final_messages,
            tools=[{"type": "function", "function": {"name": "execute"}}],
        )
        assert batch.batch["responses"].shape[0] == 1
        assert batch.batch["responses"][0, :3].tolist() == [12, 13, 14]
        assert batch.batch["response_mask"][0, :3].tolist() == [1, 0, 1]
        assert "rollout_log_probs" not in batch.batch
        assert "rollout_log_probs_provenance" not in batch.meta_info
        metrics = batch.meta_info["merge_metrics"]
        assert metrics["batch/behavior_logprobs_aligned_rows"] == 0
        assert metrics["batch/behavior_logprobs_rejected_rows"] == 1
        assert metrics["batch/behavior_logprobs_rejected/prompt_token_mismatch"] == 1

    def test_qwen_cumulative_chat_preserves_exact_behavior_logprobs(self):
        """Exact served contexts retain processed logprobs on action tokens."""

        first_messages = [
            {"role": "user", "content": "solve"},
            {"role": "assistant", "content": "first"},
        ]
        final_messages = first_messages + [
            {"role": "tool", "content": "result"},
            {"role": "assistant", "content": "answer"},
        ]
        trajectory = Trajectory(
            steps=[
                Step(
                    chat_completions=first_messages,
                    model_output=ModelOutput(
                        prompt_ids=[10, 11],
                        completion_ids=[12, 13],
                        logprobs=[-0.1, -0.2],
                        logprobs_mode="processed_logprobs",
                        behavior_sampling_params=dict(
                            _EXPECTED_BEHAVIOR_SAMPLING
                        ),
                    ),
                ),
                Step(
                    chat_completions=final_messages,
                    model_output=ModelOutput(
                        prompt_ids=[10, 11, 12, 13, 14],
                        completion_ids=[15, 16],
                        logprobs=[-0.3, -0.4],
                        logprobs_mode="processed_logprobs",
                        behavior_sampling_params=dict(
                            _EXPECTED_BEHAVIOR_SAMPLING
                        ),
                    ),
                ),
            ],
            reward=1.0,
            metadata={
                "rllm_cumulative_chat": {
                    "tools": [],
                    "max_total_length": 8,
                }
            },
        )
        engine = _make_mock_rollout_engine()
        engine.chat_parser = MagicMock(spec=QwenChatTemplateParser)
        engine.chat_parser.tokenize_and_mask_cumulative.return_value = (
            torch.tensor([10, 11]),
            torch.tensor([12, 13, 14, 15, 16]),
            torch.tensor([1, 1, 0, 1, 1]),
        )

        batch = transform_episodes_to_dataproto(
            [Episode(id="task:0", trajectories=[trajectory])],
            engine,
            max_prompt_length=8,
            max_response_length=8,
            max_total_length=8,
        )

        assert batch.batch["response_mask"][0, :5].tolist() == [1, 1, 0, 1, 1]
        assert batch.batch["rollout_log_probs"][0, :5].tolist() == pytest.approx(
            [-0.1, -0.2, 0.0, -0.3, -0.4]
        )
        metrics = batch.meta_info["merge_metrics"]
        assert metrics["batch/behavior_logprobs_aligned_rows"] == 1
        assert metrics["batch/behavior_logprobs_rejected_rows"] == 0
        assert batch.meta_info["rollout_log_probs_provenance"] == {
            "logprobs_mode": "processed_logprobs",
            "sampling_params": _EXPECTED_BEHAVIOR_SAMPLING,
            "rows": 1,
        }

    def test_qwen_correction_uses_exact_served_turns_despite_chat_rerender(self):
        """No-thinking/tool re-render drift must not alter behavior rows.

        This fixture includes the two surface details from the live failure:
        literal Unicode in a tool schema and a tool-call JSON key order that a
        structured parser may normalize.  The mocked canonical renderer is
        deliberately incompatible with the served IDs.  Correction mode must
        never call it: exact prompt/completion IDs and logprobs come from each
        captured inference turn.
        """
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "execute",
                    "description": "Run — then return the result",
                },
            }
        ]
        first_messages = [
            {"role": "system", "content": "Use tools."},
            {"role": "user", "content": "Solve this."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            # Deliberately arguments-before-name.
                            "arguments": '{"code":"result = 1"}',
                            "name": "execute",
                        },
                    }
                ],
            },
        ]
        final_messages = first_messages + [
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "execute",
                "content": "1",
            },
            {"role": "assistant", "content": "ANSWER: 1"},
        ]
        first = Step(
            chat_completions=first_messages,
            model_output=ModelOutput(
                prompt_ids=[101, 102, 103],
                completion_ids=[201, 202],
                logprobs=[-0.1, -0.2],
                logprobs_mode="processed_logprobs",
                behavior_sampling_params=dict(_EXPECTED_BEHAVIOR_SAMPLING),
                generator_version_spans=[
                    {"start": 0, "end": 2, "weight_version": 4}
                ],
            ),
            advantage=2.0,
        )
        second = Step(
            chat_completions=final_messages,
            model_output=ModelOutput(
                # This is intentionally not first prompt + first completion.
                # Qwen dropped the generation-only no-thinking stub while
                # rendering the old assistant turn into new history.
                prompt_ids=[101, 102, 999, 103, 201, 202, 301],
                completion_ids=[401, 402, 403],
                logprobs=[-0.3, -0.4, -0.5],
                logprobs_mode="processed_logprobs",
                behavior_sampling_params=dict(_EXPECTED_BEHAVIOR_SAMPLING),
                generator_version_spans=[
                    {"start": 0, "end": 1, "weight_version": 4},
                    {"start": 1, "end": 3, "weight_version": 5},
                ],
            ),
            advantage=2.0,
        )
        trajectory = Trajectory(
            steps=[first, second],
            reward=1.0,
            metadata={
                "rllm_cumulative_chat": {
                    "tools": tools,
                    "max_total_length": 16,
                }
            },
        )
        episode = Episode(
            id="task:0",
            trajectories=[trajectory],
            artifacts={
                "generation_config": dict(_EXPECTED_BEHAVIOR_SAMPLING)
            },
        )
        engine = _make_mock_rollout_engine(behavior_correction=True)
        engine.chat_parser = MagicMock(spec=QwenChatTemplateParser)
        engine.chat_parser.tokenize_and_mask_cumulative.return_value = (
            torch.tensor([800, 801]),
            torch.tensor([802, 803]),
            torch.tensor([1, 1]),
        )

        batch = transform_episodes_to_dataproto(
            [episode],
            engine,
            max_prompt_length=16,
            max_response_length=8,
            max_total_length=16,
        )

        engine.chat_parser.tokenize_and_mask_cumulative.assert_not_called()
        assert batch.batch["prompts"][0, -3:].tolist() == [101, 102, 103]
        assert batch.batch["responses"][0, :2].tolist() == [201, 202]
        assert batch.batch["prompts"][1, -7:].tolist() == [
            101,
            102,
            999,
            103,
            201,
            202,
            301,
        ]
        assert batch.batch["responses"][1, :3].tolist() == [401, 402, 403]
        assert batch.batch["response_mask"][:, :3].tolist() == [
            [1, 1, 0],
            [1, 1, 1],
        ]
        assert torch.allclose(
            batch.batch["rollout_log_probs"][:, :3],
            torch.tensor([[-0.1, -0.2, 0.0], [-0.3, -0.4, -0.5]]),
        )
        assert batch.meta_info["rollout_log_probs_provenance"] == {
            "logprobs_mode": "processed_logprobs",
            "sampling_params": _EXPECTED_BEHAVIOR_SAMPLING,
            "rows": 2,
        }
        assert batch.non_tensor_batch["advantage_weights"].tolist() == pytest.approx(
            [2 / 5, 3 / 5]
        )

        update_dataproto_with_advantages(batch, [episode])
        assert batch.batch["advantages"][0, :2].tolist() == pytest.approx(
            [0.8, 0.8]
        )
        assert batch.batch["advantages"][1, :3].tolist() == pytest.approx(
            [1.2, 1.2, 1.2]
        )
        # The weighted sum of per-turn token means exactly equals the old
        # one-row whole-trajectory token mean times its scalar advantage.
        turn_losses = [torch.tensor([1.0, 3.0]), torch.tensor([2.0, 4.0, 6.0])]
        split_objective = sum(
            float(weight) * float(loss.mean()) * 2.0
            for weight, loss in zip(
                batch.non_tensor_batch["advantage_weights"],
                turn_losses,
                strict=True,
            )
        )
        merged_objective = 2.0 * float(torch.cat(turn_losses).mean())
        assert split_objective == pytest.approx(merged_objective)

    @pytest.mark.parametrize(
        ("mutation", "expected_reason"),
        [
            ({"logprobs_mode": "raw_logprobs"}, "logprobs_mode_not_processed"),
            ({"logprobs": [-0.3]}, "logprob_length_mismatch"),
            ({"logprobs": [-0.3, float("inf")]}, "nonfinite_logprob"),
            (
                {
                    "behavior_sampling_params": {
                        **_EXPECTED_BEHAVIOR_SAMPLING,
                        "top_p": 0.9,
                    }
                },
                "step_sampling_param_mismatch_top_p",
            ),
        ],
    )
    def test_exact_qwen_rows_fail_closed_for_any_invalid_turn(
        self, mutation, expected_reason
    ):
        outputs = [
            ModelOutput(
                prompt_ids=[1, 2],
                completion_ids=[3, 4],
                logprobs=[-0.1, -0.2],
                logprobs_mode="processed_logprobs",
                behavior_sampling_params=dict(_EXPECTED_BEHAVIOR_SAMPLING),
            ),
            ModelOutput(
                prompt_ids=[1, 2, 9],
                completion_ids=[5, 6],
                logprobs=[-0.3, -0.4],
                logprobs_mode="processed_logprobs",
                behavior_sampling_params=dict(_EXPECTED_BEHAVIOR_SAMPLING),
            ),
        ]
        steps = [
            Step(
                chat_completions=[
                    {"role": "user", "content": "x"},
                    *(
                        []
                        if index == 0
                        else [{"role": "assistant", "content": "first"}]
                    ),
                    {"role": "assistant", "content": f"turn {index}"},
                ],
                model_output=output,
            )
            for index, output in enumerate(outputs)
        ]
        # Mutate after Step construction so this test exercises the transform's
        # fail-closed validation rather than Step's legacy length assertion.
        for name, value in mutation.items():
            setattr(steps[1].model_output, name, value)
        # Make the structured histories genuinely cumulative; their token IDs
        # deliberately need not be.
        steps[1].chat_completions = steps[0].chat_completions + [
            {"role": "tool", "content": "result"},
            {"role": "assistant", "content": "turn 1"},
        ]
        trajectory = Trajectory(
            steps=steps,
            metadata={
                "rllm_cumulative_chat": {"tools": [], "max_total_length": 8}
            },
        )
        episode = Episode(
            id="task:0",
            trajectories=[trajectory],
            artifacts={
                "generation_config": dict(_EXPECTED_BEHAVIOR_SAMPLING)
            },
        )
        engine = _make_mock_rollout_engine(behavior_correction=True)
        engine.chat_parser = MagicMock(spec=QwenChatTemplateParser)

        batch = transform_episodes_to_dataproto(
            [episode], engine, max_prompt_length=8, max_response_length=8
        )

        assert batch.batch["responses"].shape[0] == 2
        assert "rollout_log_probs" not in batch.batch
        assert "rollout_log_probs_provenance" not in batch.meta_info
        metrics = batch.meta_info["merge_metrics"]
        assert metrics["batch/behavior_logprobs_aligned_rows"] == 0
        assert metrics["batch/behavior_logprobs_rejected_rows"] == 2
        assert metrics[f"batch/behavior_logprobs_rejected/{expected_reason}"] == 2

    def test_exact_qwen_provenance_requires_every_batch_row(self):
        def trajectory(logprobs):
            return Trajectory(
                steps=[
                    Step(
                        chat_completions=[
                            {"role": "user", "content": "x"},
                            {"role": "assistant", "content": "answer"},
                        ],
                        model_output=ModelOutput(
                            prompt_ids=[1, 2],
                            completion_ids=[3, 4],
                            logprobs=logprobs,
                            logprobs_mode="processed_logprobs",
                            behavior_sampling_params=dict(
                                _EXPECTED_BEHAVIOR_SAMPLING
                            ),
                        ),
                    )
                ],
                metadata={
                    "rllm_cumulative_chat": {
                        "tools": [],
                        "max_total_length": 8,
                    }
                },
            )

        valid_trajectory = trajectory([-0.1, -0.2])
        invalid_trajectory = trajectory([-0.3, -0.4])
        invalid_trajectory.steps[0].model_output.logprobs = [-0.3]
        episode = Episode(
            id="task:0",
            trajectories=[valid_trajectory, invalid_trajectory],
            artifacts={
                "generation_config": dict(_EXPECTED_BEHAVIOR_SAMPLING)
            },
        )
        engine = _make_mock_rollout_engine(behavior_correction=True)
        engine.chat_parser = MagicMock(spec=QwenChatTemplateParser)

        batch = transform_episodes_to_dataproto(
            [episode], engine, max_prompt_length=8, max_response_length=8
        )

        assert batch.batch["responses"].shape[0] == 2
        assert "rollout_log_probs" not in batch.batch
        assert "rollout_log_probs_provenance" not in batch.meta_info
        assert batch.meta_info["merge_metrics"][
            "batch/behavior_logprobs_aligned_rows"
        ] == 1
        assert batch.meta_info["merge_metrics"][
            "batch/behavior_logprobs_rejected_rows"
        ] == 1

    def test_exact_qwen_rows_survive_async_trajectory_group_transform(self):
        step = Step(
            chat_completions=[
                {"role": "user", "content": "solve"},
                {"role": "assistant", "content": "answer"},
            ],
            model_output=ModelOutput(
                prompt_ids=[10, 11],
                completion_ids=[12, 13],
                logprobs=[-0.2, -0.4],
                logprobs_mode="processed_logprobs",
                behavior_sampling_params=dict(_EXPECTED_BEHAVIOR_SAMPLING),
            ),
            advantage=1.5,
        )
        trajectory = Trajectory(
            steps=[step],
            metadata={
                "rllm_cumulative_chat": {"tools": [], "max_total_length": 8}
            },
        )
        group = TrajectoryGroup(
            trajectories=[trajectory], group_id="task-0:all_groups"
        )
        engine = _make_mock_rollout_engine(behavior_correction=True)
        engine.chat_parser = MagicMock(spec=QwenChatTemplateParser)

        batch = transform_trajectory_groups_to_dataproto(
            [group], engine, max_prompt_length=8, max_response_length=8
        )

        assert batch.batch["rollout_log_probs"][0, :2].tolist() == pytest.approx(
            [-0.2, -0.4]
        )
        assert batch.meta_info["rollout_log_probs_provenance"] == {
            "logprobs_mode": "processed_logprobs",
            "sampling_params": _EXPECTED_BEHAVIOR_SAMPLING,
            "rows": 1,
        }
        update_dataproto_with_advantages(batch, [group])
        assert batch.batch["advantages"][0, :2].tolist() == pytest.approx(
            [1.5, 1.5]
        )

    def test_backend_accepts_exact_30k_prompt_plus_2k_completion(self):
        prompt_ids = [1] * 30_000
        completion_ids = [2] * 2_000
        trajectory = Trajectory(
            steps=[
                Step(
                    chat_completions=[
                        {"role": "user", "content": "long context"},
                        {"role": "assistant", "content": "answer"},
                    ],
                    model_output=ModelOutput(
                        prompt_ids=prompt_ids,
                        completion_ids=completion_ids,
                        logprobs=[-0.25] * len(completion_ids),
                        logprobs_mode="processed_logprobs",
                        behavior_sampling_params=dict(
                            _EXPECTED_BEHAVIOR_SAMPLING
                        ),
                    ),
                )
            ],
            metadata={
                "rllm_cumulative_chat": {
                    "tools": [],
                    "max_total_length": 32_768,
                }
            },
        )
        episode = Episode(
            id="long:0",
            trajectories=[trajectory],
            artifacts={
                "generation_config": dict(_EXPECTED_BEHAVIOR_SAMPLING)
            },
        )
        engine = _make_mock_rollout_engine(behavior_correction=True)
        engine.chat_parser = MagicMock(spec=QwenChatTemplateParser)
        backend = _make_transform_only_backend(
            engine, behavior_correction=True
        )
        state = SimpleNamespace(
            episodes=[episode], trajectory_groups=None, metrics={}
        )

        batch = backend.transform_to_backend_batch(state)

        assert int(batch.batch["attention_mask"].sum().item()) == 32_000
        assert batch.batch["prompts"].shape[1] == 32_768
        assert batch.batch["responses"].shape[1] == 4_096
        assert batch.batch["input_ids"].shape[1] == 36_864
        assert batch.batch["prompts"][0, -30_000:].tolist() == prompt_ids
        assert batch.batch["responses"][0, :2_000].tolist() == completion_ids
        assert batch.meta_info["rollout_log_probs_provenance"]["rows"] == 1

    def test_backend_rejects_exact_turn_beyond_serving_context(self):
        prompt_ids = [1] * 30_800
        completion_ids = [2] * 2_000
        trajectory = Trajectory(
            steps=[
                Step(
                    chat_completions=[
                        {"role": "user", "content": "too long"},
                        {"role": "assistant", "content": "answer"},
                    ],
                    model_output=ModelOutput(
                        prompt_ids=prompt_ids,
                        completion_ids=completion_ids,
                        logprobs=[-0.25] * len(completion_ids),
                        logprobs_mode="processed_logprobs",
                        behavior_sampling_params=dict(
                            _EXPECTED_BEHAVIOR_SAMPLING
                        ),
                    ),
                )
            ],
            metadata={
                "rllm_cumulative_chat": {
                    "tools": [],
                    "max_total_length": 32_768,
                }
            },
        )
        episode = Episode(
            id="overlong:0",
            trajectories=[trajectory],
            artifacts={
                "generation_config": dict(_EXPECTED_BEHAVIOR_SAMPLING)
            },
        )
        engine = _make_mock_rollout_engine(behavior_correction=True)
        engine.chat_parser = MagicMock(spec=QwenChatTemplateParser)
        backend = _make_transform_only_backend(
            engine, behavior_correction=True
        )
        state = SimpleNamespace(
            episodes=[episode], trajectory_groups=None, metrics={}
        )

        with pytest.raises(ValueError, match="sequence cap"):
            backend.transform_to_backend_batch(state)

    def test_backend_keeps_ordinary_prompt_storage_width(self):
        engine = _make_mock_rollout_engine(behavior_correction=False)
        backend = _make_transform_only_backend(
            engine, behavior_correction=False
        )
        episode = _make_episode(
            prompt_ids=[1] * 28_673,
            completion_ids=[2],
        )
        state = SimpleNamespace(
            episodes=[episode], trajectory_groups=None, metrics={}
        )

        with pytest.raises(ValueError, match="prompt storage width"):
            backend.transform_to_backend_batch(state)

    @pytest.mark.parametrize(
        ("prompt_ids", "completion_ids", "logprobs", "mask", "expected_reason"),
        [
            ([10, 99], [12, 13], [-0.1, -0.2], [1, 1, 0], "prompt_token_mismatch"),
            ([10, 11], [12, 99], [-0.1, -0.2], [1, 1, 0], "completion_token_mismatch"),
            ([10, 11], [12, 13], None, [1, 1, 0], "missing_logprobs"),
            ([10, 11], [12, 13], [-0.1], [1, 1, 0], "logprob_length_mismatch"),
            ([10, 11], [12, 13], [-0.1, float("nan")], [1, 1, 0], "nonfinite_logprob"),
            ([10, 11], [12, 13], [-0.1, -0.2], [1, 0, 0], "completion_not_fully_action_masked"),
            ([10, 11], [12], [-0.1], [1, 1, 0], "incomplete_action_coverage"),
        ],
    )
    def test_qwen_behavior_logprobs_fail_closed(
        self,
        prompt_ids,
        completion_ids,
        logprobs,
        mask,
        expected_reason,
    ):
        step = SimpleNamespace(
            model_output=SimpleNamespace(
                prompt_ids=prompt_ids,
                completion_ids=completion_ids,
                logprobs=logprobs,
                logprobs_mode="processed_logprobs",
                behavior_sampling_params=dict(_EXPECTED_BEHAVIOR_SAMPLING),
            )
        )

        aligned, reason = _align_qwen_cumulative_behavior_logprobs(
            torch.tensor([10, 11]),
            torch.tensor([12, 13, 14]),
            torch.tensor(mask),
            [step],
            expected_sampling_params=_EXPECTED_BEHAVIOR_SAMPLING,
        )

        assert aligned is None
        assert reason == expected_reason

    @pytest.mark.parametrize(
        ("logprobs_mode", "sampling_params", "expected_reason"),
        [
            (
                None,
                _EXPECTED_BEHAVIOR_SAMPLING,
                "logprobs_mode_not_processed",
            ),
            (
                "raw_logprobs",
                _EXPECTED_BEHAVIOR_SAMPLING,
                "logprobs_mode_not_processed",
            ),
            (
                "processed_logprobs",
                {
                    key: value
                    for key, value in _EXPECTED_BEHAVIOR_SAMPLING.items()
                    if key != "top_p"
                },
                "missing_step_sampling_param_top_p",
            ),
            (
                "processed_logprobs",
                {**_EXPECTED_BEHAVIOR_SAMPLING, "top_p": 0.9},
                "step_sampling_param_mismatch_top_p",
            ),
            (
                "processed_logprobs",
                {**_EXPECTED_BEHAVIOR_SAMPLING, "repetition_penalty": 1.1},
                "step_sampling_param_mismatch_repetition_penalty",
            ),
            (
                "processed_logprobs",
                {**_EXPECTED_BEHAVIOR_SAMPLING, "min_p": 0.1},
                "unsupported_step_sampling_control_min_p",
            ),
            (
                "processed_logprobs",
                {**_EXPECTED_BEHAVIOR_SAMPLING, "logit_bias": {"1": 2.0}},
                "unsupported_step_sampling_control_logit_bias",
            ),
            (
                "processed_logprobs",
                {**_EXPECTED_BEHAVIOR_SAMPLING, "guided_json": {"type": "object"}},
                "unsupported_step_sampling_control_guided_json",
            ),
            (
                "processed_logprobs",
                {**_EXPECTED_BEHAVIOR_SAMPLING, "use_beam_search": True},
                "unsupported_step_sampling_control_use_beam_search",
            ),
            (
                "processed_logprobs",
                {**_EXPECTED_BEHAVIOR_SAMPLING, "best_of": 2},
                "unsupported_step_sampling_control_best_of",
            ),
        ],
    )
    def test_qwen_behavior_logprobs_require_matching_provenance(
        self,
        logprobs_mode,
        sampling_params,
        expected_reason,
    ):
        step = SimpleNamespace(
            model_output=SimpleNamespace(
                prompt_ids=[10, 11],
                completion_ids=[12, 13],
                logprobs=[-0.1, -0.2],
                logprobs_mode=logprobs_mode,
                behavior_sampling_params=sampling_params,
            )
        )

        aligned, reason = _align_qwen_cumulative_behavior_logprobs(
            torch.tensor([10, 11]),
            torch.tensor([12, 13]),
            torch.tensor([1, 1]),
            [step],
            expected_sampling_params=_EXPECTED_BEHAVIOR_SAMPLING,
        )

        assert aligned is None
        assert reason == expected_reason

    def test_qwen_behavior_logprobs_require_episode_config_agreement(self):
        step = SimpleNamespace(
            model_output=SimpleNamespace(
                prompt_ids=[10, 11],
                completion_ids=[12, 13],
                logprobs=[-0.1, -0.2],
                logprobs_mode="processed_logprobs",
                behavior_sampling_params=dict(_EXPECTED_BEHAVIOR_SAMPLING),
            )
        )

        aligned, reason = _align_qwen_cumulative_behavior_logprobs(
            torch.tensor([10, 11]),
            torch.tensor([12, 13]),
            torch.tensor([1, 1]),
            [step],
            expected_sampling_params=_EXPECTED_BEHAVIOR_SAMPLING,
            episode_sampling_params={
                **_EXPECTED_BEHAVIOR_SAMPLING,
                "temperature": 0.7,
            },
        )

        assert aligned is None
        assert reason == "episode_sampling_param_mismatch_temperature"

    def test_qwen_cumulative_chat_is_forced_even_when_raw_tokens_are_prefix_cumulative(self):
        """Opt-in Qwen traces always use one canonical rendering path."""

        first_messages = [
            {"role": "user", "content": "solve"},
            {"role": "assistant", "content": "searching"},
        ]
        final_messages = first_messages + [
            {"role": "tool", "content": "result"},
            {"role": "assistant", "content": "answer"},
        ]
        trajectory = Trajectory(
            steps=[
                Step(
                    chat_completions=first_messages,
                    model_output=ModelOutput(
                        prompt_ids=[1, 2],
                        completion_ids=[3, 4],
                    ),
                ),
                Step(
                    chat_completions=final_messages,
                    # This really is a literal extension of prompt+completion.
                    model_output=ModelOutput(
                        prompt_ids=[1, 2, 3, 4, 5],
                        completion_ids=[6],
                    ),
                ),
            ],
            reward=1.0,
            metadata={
                "rllm_cumulative_chat": {
                    "tools": [],
                    "max_total_length": 8,
                }
            },
        )
        episode = Episode(id="task_0:0", trajectories=[trajectory], is_correct=True)
        engine = _make_mock_rollout_engine()
        engine.chat_parser = MagicMock(spec=QwenChatTemplateParser)
        engine.chat_parser.tokenize_and_mask_cumulative.return_value = (
            torch.tensor([10, 11]),
            torch.tensor([12, 13, 14]),
            torch.tensor([1, 0, 1]),
        )

        batch = transform_episodes_to_dataproto(
            [episode],
            engine,
            max_prompt_length=8,
            max_response_length=8,
            max_total_length=8,
        )

        engine.chat_parser.tokenize_and_mask_cumulative.assert_called_once_with(
            final_messages,
            tools=[],
        )
        assert batch.batch["responses"][0, :3].tolist() == [12, 13, 14]

    def test_qwen_structured_expansion_is_clipped_to_exact_total_cap(self):
        """Regression: sampled 32,760 + 8 can canonicalize eight tokens larger."""

        final_messages = [
            {"role": "user", "content": "solve"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"query":"qutip"}',
                        },
                    }
                ],
            },
        ]
        # vLLM saw a 32,760-token prompt and accepted an 8-token sample.
        # rLLM's structured reserialization below is 8 tokens larger overall.
        sampled_prompt = list(range(32_760))
        trajectory = Trajectory(
            steps=[
                Step(
                    chat_completions=final_messages,
                    model_output=ModelOutput(
                        prompt_ids=sampled_prompt,
                        completion_ids=list(range(8)),
                    ),
                )
            ],
            reward=0.0,
            metadata={
                "rllm_cumulative_chat": {
                    "tools": [{"type": "function", "function": {"name": "search"}}],
                    "max_total_length": 32_768,
                }
            },
        )
        episode = Episode(id="task_0:0", trajectories=[trajectory])
        engine = _make_mock_rollout_engine()
        engine.chat_parser = MagicMock(spec=QwenChatTemplateParser)
        canonical_prompt = torch.arange(100) + 1
        canonical_response = torch.arange(32_676) + 1_000
        canonical_mask = torch.ones(32_676, dtype=torch.long)
        engine.chat_parser.tokenize_and_mask_cumulative.return_value = (
            canonical_prompt,
            canonical_response,
            canonical_mask,
        )

        batch = transform_episodes_to_dataproto(
            [episode],
            engine,
            max_prompt_length=28_672,
            max_response_length=32_768,
            max_total_length=32_768,
        )

        assert int(batch.batch["attention_mask"].sum().item()) == 32_768
        assert torch.equal(
            batch.batch["responses"][0, :32_668],
            canonical_response[:32_668],
        )
        assert int(batch.batch["response_mask"].sum().item()) == 32_668
        metrics = batch.meta_info["merge_metrics"]
        assert metrics["batch/context_clipped_rows"] == 1
        assert metrics["batch/context_clipped_tokens"] == 8

    def test_declared_qwen_cap_can_be_stricter_than_backend_cap(self):
        messages = [
            {"role": "user", "content": "solve"},
            {"role": "assistant", "content": "answer"},
        ]
        trajectory = Trajectory(
            steps=[
                Step(
                    chat_completions=messages,
                    model_output=ModelOutput(prompt_ids=[1], completion_ids=[2]),
                )
            ],
            metadata={
                "rllm_cumulative_chat": {
                    "tools": [],
                    "max_total_length": 6,
                }
            },
        )
        engine = _make_mock_rollout_engine()
        engine.chat_parser = MagicMock(spec=QwenChatTemplateParser)
        engine.chat_parser.tokenize_and_mask_cumulative.return_value = (
            torch.tensor([1, 2]),
            torch.tensor([3, 4, 5, 6, 7]),
            torch.tensor([1, 1, 1, 1, 1]),
        )

        batch = transform_episodes_to_dataproto(
            [Episode(id="task:0", trajectories=[trajectory])],
            engine,
            max_prompt_length=8,
            max_response_length=8,
            max_total_length=8,
        )

        assert batch.batch["responses"][0, :4].tolist() == [3, 4, 5, 6]
        assert batch.meta_info["merge_metrics"]["batch/context_clipped_tokens"] == 1

    def test_batch_rejects_prompt_or_mask_truncation_instead_of_hiding_it(self):
        accumulated = AccumulatedData(
            prompts=[torch.tensor([1, 2, 3])],
            responses=[torch.tensor([4, 5])],
            traj_mask=[torch.tensor([1])],
        )

        with pytest.raises(ValueError, match="response/mask lengths differ"):
            _batch_tensors_and_build_data_proto(
                accumulated,
                pad_token_id=0,
                max_prompt_length=3,
                max_response_length=3,
            )

        accumulated.traj_mask = [torch.tensor([1, 1])]
        with pytest.raises(ValueError, match="prompt storage width"):
            _batch_tensors_and_build_data_proto(
                accumulated,
                pad_token_id=0,
                max_prompt_length=2,
                max_response_length=3,
            )

    def test_explicit_sequence_cap_rejects_generic_overflow(self):
        episode = _make_episode(
            prompt_ids=[1, 2, 3, 4, 5],
            completion_ids=[6, 7, 8, 9, 10],
        )

        with pytest.raises(ValueError, match="sequence cap"):
            transform_episodes_to_dataproto(
                [episode],
                _make_mock_rollout_engine(),
                max_prompt_length=8,
                max_response_length=8,
                max_total_length=9,
            )

    def test_other_batch_fields_unchanged(self):
        """Adding logprobs should not affect existing batch fields."""
        episodes = [
            _make_episode(
                prompt_ids=[1, 2, 3],
                completion_ids=[4, 5, 6],
                logprobs=[-0.5, -0.3, -0.1],
            ),
        ]
        engine = _make_mock_rollout_engine()

        batch = transform_episodes_to_dataproto(episodes, engine, max_prompt_length=8, max_response_length=8)

        # All standard fields should still be present
        for key in ["input_ids", "attention_mask", "position_ids", "prompts", "responses", "response_mask", "traj_rewards", "step_rewards"]:
            assert key in batch.batch, f"Standard field '{key}' should be present"
