"""
Tests for the verl transform pipeline, focusing on rollout log probs propagation.

Verifies that log probs from ModelOutput are correctly carried through
_process_trajectory → AccumulatedData → _batch_tensors_and_build_data_proto → DataProto,
so that downstream importance sampling and bypass mode work.
"""

from unittest.mock import MagicMock

import pytest
import torch

from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine.rollout import ModelOutput
from rllm.parser import QwenChatTemplateParser
from rllm.trainer.algorithms.config import CompactFilteringConfig, TransformConfig
from rllm.trainer.algorithms.transform import transform_episodes_to_trajectory_groups
from rllm.trainer.verl.dataclass import AccumulatedData
from rllm.trainer.verl.transform import (
    _batch_tensors_and_build_data_proto,
    transform_episodes_to_dataproto,
)
from rllm.workflows.workflow import TerminationReason


def _make_mock_rollout_engine(pad_token_id: int = 0):
    """Create a mock VerlEngine with a tokenizer."""
    engine = MagicMock()
    engine.tokenizer.pad_token_id = pad_token_id
    engine.processor = None  # No multimodal processor
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
        )
        # Deliberately not prefixed by [1, 2, 3, 4].
        output_2 = ModelOutput(
            prompt_ids=[1, 2, 9, 4, 5],
            completion_ids=[6, 7],
            logprobs=[-0.3, -0.4],
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
