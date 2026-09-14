import base64
import json
import logging
import math
import uuid
from collections.abc import Mapping

import numpy as np
import torch
from verl.protocol import DataProto
from verl.utils.torch_functional import pad_sequence_to_length

from rllm.engine.rollout import VerlEngine
from rllm.parser import QwenChatTemplateParser
from rllm.trainer.verl.dataclass import AccumulatedData, ProcessedStepData
from rllm.types import Episode, Trajectory, TrajectoryGroup
from rllm.workflows.workflow import TerminationReason

logger = logging.getLogger(__name__)

_BEHAVIOR_SAMPLING_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
    "presence_penalty",
    "frequency_penalty",
)
_NEUTRAL_BEHAVIOR_PENALTIES = {
    "repetition_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
}

# These request fields are recorded by the model gateway because they can
# affect sampling.  The trainer currently reproduces only temperature,
# top-k/top-p, and neutral penalties.  Fields below are safe only at the
# listed neutral values; everything else must fail closed before the rollout
# log-probabilities are admitted to the loss.
_NEUTRAL_UNSUPPORTED_SAMPLING_CONTROLS = {
    "allowed_token_ids": (None, []),
    "bad_words": (None, []),
    "beam_search": (None, False),
    "best_of": (None, 1),
    "early_stopping": (None, False),
    "guided_choice": (None,),
    "guided_decoding_backend": (None,),
    "guided_grammar": (None,),
    "guided_json": (None,),
    "guided_regex": (None,),
    "ignore_eos": (None, False),
    "length_penalty": (None, 1, 1.0),
    "logit_bias": (None, {}),
    "logits_processors": (None, []),
    "min_p": (None, 0, 0.0),
    "min_tokens": (None, 0),
    "response_format": (None,),
    "structured_outputs": (None, {}),
    "typical_p": (None, 1, 1.0),
    "use_beam_search": (None, False),
}
_SAFE_SAMPLING_BOOKKEEPING_FIELDS = {
    "include_stop_str_in_output",
    "max_tokens",
    "parallel_tool_calls",
    "seed",
    "stop",
    "stop_token_ids",
    "truncate_prompt_tokens",
}


def _normalize_behavior_sampling_value(name: str, value) -> int | float:
    """Normalize one configured or observed sampling value for comparison."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not boolean")
    if name == "top_k":
        try:
            normalized = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an integer") from exc
        try:
            matches_integer = float(value) == normalized
        except (TypeError, ValueError, OverflowError):
            matches_integer = False
        if not matches_integer:
            raise ValueError(f"{name} must be an integer")
        return normalized

    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _mapping_at_path(root, *path: str) -> Mapping | None:
    current = root
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current if isinstance(current, Mapping) else None


def _resolve_expected_behavior_sampling_params(rollout_engine) -> dict[str, int | float]:
    """Resolve and cross-check the six behavior-policy sampling controls.

    ``VerlEngine.train_sampling_params`` is the primary source. The full rLLM
    train sampling config supplies controls (notably penalties) that VerlEngine
    does not copy into that convenience mapping. Values present in both sources
    must agree. Penalties must remain neutral until trainer recomputation gains
    matching history-dependent logits processors.
    """
    engine_params = getattr(rollout_engine, "train_sampling_params", None)
    if not isinstance(engine_params, Mapping):
        engine_params = {}
    config_params = _mapping_at_path(
        getattr(rollout_engine, "config", None),
        "rllm",
        "rollout",
        "train",
    )
    sources = (
        ("rollout_engine.train_sampling_params", engine_params),
        ("config.rllm.rollout.train", config_params or {}),
    )

    resolved: dict[str, int | float] = {}
    for name in _BEHAVIOR_SAMPLING_FIELDS:
        observed: list[tuple[str, int | float]] = []
        for source_name, source in sources:
            if name not in source:
                continue
            try:
                value = _normalize_behavior_sampling_value(name, source[name])
            except ValueError as exc:
                raise ValueError(
                    f"invalid behavior sampling configuration in {source_name}: {exc}"
                ) from exc
            observed.append((source_name, value))
        if not observed:
            raise ValueError(
                f"behavior-logprob provenance requires configured {name}"
            )
        if any(value != observed[0][1] for _, value in observed[1:]):
            details = ", ".join(
                f"{source_name}={value!r}" for source_name, value in observed
            )
            raise ValueError(
                f"behavior sampling configuration disagrees for {name}: {details}"
            )
        resolved[name] = observed[0][1]

    for name, neutral in _NEUTRAL_BEHAVIOR_PENALTIES.items():
        if resolved[name] != neutral:
            raise ValueError(
                f"behavior-logprob provenance requires neutral {name}={neutral!r}, "
                f"got {resolved[name]!r}"
            )
    return resolved


def _sampling_params_rejection_reason(
    actual,
    expected: Mapping[str, int | float],
    *,
    source: str,
) -> str | None:
    """Return a stable fail-closed reason for missing or mismatched controls."""
    if not isinstance(actual, Mapping):
        return f"missing_{source}_sampling_params"
    for name in _BEHAVIOR_SAMPLING_FIELDS:
        if name not in actual:
            return f"missing_{source}_sampling_param_{name}"
        try:
            value = _normalize_behavior_sampling_value(name, actual[name])
        except ValueError:
            return f"invalid_{source}_sampling_param_{name}"
        if value != expected[name]:
            return f"{source}_sampling_param_mismatch_{name}"
    for name, neutral_values in _NEUTRAL_UNSUPPORTED_SAMPLING_CONTROLS.items():
        if name in actual and actual[name] not in neutral_values:
            return f"unsupported_{source}_sampling_control_{name}"
    if "n" in actual and actual["n"] != 1:
        return f"unsupported_{source}_sampling_control_n"
    if "tool_choice" in actual and actual["tool_choice"] not in (None, "auto"):
        return f"unsupported_{source}_sampling_control_tool_choice"
    known_fields = (
        set(_BEHAVIOR_SAMPLING_FIELDS)
        | set(_NEUTRAL_UNSUPPORTED_SAMPLING_CONTROLS)
        | _SAFE_SAMPLING_BOOKKEEPING_FIELDS
        | {"n", "tool_choice"}
    )
    unknown_fields = sorted(set(actual) - known_fields)
    if unknown_fields:
        return f"unsupported_{source}_sampling_control_{unknown_fields[0]}"
    return None


def _extract_episode_behavior_sampling_params(episode: Episode) -> Mapping | None:
    """Return only EasySim's six distribution-defining root settings.

    ``generation_config`` also contains context-budget and transcript
    bookkeeping. Those values describe the episode but do not change token
    probabilities. Per-step gateway traces remain authoritative for detecting
    unsupported request controls such as ``min_p`` or guided decoding.
    """
    artifacts = getattr(episode, "artifacts", None)
    if not isinstance(artifacts, Mapping):
        return None
    generation_config = artifacts.get("generation_config")
    if not isinstance(generation_config, Mapping):
        return None
    if not any(name in generation_config for name in _BEHAVIOR_SAMPLING_FIELDS):
        return None
    return {
        name: generation_config[name]
        for name in _BEHAVIOR_SAMPLING_FIELDS
        if name in generation_config
    }


def _uses_qwen_cumulative_chat(trajectories, chat_parser) -> bool:
    if not isinstance(chat_parser, QwenChatTemplateParser):
        return False
    return any(
        (trajectory.metadata or {}).get("rllm_cumulative_chat") is not None
        for trajectory in trajectories
    )


def _requires_exact_processed_behavior_rows(rollout_engine) -> bool:
    """Return whether the trainer explicitly consumes processed behavior lps.

    ``rllm_cumulative_chat`` is present on ordinary EasySim trajectories too,
    so it must not by itself change the batch representation.  The actor-side
    behavior-logprob mode is the explicit correction contract shared with the
    loss implementation.
    """
    actor_config = _mapping_at_path(
        getattr(rollout_engine, "config", None),
        "actor_rollout_ref",
        "actor",
    )
    return (
        isinstance(actor_config, Mapping)
        and actor_config.get("behavior_logprobs_mode") == "processed_logprobs"
    )


def _validate_exact_processed_behavior_steps(
    steps: list,
    *,
    expected_sampling_params: Mapping[str, int | float],
    episode_sampling_params: Mapping | None = None,
) -> tuple[list[list[float]] | None, str | None]:
    """Validate exact per-turn behavior logprobs without re-rendering chat.

    Each accepted row uses the prompt and completion IDs captured directly
    from the inference request.  Unlike the cumulative-chat representation,
    this remains exact when a chat template rewrites old tool calls, thinking
    markers, Unicode, or JSON formatting on a later turn.
    """
    episode_rejection = (
        _sampling_params_rejection_reason(
            episode_sampling_params,
            expected_sampling_params,
            source="episode",
        )
        if episode_sampling_params is not None
        else None
    )
    if episode_rejection is not None:
        return None, episode_rejection

    validated: list[list[float]] = []
    for step in steps:
        model_output = step.model_output
        if getattr(model_output, "logprobs_mode", None) != "processed_logprobs":
            return None, "logprobs_mode_not_processed"
        sampling_rejection = _sampling_params_rejection_reason(
            getattr(model_output, "behavior_sampling_params", None),
            expected_sampling_params,
            source="step",
        )
        if sampling_rejection is not None:
            return None, sampling_rejection

        prompt_ids = getattr(model_output, "prompt_ids", None)
        completion_ids = getattr(model_output, "completion_ids", None)
        logprobs = getattr(model_output, "logprobs", None)
        if prompt_ids is None:
            return None, "missing_prompt_tokens"
        if completion_ids is None or len(completion_ids) == 0:
            return None, "missing_completion_tokens"
        if logprobs is None or len(logprobs) == 0:
            return None, "missing_logprobs"
        try:
            [int(token) for token in list(prompt_ids)]
        except (TypeError, ValueError):
            return None, "invalid_prompt_token"
        try:
            completion = [int(token) for token in list(completion_ids)]
        except (TypeError, ValueError):
            return None, "invalid_completion_token"
        if len(logprobs) != len(completion):
            return None, "logprob_length_mismatch"
        try:
            values = [float(value) for value in logprobs]
        except (TypeError, ValueError):
            return None, "invalid_logprob_value"
        if not all(math.isfinite(value) for value in values):
            return None, "nonfinite_logprob"
        validated.append(values)

    return validated, None


def _align_qwen_cumulative_behavior_logprobs(
    prompt: torch.Tensor,
    response: torch.Tensor,
    mask: torch.Tensor,
    steps: list,
    *,
    expected_sampling_params: Mapping[str, int | float],
    episode_sampling_params: Mapping | None = None,
) -> tuple[list[float] | None, str | None]:
    """Align served action log-probabilities to one canonical Qwen row.

    The cumulative Qwen transform re-renders structured chat messages. A
    served log-probability is valid in that row only when the exact token
    prefix seen by the rollout engine and the exact sampled completion both
    occur at the expected canonical positions. This function intentionally
    has no suffix/fuzzy alignment: a prompt mismatch changes the conditional
    distribution, even if the completion tokens happen to match.

    Non-action tokens (tool calls/results and other context) retain a 0.0
    placeholder and are ignored by ``response_mask``. Any incomplete,
    non-finite, overlapping, or partial alignment rejects the entire row.
    """

    episode_rejection = (
        _sampling_params_rejection_reason(
            episode_sampling_params,
            expected_sampling_params,
            source="episode",
        )
        if episode_sampling_params is not None
        else None
    )
    if episode_rejection is not None:
        return None, episode_rejection

    prompt_ids = [int(token) for token in prompt.tolist()]
    response_ids = [int(token) for token in response.tolist()]
    response_mask = [int(value) for value in mask.tolist()]
    if len(response_ids) != len(response_mask):
        return None, "canonical_response_mask_length_mismatch"
    if any(value not in (0, 1) for value in response_mask):
        return None, "canonical_response_mask_not_binary"

    canonical_ids = prompt_ids + response_ids
    response_offset = len(prompt_ids)
    aligned = [0.0] * len(response_ids)
    covered = [False] * len(response_ids)

    for step in steps:
        model_output = step.model_output
        if getattr(model_output, "logprobs_mode", None) != "processed_logprobs":
            return None, "logprobs_mode_not_processed"
        sampling_rejection = _sampling_params_rejection_reason(
            getattr(model_output, "behavior_sampling_params", None),
            expected_sampling_params,
            source="step",
        )
        if sampling_rejection is not None:
            return None, sampling_rejection
        served_prompt = model_output.prompt_ids
        served_completion = model_output.completion_ids
        served_logprobs = model_output.logprobs

        if served_prompt is None:
            return None, "missing_prompt_tokens"
        if served_completion is None or len(served_completion) == 0:
            return None, "missing_completion_tokens"
        if served_logprobs is None or len(served_logprobs) == 0:
            return None, "missing_logprobs"

        try:
            served_prompt_ids = [int(token) for token in list(served_prompt)]
        except (TypeError, ValueError):
            return None, "invalid_prompt_token"
        try:
            served_completion_ids = [int(token) for token in list(served_completion)]
        except (TypeError, ValueError):
            return None, "invalid_completion_token"
        if len(served_logprobs) != len(served_completion_ids):
            return None, "logprob_length_mismatch"
        try:
            logprobs = [float(value) for value in served_logprobs]
        except (TypeError, ValueError):
            return None, "invalid_logprob_value"
        if not all(math.isfinite(value) for value in logprobs):
            return None, "nonfinite_logprob"

        prompt_end = len(served_prompt_ids)
        if prompt_end > len(canonical_ids) or canonical_ids[:prompt_end] != served_prompt_ids:
            return None, "prompt_token_mismatch"

        completion_end = prompt_end + len(served_completion_ids)
        if completion_end > len(canonical_ids) or canonical_ids[prompt_end:completion_end] != served_completion_ids:
            return None, "completion_token_mismatch"

        action_start = prompt_end - response_offset
        action_end = completion_end - response_offset
        if action_start < 0 or action_end > len(response_ids):
            return None, "completion_outside_response"
        if any(value != 1 for value in response_mask[action_start:action_end]):
            return None, "completion_not_fully_action_masked"
        if any(covered[action_start:action_end]):
            return None, "overlapping_completion_spans"

        aligned[action_start:action_end] = logprobs
        covered[action_start:action_end] = [True] * len(logprobs)

    for is_action, is_covered in zip(response_mask, covered, strict=True):
        if bool(is_action) != is_covered:
            return None, "incomplete_action_coverage"

    return aligned, None


def _pad_sequence_batch(sequences: list[torch.Tensor], pad_token_id: int, max_length: int, left_pad: bool = True) -> torch.Tensor:
    """Pads a list of sequences to a maximum length.

    Args:
        sequences: List of sequences to pad.
        pad_token_id: The token ID to use for padding.
        max_length: The maximum length to pad to.
        left_pad: Whether to pad on the left or right.
    Returns:
        torch.Tensor: The padded sequences.
    """
    if left_pad:
        rev_sequences = [torch.flip(seq, dims=[0]) for seq in sequences]
        batch = torch.nn.utils.rnn.pad_sequence(rev_sequences, batch_first=True, padding_value=pad_token_id).flip(dims=[1])
    else:
        batch = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=pad_token_id)

    batch = pad_sequence_to_length(batch, max_length, pad_token_id, left_pad=left_pad)
    # additional truncation check
    batch = batch[:, -max_length:] if left_pad else batch[:, :max_length]
    return batch


def _retrieve_batch_attention_masks(batch: torch.Tensor, pad_token_id: int, max_length: int) -> torch.Tensor:
    """Retrieves the attention masks for a batch of prompts/responses.

    Note that in original implementation, this operation is padding-aware, i.e. it has DIFFERENT behavior for left-pad (prompts)
    and right-pad (responses) sequences. This is to ensure compatibility with the `input_ids` constructed with results from function `_pad_sequence_batch`.

    The current version simply uses the constructed `promits_batch` (`responses_batch`) instead of the original `prompts` (`responses`) lengths.
    """
    assert len(batch.shape) == 2, f"batch must be a 2D tensor, but got {batch.shape}"
    assert batch.shape[1] == max_length, f"input batch must have been padded to {max_length}, but got {batch.shape[1]}"
    return batch != pad_token_id


def _build_step_and_trajectory_rewards(step_rewards: list[float], trajectory_rewards: list[float], responses_batch: torch.Tensor, responses: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Builds the step and trajectory rewards for a batch of prompts/responses.

    Args:
        step_rewards: List of step rewards.
        trajectory_rewards: List of trajectory rewards.
        shape: Shape of the step and trajectory rewards. Should be (bs, max_response_length).
        responses: List of responses. Should be a list of tensors with shape (seq_len,).
    Returns:
        tuple[torch.Tensor, torch.Tensor]: The step and trajectory rewards.
    """
    assert len(step_rewards) == len(trajectory_rewards), "step_rewards and trajectory_rewards must have the same length"

    step_rewards_batch = torch.zeros(responses_batch.shape, dtype=torch.float32)  # shape: [bs, max_response_length]
    trajectory_rewards_batch = torch.zeros(responses_batch.shape, dtype=torch.float32)  # shape: [bs, max_response_length]
    for i, (step_reward, trajectory_reward, response) in enumerate(zip(step_rewards, trajectory_rewards, responses, strict=False)):
        resp_len = len(response)
        if resp_len > 0 and resp_len <= responses_batch.shape[1]:
            step_rewards_batch[i, resp_len - 1] = step_reward
            trajectory_rewards_batch[i, resp_len - 1] = trajectory_reward

    return step_rewards_batch, trajectory_rewards_batch


def _build_per_step_advantages(response_mask: torch.Tensor, advantages: list[float] | list[list[float]]) -> torch.Tensor:
    """Builds the per-step advantages for a batch of prompts/responses."""
    assert response_mask.shape[0] == len(advantages), "response_mask and advantages must have the same length"
    if isinstance(advantages[0], list):
        # verticle stack the advantages (which implicitly ensures that all advantages have the same length)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32)
    else:
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32).unsqueeze(-1)
    return advantages_tensor * response_mask


def _handle_multimodal_position_ids(processor, input_ids: torch.Tensor, attention_mask: torch.Tensor, multi_modal_inputs: list[dict]) -> torch.Tensor:
    """Handle multimodal position ids calculation. Borrowed from verl.utils.dataset.rl_dataset.py

    Args:
        processor: The multimodal processor (e.g., Qwen2VLProcessor or Qwen3VLProcessor).
        input_ids: Tensor of input token IDs with shape (batch_size, seq_length).
        attention_mask: Tensor of attention masks with shape (batch_size, seq_length).
        multi_modal_inputs: List of dicts containing multimodal inputs per batch item.
    Returns:
        torch.Tensor: Position IDs tensor with shape (batch_size, 4, seq_length) for Qwen-VL models.
    """
    batch_size = input_ids.shape[0]
    position_ids_list = []

    if processor is not None and "Qwen2VLImageProcessor" in processor.image_processor.__class__.__name__:
        # qwen-vl mrope
        if "Qwen3VLProcessor" in processor.__class__.__name__:
            from verl.models.transformers.qwen3_vl import get_rope_index
        else:
            from verl.models.transformers.qwen2_vl import get_rope_index

        for i in range(batch_size):
            model_inputs = multi_modal_inputs[i] if i < len(multi_modal_inputs) else {}
            vision_position_ids = get_rope_index(
                processor,
                input_ids=input_ids[i],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask[i],
            )  # (3, seq_length)
            valid_mask = attention_mask[i].bool()
            text_position_ids = torch.ones((1, len(input_ids[i])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids_list.append(torch.cat((text_position_ids, vision_position_ids), dim=0))  # (4, seq_length)

    else:
        # Fallback: should not reach here if called correctly
        raise ValueError(f"Unsupported processor type: {processor.__class__.__name__ if processor else None}")

    # Stack all position_ids to form batch: (batch_size, 4, seq_length)
    position_ids = torch.stack(position_ids_list, dim=0)
    return position_ids


def _batch_tensors_and_build_data_proto(
    accumulated: AccumulatedData,
    pad_token_id: int,
    max_prompt_length: int,
    max_response_length: int,
    processor=None,
    *,
    max_total_length: int | None = None,
) -> "DataProto":
    """Batches the tensors from an AccumulatedData.

    Args:
        accumulated: AccumulatedData to batch the tensors from.
        pad_token_id: The token ID to use for padding.
        max_prompt_length: The maximum length to pad the prompts to.
        max_response_length: The maximum length to pad the responses to.
        max_total_length: Optional cap on prompt plus response non-padding
            tokens. When set, exceeding it is an error rather than a silent
            padding-time truncation.
        processor: Optional multimodal processor for handling position IDs (e.g., Qwen2VLProcessor).
    Returns:
        DataProto: The DataProto built from the AccumulatedData.
    """
    for row, (prompt, response, mask) in enumerate(
        zip(
            accumulated.prompts,
            accumulated.responses,
            accumulated.traj_mask,
            strict=True,
        )
    ):
        if prompt.numel() > max_prompt_length:
            raise ValueError(f"trainer row {row} prompt has {prompt.numel()} tokens, exceeding the {max_prompt_length}-token prompt storage width")
        if response.numel() != mask.numel():
            raise ValueError(f"trainer row {row} response/mask lengths differ: {response.numel()} != {mask.numel()}")
        if response.numel() > max_response_length:
            raise ValueError(f"trainer row {row} response has {response.numel()} tokens, exceeding the {max_response_length}-token response storage width")
        if max_total_length is not None and prompt.numel() + response.numel() > max_total_length:
            raise ValueError(f"trainer row {row} has {prompt.numel() + response.numel()} non-padding tokens, exceeding the {max_total_length}-token sequence cap")

    prompts_batch = _pad_sequence_batch(accumulated.prompts, pad_token_id, max_prompt_length, left_pad=True)  # shape: [bs, max_prompt_length]
    responses_batch = _pad_sequence_batch(accumulated.responses, pad_token_id, max_response_length, left_pad=False)  # shape: [bs, max_response_length]
    input_ids = torch.concat([prompts_batch, responses_batch], dim=1)  # shape: [bs, max_prompt_length + max_response_length]

    prompts_mask = _retrieve_batch_attention_masks(prompts_batch, pad_token_id, max_prompt_length)
    responses_mask = _retrieve_batch_attention_masks(responses_batch, pad_token_id, max_response_length)
    attention_mask = torch.concat([prompts_mask, responses_mask], dim=1)  # shape: [bs, max_prompt_length + max_response_length]

    # Handle position_ids: use multimodal handler if processor is available
    if processor is not None and hasattr(processor, "image_processor") and "Qwen2VLImageProcessor" in processor.image_processor.__class__.__name__:
        position_ids = _handle_multimodal_position_ids(
            processor=processor,
            input_ids=input_ids,
            attention_mask=attention_mask,
            multi_modal_inputs=accumulated.multi_modal_inputs,
        )
    else:
        position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask  # shape: [bs, max_prompt_length + max_response_length]

    traj_mask = _pad_sequence_batch(accumulated.traj_mask, 0, max_response_length, left_pad=False)  # shape: [bs, max_response_length]

    step_rewards_batch, traj_rewards_batch = _build_step_and_trajectory_rewards(
        accumulated.step_rewards, accumulated.traj_rewards, responses_batch, accumulated.responses
    )  # shape: [bs, max_response_length]

    non_tensors = {
        "episode_ids": np.array(accumulated.episode_ids),  # unique identifier for each rollout
        "trajectory_ids": np.array(accumulated.trajectory_ids),
        "step_ids": np.array(accumulated.step_ids),
        "batch_ids": np.array([str(uuid.uuid4())] * len(accumulated.trajectory_ids)),
        "step_nums": np.array(accumulated.step_nums),
        "is_correct": np.array(accumulated.is_correct),
        "termination_reasons": np.array([x.value for x in accumulated.termination_reasons]),
        "metrics": np.array(accumulated.metrics, dtype=object),
        "is_valid": np.ones(len(accumulated.trajectory_ids), dtype=bool),
        "is_last_step": np.array(accumulated.is_last_step),
        # The padding is done after the transform (in `_pad_dataproto_to_world_size`), so we simply set all to False here
        "is_pad_step": np.zeros(len(accumulated.trajectory_ids), dtype=bool),
        # Per-row trajectory role name (for per-role loss routing)
        "group_roles": np.array(accumulated.group_roles, dtype=object),
        "advantage_weights": np.array(
            accumulated.advantage_weights, dtype=np.float32
        ),
    }

    # Include multi_modal_inputs in non_tensors if any are present
    if any(mm_inputs for mm_inputs in accumulated.multi_modal_inputs):
        non_tensors["multi_modal_inputs"] = np.array(accumulated.multi_modal_inputs, dtype=object)

    tensors = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "prompts": prompts_batch,
        "responses": responses_batch,
        "response_mask": traj_mask,
        "traj_rewards": traj_rewards_batch,
        "step_rewards": step_rewards_batch,
    }

    meta_info = {
        "repeat_counts": accumulated.repeat_counts,
    }

    # Include rollout log probs if available (enables importance sampling & bypass mode)
    if accumulated.rollout_logprobs and len(accumulated.rollout_logprobs) == len(accumulated.responses):
        rollout_logprobs_batch = _pad_sequence_batch(accumulated.rollout_logprobs, 0, max_response_length, left_pad=False)
        tensors["rollout_log_probs"] = rollout_logprobs_batch
        if (
            accumulated.behavior_logprobs_provenance is not None
            and accumulated.behavior_logprobs_aligned_rows
            == len(accumulated.responses)
        ):
            meta_info["rollout_log_probs_provenance"] = {
                **accumulated.behavior_logprobs_provenance,
                "rows": len(accumulated.responses),
            }

    # Routed experts for R3 router replay. Each routing tensor covers
    # (prompt + response) tokens for its row (last step's routing in the
    # cumulative segment). Place at [max_prompt_length - len(prompt) :
    # max_prompt_length + len(response)] to match verl's agent_loop layout —
    # left-padded prompt region, right-padded response region, zeros elsewhere.
    if accumulated.routing_matrices and len(accumulated.routing_matrices) == len(accumulated.responses):
        _, num_layers, topk = accumulated.routing_matrices[0].shape
        ref_dtype = accumulated.routing_matrices[0].dtype
        bs = len(accumulated.routing_matrices)
        total_length = max_prompt_length + max_response_length
        routed_experts = torch.zeros(bs, total_length, num_layers, topk, dtype=ref_dtype)
        for i, r in enumerate(accumulated.routing_matrices):
            len_p = min(accumulated.prompts[i].shape[0], max_prompt_length)
            start_pos = max_prompt_length - len_p
            end_pos = min(start_pos + r.shape[0], total_length)
            routed_experts[i, start_pos:end_pos] = r[: end_pos - start_pos]
        tensors["routed_experts"] = routed_experts

    return DataProto.from_dict(
        tensors=tensors,
        non_tensors=non_tensors,
        meta_info=meta_info,
    )


def _decode_routing_matrices(encoded: list[str] | None) -> torch.Tensor | None:
    """Decode the [shape_header_json, base64_blob] wire format into a (length, num_layers, topk) tensor."""
    if not encoded or len(encoded) != 2:
        return None
    header = json.loads(encoded[0])
    num_layers, topk = header["shape"]
    dtype = np.dtype(header["dtype"])
    arr = np.frombuffer(base64.b64decode(encoded[1]), dtype=dtype).reshape(-1, num_layers, topk)
    return torch.from_numpy(arr.copy())


def _process_trajectory(
    trajectory: Trajectory,
    task_id: str,
    accumulated: AccumulatedData,
    chat_parser=None,
    *,
    max_prompt_length: int | None = None,
    max_total_length: int | None = None,
    expected_behavior_sampling_params: Mapping[str, int | float] | None = None,
    episode_behavior_sampling_params: Mapping | None = None,
    exact_processed_behavior_rows: bool = False,
) -> int:
    """Processes a trajectory and returns an AccumulatedData.

    Multi-turn trajectories whose steps form a cumulative-prefix chain
    (each step's prompt is an extension of the previous step's full sequence,
    e.g. a ReAct/tool-call agent that appends tool messages and assistant
    responses to a growing message list) are merged into a SINGLE row whose
    response is the concatenation of [A0, obs1, A1, obs2, A2, ...] with
    response_mask = 1 only on action tokens (the model's outputs at each
    turn) and 0 on observation tokens (tool messages, system messages
    inserted between turns).

    This mirrors Tinker's ``trajectory_to_datums`` representation. Combined
    with ``loss_agg_mode=seq-mean-token-mean`` it gives per-trajectory
    equal-weighted gradients regardless of step count: a 6-turn rollout
    contributes the same to the loss as a 2-turn rollout, which matches
    Tinker's per-Datum aggregation. Without merging, verl emits one row per
    step and per-trajectory weight scales with step count.

    A step that is *not* a prefix-extension of the running segment (e.g.
    the agent reset its context mid-trajectory) closes the current segment
    and starts a new one — the trajectory then contributes multiple rows.
    For typical agents this never fires, so the common case is one row per
    trajectory.

    Args:
        trajectory: Trajectory to process.
        task_id: Task identifier corresponding to the episode.
        accumulated: AccumulatedData to process the trajectory into.
    Returns:
        Number of rows emitted to ``accumulated`` (typically 1; >1 only if
        the trajectory's steps couldn't all be prefix-merged).
    """
    name = trajectory.name
    trajectory_id = f"{task_id}_{name}"
    if len(trajectory.steps) == 0:
        print(f"Trajectory {trajectory_id} has no steps, skipping")
        return 0

    traj_reward = 0.0 if trajectory.reward is None else trajectory.reward

    # Drop steps without valid model_output up-front; the merge logic below
    # assumes every entry has prompt_ids and completion_ids.
    valid_steps = []
    for step_idx, step in enumerate(trajectory.steps):
        if step.model_output is None or step.model_output.prompt_ids is None:
            logger.warning(f"Step {step_idx} in trajectory {trajectory_id} has no valid model_output, skipping")
            continue
        valid_steps.append(step)

    if not valid_steps:
        return 0

    cumulative_config = (trajectory.metadata or {}).get("rllm_cumulative_chat")
    if cumulative_config is True:
        cumulative_config = {}
    if cumulative_config is not None and not isinstance(cumulative_config, dict):
        raise ValueError("trajectory metadata rllm_cumulative_chat must be true or a mapping")

    # ------------------------------------------------------------------
    # Walk steps and merge prefix-extending steps into segments.
    # ------------------------------------------------------------------
    # A *segment* is one merged row in the batch. We accumulate response
    # tokens and a parallel mask:
    #   response = [action_tokens for step0,
    #               delta_obs_for_step1, action_tokens_step1,
    #               delta_obs_for_step2, action_tokens_step2, ...]
    #   mask     = [1*N_act0,
    #               0*N_obs1, 1*N_act1,
    #               0*N_obs2, 1*N_act2, ...]
    # The segment's prompt is the *initial* prompt of the first step in
    # that segment. ``full_seq`` tracks prompt+all-action-and-obs tokens
    # so we can detect prefix-extension on the next step.

    def _new_segment(step):
        prompt = list(step.model_output.prompt_ids)
        action = list(step.model_output.completion_ids)
        action_lp = list(step.model_output.logprobs or [])
        # If logprobs missing/short, pad to action length with zeros so
        # accumulator lists stay aligned. add_step skips logprobs entirely
        # when the list is empty, but we keep parity with action_tokens.
        if action_lp and len(action_lp) != len(action):
            action_lp = list(action_lp) + [0.0] * (len(action) - len(action_lp))
        return {
            "prompt": prompt,
            "response": list(action),
            "mask": [1] * len(action),
            "logprobs": list(action_lp),
            "full_seq": list(prompt) + list(action),
            "multi_modal": step.model_output.multi_modal_inputs or {},
            # Hold the latest step that produced routing in this segment. Each step's
            # routing covers (step.prompt + step.action), and the segment is cumulative
            # by construction, so the last step's routing covers seg["full_seq"]. We
            # decode at emit time rather than per-step.
            "last_routing_step": step if step.routing_matrices else None,
        }

    def _emit(seg):
        prompt_t = torch.tensor(seg["prompt"], dtype=torch.long)
        response_t = torch.tensor(seg["response"], dtype=torch.long)
        mask_t = torch.tensor(seg["mask"], dtype=torch.long)
        last_routing_step = seg["last_routing_step"]
        routing_t = _decode_routing_matrices(last_routing_step.routing_matrices) if last_routing_step is not None else None
        # step_id is keyed by trajectory.uid (no per-segment suffix). All
        # segments of one trajectory share the same scalar advantage from
        # collect_reward_and_advantage_from_trajectory_groups (broadcast
        # mode), so collisions across segments are harmless: the dict in
        # update_dataproto_with_advantages would write the same value
        # for either key.
        step_data = ProcessedStepData(
            prompt=prompt_t,
            response=response_t,
            mask=mask_t,
            step_reward=traj_reward,
            step_id=trajectory.uid,
            multi_modal_inputs=seg["multi_modal"],
            advantage=None,
            logprobs=seg["logprobs"] if seg["logprobs"] else None,
            routing_matrices=routing_t,
        )
        accumulated.add_step(
            step_data=step_data,
            trajectory_id=trajectory_id,
            traj_reward=traj_reward,
            step_num=1,
            is_last=True,
            group_role=name,
        )

    def _emit_qwen_cumulative_chat() -> bool:
        """Use rLLM's chat masker when Qwen re-rendering breaks token prefixes."""
        if cumulative_config is None or not isinstance(chat_parser, QwenChatTemplateParser):
            return False
        if expected_behavior_sampling_params is None:
            raise ValueError(
                "rllm_cumulative_chat behavior logprobs require configured sampling parameters"
            )
        if len(valid_steps) != len(trajectory.steps) or not trajectory.is_cumulative():
            raise ValueError("rllm_cumulative_chat requires a complete cumulative message history")

        final_messages = valid_steps[-1].chat_completions
        assistant_count = sum(message.get("role") == "assistant" for message in final_messages)
        if not final_messages or assistant_count != len(valid_steps):
            raise ValueError("rllm_cumulative_chat requires one generated assistant message per model step")

        prompt, response, mask = chat_parser.tokenize_and_mask_cumulative(final_messages, tools=cumulative_config.get("tools"))
        if response.numel() == 0 or mask.sum().item() == 0:
            raise ValueError("rllm cumulative chat conversion produced no assistant tokens")

        aligned_logprobs, rejection_reason = _align_qwen_cumulative_behavior_logprobs(
            prompt,
            response,
            mask,
            valid_steps,
            expected_sampling_params=expected_behavior_sampling_params,
            episode_sampling_params=episode_behavior_sampling_params,
        )
        if rejection_reason is None:
            accumulated.behavior_logprobs_aligned_rows += 1
            provenance = {
                "logprobs_mode": "processed_logprobs",
                "sampling_params": dict(expected_behavior_sampling_params),
            }
            if accumulated.behavior_logprobs_provenance is None:
                accumulated.behavior_logprobs_provenance = provenance
            elif accumulated.behavior_logprobs_provenance != provenance:
                raise ValueError(
                    "behavior-logprob provenance changed within one trainer batch"
                )
        else:
            accumulated.behavior_logprobs_rejected_rows += 1
            accumulated.behavior_logprobs_rejection_reasons[rejection_reason] = (
                accumulated.behavior_logprobs_rejection_reasons.get(rejection_reason, 0) + 1
            )
            logger.warning(
                "Rejecting Qwen cumulative behavior logprobs for trajectory %s: %s",
                trajectory_id,
                rejection_reason,
            )

        if max_prompt_length is None:
            raise ValueError("rllm_cumulative_chat requires an explicit prompt-token limit")
        if prompt.numel() > max_prompt_length:
            raise ValueError(f"rllm cumulative prompt has {prompt.numel()} tokens, exceeding the {max_prompt_length}-token prompt storage width")

        declared_limit = cumulative_config.get("max_total_length")
        if declared_limit is None and max_total_length is None:
            raise ValueError("rllm_cumulative_chat requires an explicit total-token limit")
        if declared_limit is None:
            declared_limit = max_total_length
        if not isinstance(declared_limit, int) or isinstance(declared_limit, bool) or declared_limit <= 0:
            raise ValueError("rllm_cumulative_chat max_total_length must be a positive integer")
        sequence_limit = min(max_total_length, declared_limit) if max_total_length is not None else declared_limit
        available_response_tokens = sequence_limit - prompt.numel()
        if available_response_tokens <= 0:
            raise ValueError(f"rllm cumulative prompt leaves no response room in the {sequence_limit}-token trainer context")
        if response.numel() > available_response_tokens:
            clipped_tokens = response.numel() - available_response_tokens
            logger.warning(
                "Clipping %d tokens from Qwen cumulative response to fit %d-token trainer context",
                clipped_tokens,
                sequence_limit,
            )
            response = response[:available_response_tokens]
            mask = mask[:available_response_tokens]
            if aligned_logprobs is not None:
                aligned_logprobs = aligned_logprobs[:available_response_tokens]
            accumulated.context_clipped_rows += 1
            accumulated.context_clipped_tokens += clipped_tokens
            if response.numel() == 0 or mask.sum().item() == 0:
                raise ValueError("rllm cumulative context clipping removed all assistant tokens")

        accumulated.add_step(
            step_data=ProcessedStepData(
                prompt=prompt,
                response=response,
                mask=mask,
                step_reward=traj_reward,
                step_id=trajectory.uid,
                multi_modal_inputs={},
                advantage=None,
                # Retain served behavior-policy logprobs only when exact
                # prompt and completion token alignment was proven above.
                logprobs=aligned_logprobs,
                routing_matrices=None,
            ),
            trajectory_id=trajectory_id,
            traj_reward=traj_reward,
            step_num=1,
            is_last=True,
            group_role=name,
        )
        return True

    def _emit_exact_processed_behavior_rows() -> int:
        """Emit one exact served prompt/completion row for every model turn."""
        if expected_behavior_sampling_params is None:
            raise ValueError(
                "processed behavior logprobs require configured sampling parameters"
            )
        if len(valid_steps) != len(trajectory.steps):
            raise ValueError(
                "processed behavior logprobs require a model output for every turn"
            )

        validated_logprobs, rejection_reason = (
            _validate_exact_processed_behavior_steps(
                valid_steps,
                expected_sampling_params=expected_behavior_sampling_params,
                episode_sampling_params=episode_behavior_sampling_params,
            )
        )
        total_action_tokens = sum(
            len(step.model_output.completion_ids) for step in valid_steps
        )
        if total_action_tokens <= 0:
            raise ValueError(
                "processed behavior logprobs require at least one completion token"
            )
        if rejection_reason is None:
            assert validated_logprobs is not None
            accumulated.behavior_logprobs_aligned_rows += len(valid_steps)
            provenance = {
                "logprobs_mode": "processed_logprobs",
                "sampling_params": dict(expected_behavior_sampling_params),
            }
            if accumulated.behavior_logprobs_provenance is None:
                accumulated.behavior_logprobs_provenance = provenance
            elif accumulated.behavior_logprobs_provenance != provenance:
                raise ValueError(
                    "behavior-logprob provenance changed within one trainer batch"
                )
        else:
            accumulated.behavior_logprobs_rejected_rows += len(valid_steps)
            accumulated.behavior_logprobs_rejection_reasons[rejection_reason] = (
                accumulated.behavior_logprobs_rejection_reasons.get(
                    rejection_reason, 0
                )
                + len(valid_steps)
            )
            logger.warning(
                "Rejecting exact Qwen behavior logprobs for trajectory %s (%d rows): %s",
                trajectory_id,
                len(valid_steps),
                rejection_reason,
            )

        for index, step in enumerate(valid_steps):
            output = step.model_output
            prompt = torch.tensor(list(output.prompt_ids), dtype=torch.long)
            response = torch.tensor(list(output.completion_ids), dtype=torch.long)
            mask = torch.ones(response.numel(), dtype=torch.long)
            routing = (
                _decode_routing_matrices(step.routing_matrices)
                if step.routing_matrices is not None
                else None
            )
            accumulated.add_step(
                step_data=ProcessedStepData(
                    prompt=prompt,
                    response=response,
                    mask=mask,
                    step_reward=traj_reward,
                    step_id=trajectory.uid,
                    multi_modal_inputs=output.multi_modal_inputs or {},
                    advantage=None,
                    # seq-mean-token-mean averages each row independently.
                    # Weight turn rows by their action-token share so their
                    # sum exactly equals the former one-row trajectory mean.
                    advantage_weight=response.numel() / total_action_tokens,
                    logprobs=(
                        validated_logprobs[index]
                        if validated_logprobs is not None
                        else None
                    ),
                    routing_matrices=routing,
                ),
                trajectory_id=trajectory_id,
                traj_reward=traj_reward,
                step_num=len(valid_steps),
                is_last=index == len(valid_steps) - 1,
                group_role=name,
            )
        return len(valid_steps)

    # Qwen may happen to produce a literal token-prefix extension on some
    # turns and not others.  The opt-in contract is canonical cumulative-chat
    # tokenization, so apply it deterministically rather than only as a
    # fallback after prefix matching fails.
    if (
        exact_processed_behavior_rows
        and cumulative_config is not None
        and isinstance(chat_parser, QwenChatTemplateParser)
    ):
        return _emit_exact_processed_behavior_rows()

    if cumulative_config is not None and isinstance(chat_parser, QwenChatTemplateParser):
        if _emit_qwen_cumulative_chat():
            return 1

    seg = _new_segment(valid_steps[0])
    segments_emitted = 0
    for step in valid_steps[1:]:
        prompt_ids = list(step.model_output.prompt_ids)
        if len(prompt_ids) >= len(seg["full_seq"]) and prompt_ids[: len(seg["full_seq"])] == seg["full_seq"]:
            # Cumulative — extend the current segment.
            delta_obs = prompt_ids[len(seg["full_seq"]) :]
            action = list(step.model_output.completion_ids)
            action_lp = list(step.model_output.logprobs or [])
            if action_lp and len(action_lp) != len(action):
                action_lp = list(action_lp) + [0.0] * (len(action) - len(action_lp))

            seg["response"].extend(delta_obs)
            seg["response"].extend(action)
            seg["mask"].extend([0] * len(delta_obs))
            seg["mask"].extend([1] * len(action))
            seg["logprobs"].extend([0.0] * len(delta_obs))
            seg["logprobs"].extend(action_lp)
            seg["full_seq"].extend(delta_obs)
            seg["full_seq"].extend(action)

            if step.routing_matrices is not None:
                seg["last_routing_step"] = step
        else:
            if segments_emitted == 0 and _emit_qwen_cumulative_chat():
                return 1
            # Non-cumulative — close out current segment, start a new one.
            _emit(seg)
            segments_emitted += 1
            seg = _new_segment(step)

    _emit(seg)
    segments_emitted += 1
    return segments_emitted


def _process_episode(
    episode: Episode,
    task_id: str,
    accumulated: AccumulatedData,
    chat_parser=None,
    *,
    max_prompt_length: int | None = None,
    max_total_length: int | None = None,
    expected_behavior_sampling_params: Mapping[str, int | float] | None = None,
    exact_processed_behavior_rows: bool = False,
) -> int:
    """Processes an episode and returns an AccumulatedData.

    Args:
        episode: Episode to process.
        task_id: Task identifier corresponding to the episode.
        accumulated: AccumulatedData to process the episode into.
    Returns:
        repeated_count: The total steps in this episode.
    """
    total_steps = 0

    if all(len(trajectory.steps) == 0 for trajectory in episode.trajectories):
        # termination hits before an agent finishes it's first step
        # (e.g., the initial prompt exceeds max_prompt_length or a timeout occurs)
        # we delete the episode from the batch by setting repeat_counts to 0
        print(f"Episode {episode.id} has no valid trajectories, dropping it from the batch")
        return 0

    episode_behavior_sampling_params = _extract_episode_behavior_sampling_params(
        episode
    )
    for trajectory in episode.trajectories:
        n_steps = _process_trajectory(
            trajectory,
            task_id,
            accumulated,
            chat_parser=chat_parser,
            max_prompt_length=max_prompt_length,
            max_total_length=max_total_length,
            expected_behavior_sampling_params=expected_behavior_sampling_params,
            episode_behavior_sampling_params=episode_behavior_sampling_params,
            exact_processed_behavior_rows=exact_processed_behavior_rows,
        )
        total_steps += n_steps

    # Extend episode-level data for all steps in this episode
    accumulated.episode_ids.extend([episode.id] * total_steps)
    accumulated.is_correct.extend([episode.is_correct] * total_steps)
    termination_reason = episode.termination_reason if episode.termination_reason is not None else TerminationReason.UNKNOWN
    accumulated.termination_reasons.extend([termination_reason] * total_steps)
    accumulated.metrics.extend([episode.metrics] * total_steps)

    return total_steps


def _process_trajectory_group(
    trajectory_group: TrajectoryGroup,
    task_id: str,
    accumulated: AccumulatedData,
    chat_parser=None,
    *,
    max_prompt_length: int | None = None,
    max_total_length: int | None = None,
    expected_behavior_sampling_params: Mapping[str, int | float] | None = None,
    exact_processed_behavior_rows: bool = False,
) -> int:
    """Processes a trajectory group and returns an AccumulatedData."""
    total_steps = 0
    for trajectory in trajectory_group.trajectories:
        n_steps = _process_trajectory(
            trajectory,
            task_id,
            accumulated,
            chat_parser=chat_parser,
            max_prompt_length=max_prompt_length,
            max_total_length=max_total_length,
            expected_behavior_sampling_params=expected_behavior_sampling_params,
            exact_processed_behavior_rows=exact_processed_behavior_rows,
        )
        total_steps += n_steps

    # Extend episode-level data for all steps in this trajectory group
    # TrajectoryGroup doesn't have episode-level metadata, so we use reasonable defaults
    # TODO(listar2000): check whether and how we should supplement these info from trajectory groups.
    group_id = trajectory_group.group_id if trajectory_group.group_id else task_id
    accumulated.episode_ids.extend([group_id] * total_steps)
    accumulated.is_correct.extend([False] * total_steps)  # default to False for trajectory groups
    accumulated.termination_reasons.extend([TerminationReason.UNKNOWN] * total_steps)
    accumulated.metrics.extend([{}] * total_steps)  # empty metrics for trajectory groups

    return total_steps


def _compute_merge_metrics(accumulated: AccumulatedData, total_agent_steps: int) -> dict[str, float]:
    """Per-batch metrics characterising the merge step.

    Naming matches Tinker's transform_trajectory_groups_to_datums so the
    same metric paths show up regardless of backend:

    - batch/steps_per_traj/{mean,min,max}: number of rows emitted per
      trajectory after prefix-merging. =1 for cumulative trajectories,
      >1 if a prefix break forced a split mid-trajectory.

    - batch/step_response_length/{mean,min,max}: length of the response
      region per row (action tokens + any interleaved observation tokens).
      For unmerged single-step trajectories this is just the action token
      count; for merged multi-turn it's actions + tool/observation tokens.

    - batch/action_token_ratio/{mean,min,max}: fraction of response
      tokens per row that are trainable (mask=1). =1.0 for single-step
      rows (no observations); <1.0 for merged multi-turn (the lower it
      is, the more tool/observation overhead is in the row).

    - batch/merge_compression_ratio: total agent steps ÷ total emitted
      rows. =N for a fully cumulative N-turn batch; =1 means no merging
      occurred (per-step rows, or all single-step trajectories).
    """
    if not accumulated.responses:
        return {}

    # Each row's step_id is trajectory.uid (set by _process_trajectory),
    # so counting occurrences gives rows-per-trajectory.
    from collections import Counter

    import numpy as _np

    rows_per_traj = list(Counter(accumulated.step_ids).values())
    response_lens = [int(r.numel()) for r in accumulated.responses]
    action_token_ratios = []
    for mask in accumulated.traj_mask:
        n = int(mask.numel())
        if n > 0:
            action_token_ratios.append(float(mask.sum().item()) / n)
    total_emitted_rows = len(accumulated.responses)

    metrics = {
        "batch/steps_per_traj/mean": float(_np.mean(rows_per_traj)),
        "batch/steps_per_traj/min": int(_np.min(rows_per_traj)),
        "batch/steps_per_traj/max": int(_np.max(rows_per_traj)),
        "batch/step_response_length/mean": float(_np.mean(response_lens)),
        "batch/step_response_length/min": int(_np.min(response_lens)),
        "batch/step_response_length/max": int(_np.max(response_lens)),
        "batch/action_token_ratio/mean": float(_np.mean(action_token_ratios)) if action_token_ratios else 0.0,
        "batch/action_token_ratio/min": float(_np.min(action_token_ratios)) if action_token_ratios else 0.0,
        "batch/action_token_ratio/max": float(_np.max(action_token_ratios)) if action_token_ratios else 0.0,
        "batch/merge_compression_ratio": (total_agent_steps / total_emitted_rows if total_emitted_rows > 0 else 0.0),
        "batch/context_clipped_rows": int(accumulated.context_clipped_rows),
        "batch/context_clipped_tokens": int(accumulated.context_clipped_tokens),
        "batch/behavior_logprobs_aligned_rows": int(accumulated.behavior_logprobs_aligned_rows),
        "batch/behavior_logprobs_rejected_rows": int(accumulated.behavior_logprobs_rejected_rows),
    }
    metrics.update(
        {
            f"batch/behavior_logprobs_rejected/{reason}": int(count)
            for reason, count in sorted(accumulated.behavior_logprobs_rejection_reasons.items())
        }
    )
    return metrics


def transform_episodes_to_dataproto(
    episodes: list[Episode],
    rollout_engine: VerlEngine,
    max_prompt_length: int,
    max_response_length: int,
    max_total_length: int | None = None,
) -> DataProto:
    """
    Transforms a list of episodes (from running a rLLM workflow) into a verl-compatible DataProto.

    Args:
        episodes: List of episodes to transform.
        rollout_engine: Rollout engine that contains the tokenizer and (optional) multimodal processor.
        max_prompt_length: The maximum length of the prompts.
        max_response_length: The maximum length of the responses.
        max_total_length: Optional cap on prompt plus response non-padding
            tokens. Qwen cumulative-chat rows are clipped to this cap before
            batching; other overlong rows are rejected.
    Returns:
        DataProto: The DataProto built from the episodes. Per-batch merge
        metrics (batch/steps_per_traj, batch/step_response_length) are
        stashed on ``meta_info["merge_metrics"]`` so the caller can lift
        them into trainer_state.metrics without a signature change.
    """
    tokenizer = rollout_engine.tokenizer
    processor = getattr(rollout_engine, "processor", None)
    trajectories = [
        trajectory for episode in episodes for trajectory in episode.trajectories
    ]
    expected_behavior_sampling_params = (
        _resolve_expected_behavior_sampling_params(rollout_engine)
        if _uses_qwen_cumulative_chat(trajectories, rollout_engine.chat_parser)
        else None
    )
    exact_processed_behavior_rows = _requires_exact_processed_behavior_rows(
        rollout_engine
    )

    accumulated = AccumulatedData()
    total_agent_steps = 0
    for episode in episodes:
        task_id = episode.task_id
        total_agent_steps += sum(len(traj.steps) for traj in episode.trajectories)
        total_steps = _process_episode(
            episode,
            task_id,
            accumulated,
            chat_parser=rollout_engine.chat_parser,
            max_prompt_length=max_prompt_length,
            max_total_length=max_total_length,
            expected_behavior_sampling_params=expected_behavior_sampling_params,
            exact_processed_behavior_rows=exact_processed_behavior_rows,
        )
        accumulated.repeat_counts.append(total_steps)

    assert hasattr(tokenizer, "pad_token_id"), "Tokenizer must have a pad token ID"
    pad_token_id = tokenizer.pad_token_id
    batch = _batch_tensors_and_build_data_proto(
        accumulated,
        pad_token_id,
        max_prompt_length,
        max_response_length,
        processor,
        max_total_length=max_total_length,
    )
    batch.meta_info["merge_metrics"] = _compute_merge_metrics(accumulated, total_agent_steps)
    return batch


# TODO: extract common logic from transform_episodes_to_dataproto and transform_trajectory_groups_to_dataproto
def transform_trajectory_groups_to_dataproto(
    trajectory_groups: list[TrajectoryGroup],
    rollout_engine: VerlEngine,
    max_prompt_length: int,
    max_response_length: int,
    max_total_length: int | None = None,
) -> DataProto:
    """
    Transforms a list of trajectory groups (from running a rLLM workflow) into a verl-compatible DataProto.
    """
    tokenizer = rollout_engine.tokenizer
    processor = getattr(rollout_engine, "processor", None)
    trajectories = [
        trajectory
        for trajectory_group in trajectory_groups
        for trajectory in trajectory_group.trajectories
    ]
    expected_behavior_sampling_params = (
        _resolve_expected_behavior_sampling_params(rollout_engine)
        if _uses_qwen_cumulative_chat(trajectories, rollout_engine.chat_parser)
        else None
    )
    exact_processed_behavior_rows = _requires_exact_processed_behavior_rows(
        rollout_engine
    )

    accumulated = AccumulatedData()
    for trajectory_group in trajectory_groups:
        task_id = trajectory_group.task_id
        total_steps = _process_trajectory_group(
            trajectory_group,
            task_id,
            accumulated,
            chat_parser=rollout_engine.chat_parser,
            max_prompt_length=max_prompt_length,
            max_total_length=max_total_length,
            expected_behavior_sampling_params=expected_behavior_sampling_params,
            exact_processed_behavior_rows=exact_processed_behavior_rows,
        )
        accumulated.repeat_counts.append(total_steps)

    assert tokenizer is not None and hasattr(tokenizer, "pad_token_id"), "Tokenizer must have a pad token ID"
    pad_token_id = tokenizer.pad_token_id
    batch = _batch_tensors_and_build_data_proto(
        accumulated,
        pad_token_id,
        max_prompt_length,
        max_response_length,
        processor,
        max_total_length=max_total_length,
    )
    total_agent_steps = sum(len(trajectory.steps) for group in trajectory_groups for trajectory in group.trajectories)
    batch.meta_info["merge_metrics"] = _compute_merge_metrics(
        accumulated,
        total_agent_steps,
    )
    return batch


def update_dataproto_with_advantages(batch: DataProto, container: list[Episode] | list[TrajectoryGroup], mode: str = "broadcast") -> DataProto:
    """
    Updates a DataProto with advantages. Useful when we use rLLM-native advantage computation,
    after which we need to update the DataProto with the advantages.
    """
    # Build a step_id → advantage mapping from episodes/trajectory groups.
    # step_id format must match _process_trajectory's emit: just trajectory.uid.
    # _process_trajectory now emits one row per *trajectory* (prefix-merged
    # multi-step) rather than one row per step, so a single advantage per
    # trajectory is sufficient. In broadcast mode all steps in a trajectory
    # share the same scalar advantage from
    # collect_reward_and_advantage_from_trajectory_groups, so reading from
    # the first valid step is safe.
    adv_by_traj_uid: dict[str, float] = {}
    for item in container:
        for trajectory in item.trajectories:
            if not trajectory.steps:
                continue
            adv = next(
                (s.advantage for s in trajectory.steps if s.advantage is not None),
                0.0,
            )
            adv_by_traj_uid[trajectory.uid] = adv if isinstance(adv, float) else float(adv)

    # Match advantages to batch entries by step_id (robust to batch reordering and padding)
    n_total = len(batch.non_tensor_batch["trajectory_ids"])
    step_ids = batch.non_tensor_batch["step_ids"]
    is_pad = batch.non_tensor_batch.get("is_pad_step", np.zeros(n_total, dtype=bool))

    # step_ids in the batch are trajectory.uid values (set by _process_trajectory).
    # The scalar advantage is broadcast across response tokens by
    # _build_per_step_advantages, multiplied by response_mask which is 0
    # on observation tokens between actions — so observation tokens
    # automatically receive zero advantage in the loss.
    advantage_weights = batch.non_tensor_batch.get(
        "advantage_weights", np.ones(n_total, dtype=np.float32)
    )
    advantages = [
        0.0
        if is_pad[i]
        else adv_by_traj_uid.get(str(step_ids[i]), 0.0)
        * float(advantage_weights[i])
        for i in range(n_total)
    ]

    advantage_tensor = _build_per_step_advantages(batch.batch["response_mask"], advantages)
    batch.batch["advantages"] = advantage_tensor
    batch.batch["returns"] = advantage_tensor

    # TODO(listar2000): we should support `token_level_scores` from the `Step` attribute level.
    # we also need to implement the `kl_penalty` logic used in `verl`.
    if mode == "broadcast":
        batch.batch["token_level_scores"] = batch.batch["traj_rewards"]
        batch.batch["token_level_rewards"] = batch.batch["traj_rewards"]
    else:
        raise ValueError(f"Stepwise advantage mode {mode} not supported in experimental unified trainer.")

    return batch
