"""Resume non-streaming vLLM generations interrupted by a weight update.

The first request remains a normal OpenAI chat-completions request.  If vLLM
returns ``finish_reason=\"abort\"``, subsequent requests use the raw prompt and
completion token IDs with the completions endpoint.  This avoids re-rendering
the chat template and keeps the resumed tokens on exactly the same prefix.
"""

from __future__ import annotations

import copy
import json
import re
import uuid
from collections.abc import Callable
from typing import Any

from rllm_model_gateway.data_process import (
    extract_completion_token_ids,
    extract_logprobs,
    extract_prompt_token_ids,
)

TokenDecoder = Callable[[list[int]], str]

_ABORT_REASONS = frozenset({"abort", "aborted"})
_SUPPORTED_TOOL_PARSERS = frozenset({"hermes", "psi_hermes_concurrent"})
_TOOL_CALL_RE = re.compile(
    r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)",
    re.DOTALL,
)
_CHAT_ONLY_FIELDS = frozenset(
    {
        "add_generation_prompt",
        "chat_template",
        "chat_template_kwargs",
        "continue_final_message",
        "messages",
        "parallel_tool_calls",
        "response_format",
        "stream_options",
        "top_logprobs",
        "tool_choice",
        "tools",
    }
)


class AbortResumeError(RuntimeError):
    """The gateway could not safely resume an interrupted generation."""


def is_aborted_response(response: dict[str, Any]) -> bool:
    """Return whether the first choice says vLLM aborted this generation."""
    choices = response.get("choices") or []
    if not choices:
        return False
    choice = choices[0]
    return choice.get("finish_reason") in _ABORT_REASONS or choice.get("stop_reason") in _ABORT_REASONS


def _choice(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices") or []
    if len(choices) != 1:
        raise AbortResumeError(f"Interrupted generation can only be resumed when the upstream returns exactly one choice; received {len(choices)}")
    return choices[0]


def _chat_logprob_entries(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize chat and completions logprobs to chat-completions entries."""
    choice = _choice(response)
    logprobs = choice.get("logprobs") or {}
    content = logprobs.get("content")
    if content is not None:
        return [copy.deepcopy(entry) for entry in content if entry is not None]

    values = logprobs.get("token_logprobs")
    if values is None:
        return []
    tokens = logprobs.get("tokens") or []
    entries: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        if value is None:
            continue
        entries.append(
            {
                "token": tokens[index] if index < len(tokens) else "",
                "logprob": float(value),
                "bytes": None,
                "top_logprobs": [],
            }
        )
    return entries


def _parse_hermes_message(text: str) -> tuple[dict[str, Any], bool]:
    """Apply the non-streaming Hermes tool-call contract to decoded text."""
    matches = list(_TOOL_CALL_RE.finditer(text))
    if not matches:
        return {"role": "assistant", "content": text}, False

    tool_calls: list[dict[str, Any]] = []
    try:
        for match in matches:
            raw_call = match.group(1) if match.group(1) is not None else match.group(2)
            call = json.loads(raw_call)
            tool_calls.append(
                {
                    "id": f"chatcmpl-tool-{uuid.uuid4().hex}",
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": json.dumps(
                            call["arguments"],
                            ensure_ascii=False,
                        ),
                    },
                }
            )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        # Match vLLM's safe fallback: malformed tool syntax remains ordinary
        # assistant text so the agent harness can grade or reprompt it.
        return {"role": "assistant", "content": text}, False

    leading_content = text[: matches[0].start()]
    return {
        "role": "assistant",
        "content": leading_content or None,
        "tool_calls": tool_calls,
    }, True


class AbortedGeneration:
    """Accumulate all pieces of one interrupted chat-completions response."""

    def __init__(
        self,
        request_body: dict[str, Any],
        first_response: dict[str, Any],
        *,
        decoder: TokenDecoder,
        tool_parser: str | None,
    ) -> None:
        if request_body.get("tools") and tool_parser not in _SUPPORTED_TOOL_PARSERS:
            raise AbortResumeError(f"Unsupported abort-resume tool parser {tool_parser!r}; supported parsers: {sorted(_SUPPORTED_TOOL_PARSERS)}")
        if request_body.get("stream", False):
            raise AbortResumeError("Interrupted streaming requests cannot be resumed")
        if request_body.get("n", 1) != 1:
            raise AbortResumeError("Interrupted requests with n != 1 cannot be resumed")
        self.request_body = copy.deepcopy(request_body)
        self.first_response = copy.deepcopy(first_response)
        self.decoder = decoder
        self.tool_parser = tool_parser
        self.prompt_token_ids = extract_prompt_token_ids(first_response)
        if not self.prompt_token_ids:
            raise AbortResumeError("Interrupted response did not include prompt_token_ids; token-exact continuation is impossible")

        configured_max = request_body.get("max_tokens")
        self.original_max_tokens = int(configured_max) if configured_max is not None else None
        self.completion_token_ids: list[int] = []
        self.logprob_entries: list[dict[str, Any]] = []
        self.last_response: dict[str, Any] = {}
        self.interruption_count = 0
        self.append(first_response)

    def append(self, response: dict[str, Any]) -> None:
        """Append one upstream segment, validating token/logprob alignment."""
        _choice(response)
        token_ids = extract_completion_token_ids(response)
        logprob_values = extract_logprobs(response)
        entries = _chat_logprob_entries(response)
        if token_ids and len(logprob_values) != len(token_ids):
            raise AbortResumeError(f"Interrupted response did not include one logprob per generated token ({len(logprob_values)} logprobs for {len(token_ids)} tokens)")
        if token_ids and len(entries) != len(token_ids):
            raise AbortResumeError("Interrupted response logprob metadata did not align with its token IDs")
        self.completion_token_ids.extend(token_ids)
        self.logprob_entries.extend(entries)
        self.last_response = copy.deepcopy(response)
        if is_aborted_response(response):
            self.interruption_count += 1

    @property
    def exhausted(self) -> bool:
        return self.original_max_tokens is not None and len(self.completion_token_ids) >= self.original_max_tokens

    @property
    def should_resume(self) -> bool:
        return is_aborted_response(self.last_response) and not self.exhausted

    def continuation_body(self) -> dict[str, Any]:
        """Build a raw-token completions request for the unfinished suffix."""
        body = {key: copy.deepcopy(value) for key, value in self.request_body.items() if key not in _CHAT_ONLY_FIELDS and key != "stream"}
        body.update(
            {
                "prompt": self.prompt_token_ids + self.completion_token_ids,
                "add_special_tokens": False,
                "echo": False,
                "stream": False,
                "return_token_ids": True,
            }
        )
        # The chat endpoint accepts a boolean, while the legacy completions
        # endpoint expects the number of alternatives to return. A value of 1
        # still provides the chosen token's logprob for every generated token.
        body["logprobs"] = 1
        if self.original_max_tokens is not None:
            body["max_tokens"] = max(
                0,
                self.original_max_tokens - len(self.completion_token_ids),
            )
        return body

    def merged_response(self) -> dict[str, Any]:
        """Return one chat response containing all generated token pieces."""
        response = copy.deepcopy(self.first_response)
        choice = _choice(response)
        final_choice = _choice(self.last_response)
        decoded = self.decoder(self.completion_token_ids)

        has_tool_calls = False
        if self.request_body.get("tools"):
            message, has_tool_calls = _parse_hermes_message(decoded)
        else:
            message = {"role": "assistant", "content": decoded}

        finish_reason = final_choice.get("finish_reason")
        stop_reason = final_choice.get("stop_reason")
        if self.exhausted and is_aborted_response(self.last_response):
            finish_reason = "length"
            stop_reason = "length"
        elif has_tool_calls:
            finish_reason = "tool_calls"

        choice.pop("text", None)
        choice["message"] = message
        choice["finish_reason"] = finish_reason
        choice["stop_reason"] = stop_reason
        choice["token_ids"] = list(self.completion_token_ids)
        choice["logprobs"] = {"content": copy.deepcopy(self.logprob_entries)}

        response["object"] = "chat.completion"
        response["prompt_token_ids"] = list(self.prompt_token_ids)
        response["usage"] = {
            "prompt_tokens": len(self.prompt_token_ids),
            "completion_tokens": len(self.completion_token_ids),
            "total_tokens": len(self.prompt_token_ids) + len(self.completion_token_ids),
        }
        return response
