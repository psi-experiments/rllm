"""Token-exact continuation for vLLM requests aborted during weight sync."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from pydantic import ValidationError
from rllm_model_gateway import GatewayConfig, create_app

from tests.helpers.mock_vllm import MockVLLMServer


class _RenderedTokens:
    def __init__(self, token_ids: list[int]) -> None:
        self.token_ids = token_ids


class _CumulativeRenderer:
    def bridge_to_next_turn(
        self,
        prev_prompt_ids,
        prev_completion_ids,
        new_messages,
        *,
        tools=None,
    ):
        del new_messages, tools
        return _RenderedTokens(list(prev_prompt_ids) + list(prev_completion_ids) + [99])


def test_resume_loads_tokenizer_from_separate_local_path(monkeypatch):
    loaded_paths: list[str] = []

    class FakeTokenizer:
        def decode(self, token_ids, skip_special_tokens=False):
            del skip_special_tokens
            return " ".join(map(str, token_ids))

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, path):
            loaded_paths.append(path)
            return FakeTokenizer()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
    )
    config = GatewayConfig(
        model="Qwen/Qwen3-8B",
        tokenizer_path="/models/qwen3-8b/snapshots/exact",
        resume_aborted_requests=True,
    )

    app = create_app(config)

    assert loaded_paths == ["/models/qwen3-8b/snapshots/exact"]
    assert app.state.config.model == "Qwen/Qwen3-8B"


def test_no_progress_resume_limit_must_be_positive():
    with pytest.raises(ValidationError):
        GatewayConfig(max_consecutive_no_progress_resumes=0)


def _response(
    *,
    prompt_ids: list[int],
    token_ids: list[int],
    logprobs: list[float],
    finish_reason: str,
    text: str,
    chat: bool,
) -> dict[str, Any]:
    if chat:
        choice = {
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": finish_reason,
            "stop_reason": finish_reason,
            "token_ids": token_ids,
            "logprobs": {
                "content": [
                    {
                        "token": f"token-{token_id}",
                        "logprob": logprob,
                        "bytes": None,
                        "top_logprobs": [],
                    }
                    for token_id, logprob in zip(token_ids, logprobs, strict=True)
                ]
            },
        }
        object_type = "chat.completion"
    else:
        choice = {
            "index": 0,
            "text": text,
            "finish_reason": finish_reason,
            "stop_reason": finish_reason,
            "token_ids": token_ids,
            "logprobs": {
                "tokens": [f"token-{token_id}" for token_id in token_ids],
                "token_logprobs": logprobs,
                "token_ids": token_ids,
            },
        }
        object_type = "text_completion"
    return {
        "id": "completion-test",
        "object": object_type,
        "model": "mock-model",
        "prompt_token_ids": prompt_ids,
        "choices": [choice],
        "usage": {
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(token_ids),
            "total_tokens": len(prompt_ids) + len(token_ids),
        },
    }


def _app(
    mock_vllm: MockVLLMServer,
    decoder,
    *,
    tool_parser: str | None = None,
    max_consecutive_no_progress_resumes: int = 120,
):
    config = GatewayConfig(
        store_worker="memory",
        workers=[{"url": f"{mock_vllm.url}/v1", "worker_id": "w0"}],
        health_check_interval=999,
        sync_traces=True,
        resume_aborted_requests=True,
        abort_resume_tool_parser=tool_parser,
        max_consecutive_no_progress_resumes=max_consecutive_no_progress_resumes,
    )
    app = create_app(config, token_decoder=decoder)
    app.state.proxy.abort_resume_delay_s = 0
    return app


@pytest.mark.asyncio
async def test_normal_response_is_unchanged(mock_vllm: MockVLLMServer):
    """Enabling continuation does not re-decode an uninterrupted response."""
    decoder_called = False

    def decoder(_: list[int]) -> str:
        nonlocal decoder_called
        decoder_called = True
        return "should not be used"

    app = _app(mock_vllm, decoder)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/sessions/normal/v1/chat/completions",
            json={
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "Hello from mock!"
    assert decoder_called is False


@pytest.mark.asyncio
async def test_aborted_response_resumes_with_saved_tokens(
    mock_vllm: MockVLLMServer,
):
    first = _response(
        prompt_ids=[1, 2],
        token_ids=[10, 11],
        logprobs=[-0.1, -0.2],
        finish_reason="abort",
        text="partial",
        chat=True,
    )
    second = _response(
        prompt_ids=[1, 2, 10, 11],
        token_ids=[12, 13],
        logprobs=[-0.3, -0.4],
        finish_reason="stop",
        text=" suffix",
        chat=False,
    )
    mock_vllm.queue_responses(first, second)
    app = _app(
        mock_vllm,
        lambda ids: "complete answer" if ids == [10, 11, 12, 13] else "wrong",
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/sessions/resume/v1/chat/completions",
            json={
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 6,
            },
        )
        traces = (await client.get("/sessions/resume/traces")).json()

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"] == {
        "role": "assistant",
        "content": "complete answer",
    }
    assert len(mock_vllm.request_log) == 2
    continuation = mock_vllm.request_log[1]
    assert continuation["prompt"] == [1, 2, 10, 11]
    assert continuation["max_tokens"] == 4
    assert continuation["add_special_tokens"] is False
    assert "messages" not in continuation

    assert len(traces) == 1
    assert traces[0]["completion_token_ids"] == [10, 11, 12, 13]
    assert traces[0]["logprobs"] == [-0.1, -0.2, -0.3, -0.4]
    assert traces[0]["metadata"]["interruption_count"] == 1


@pytest.mark.asyncio
async def test_repeated_interruptions_keep_extending_the_same_turn(
    mock_vllm: MockVLLMServer,
):
    mock_vllm.queue_responses(
        _response(
            prompt_ids=[1, 2],
            token_ids=[10],
            logprobs=[-0.1],
            finish_reason="abort",
            text="a",
            chat=True,
        ),
        _response(
            prompt_ids=[1, 2, 10],
            token_ids=[11],
            logprobs=[-0.2],
            finish_reason="aborted",
            text="b",
            chat=False,
        ),
        _response(
            prompt_ids=[1, 2, 10, 11],
            token_ids=[12],
            logprobs=[-0.3],
            finish_reason="stop",
            text="c",
            chat=False,
        ),
    )
    app = _app(mock_vllm, lambda ids: "abc")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/sessions/repeated/v1/chat/completions",
            json={
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 5,
            },
        )
        traces = (await client.get("/sessions/repeated/traces")).json()

    assert response.status_code == 200
    assert [request["prompt"] for request in mock_vllm.request_log[1:]] == [
        [1, 2, 10],
        [1, 2, 10, 11],
    ]
    assert [request["max_tokens"] for request in mock_vllm.request_log[1:]] == [
        4,
        3,
    ]
    assert traces[0]["completion_token_ids"] == [10, 11, 12]
    assert traces[0]["metadata"]["interruption_count"] == 2


@pytest.mark.asyncio
async def test_cumulative_second_turn_resumes_before_returning_to_agent(
    mock_vllm: MockVLLMServer,
):
    first_turn = _response(
        prompt_ids=[1, 2],
        token_ids=[10],
        logprobs=[-0.1],
        finish_reason="stop",
        text="first answer",
        chat=True,
    )
    cumulative_prompt = [1, 2, 10, 99]
    interrupted = _response(
        prompt_ids=cumulative_prompt,
        token_ids=[20],
        logprobs=[-0.2],
        finish_reason="abort",
        text="partial",
        chat=False,
    )
    resumed = _response(
        prompt_ids=cumulative_prompt + [20],
        token_ids=[21],
        logprobs=[-0.3],
        finish_reason="stop",
        text=" suffix",
        chat=False,
    )
    mock_vllm.queue_responses(first_turn, interrupted, resumed)
    app = _app(
        mock_vllm,
        lambda ids: "complete second answer" if ids == [20, 21] else "wrong",
    )
    app.state.proxy.cumulative_token_mode = True
    app.state.proxy.renderer = _CumulativeRenderer()
    release_request_counts: list[int] = []
    original_release = app.state.proxy.router.release

    def record_release(worker_url: str) -> None:
        release_request_counts.append(len(mock_vllm.request_log))
        original_release(worker_url)

    app.state.proxy.router.release = record_release

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        first_response = await client.post(
            "/sessions/cumulative/v1/chat/completions",
            json={
                "model": "mock-model",
                "messages": [{"role": "user", "content": "first"}],
                "max_tokens": 4,
            },
        )
        second_response = await client.post(
            "/sessions/cumulative/v1/chat/completions",
            json={
                "model": "mock-model",
                "messages": [
                    {"role": "user", "content": "first"},
                    {
                        "role": "assistant",
                        "content": first_response.json()["choices"][0]["message"]["content"],
                    },
                    {"role": "user", "content": "second"},
                ],
                "max_tokens": 4,
            },
        )
        traces = (await client.get("/sessions/cumulative/traces")).json()

    assert second_response.status_code == 200
    assert second_response.json()["choices"][0]["finish_reason"] == "stop"
    assert second_response.json()["choices"][0]["message"] == {
        "role": "assistant",
        "content": "complete second answer",
    }
    assert len(mock_vllm.request_log) == 3
    assert mock_vllm.request_log[1]["prompt"] == cumulative_prompt
    assert mock_vllm.request_log[2]["prompt"] == cumulative_prompt + [20]
    assert release_request_counts == [1, 3]
    assert traces[1]["completion_token_ids"] == [20, 21]
    assert traces[1]["metadata"]["interruption_count"] == 1


@pytest.mark.asyncio
async def test_zero_progress_interruptions_return_bounded_gateway_error(
    mock_vllm: MockVLLMServer,
):
    empty_abort = _response(
        prompt_ids=[1, 2],
        token_ids=[],
        logprobs=[],
        finish_reason="abort",
        text="",
        chat=True,
    )
    mock_vllm.queue_responses(*[empty_abort for _ in range(5)])
    app = _app(
        mock_vllm,
        lambda ids: "",
        max_consecutive_no_progress_resumes=2,
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/sessions/stalled/v1/chat/completions",
            json={
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 4,
            },
        )

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "rllm_abort_resume_error"
    assert "no token progress" in response.json()["error"]["message"]
    assert len(mock_vllm.request_log) == 2


@pytest.mark.asyncio
async def test_tool_json_split_by_interruption_is_returned_once(
    mock_vllm: MockVLLMServer,
):
    first = _response(
        prompt_ids=[1, 2],
        token_ids=[20, 21],
        logprobs=[-0.1, -0.2],
        finish_reason="abort",
        text='<tool_call>\n{"name":"search",',
        chat=True,
    )
    second = _response(
        prompt_ids=[1, 2, 20, 21],
        token_ids=[22, 23],
        logprobs=[-0.3, -0.4],
        finish_reason="stop",
        text='"arguments":{"query":"qutip"}}\n</tool_call>',
        chat=False,
    )
    mock_vllm.queue_responses(first, second)
    decoded = '<tool_call>\n{"name":"search","arguments":{"query":"qutip"}}\n</tool_call>'
    app = _app(
        mock_vllm,
        lambda ids: decoded,
        tool_parser="psi_hermes_concurrent",
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/sessions/tool/v1/chat/completions",
            json={
                "model": "mock-model",
                "messages": [{"role": "user", "content": "use a tool"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "search", "parameters": {}},
                    }
                ],
                "tool_choice": "auto",
                "max_tokens": 8,
            },
        )
        traces = (await client.get("/sessions/tool/traces")).json()

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    assert len(choice["message"]["tool_calls"]) == 1
    function = choice["message"]["tool_calls"][0]["function"]
    assert function["name"] == "search"
    assert function["arguments"] == '{"query": "qutip"}'
    assert traces[0]["completion_token_ids"] == [20, 21, 22, 23]
