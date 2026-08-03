"""Token-exact continuation for vLLM requests aborted during weight sync."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from rllm_model_gateway import GatewayConfig, create_app

from tests.helpers.mock_vllm import MockVLLMServer


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
):
    config = GatewayConfig(
        store_worker="memory",
        workers=[{"url": f"{mock_vllm.url}/v1", "worker_id": "w0"}],
        health_check_interval=999,
        sync_traces=True,
        resume_aborted_requests=True,
        abort_resume_tool_parser=tool_parser,
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
