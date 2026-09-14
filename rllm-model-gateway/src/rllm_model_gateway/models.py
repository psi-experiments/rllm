"""Pydantic data models for the rllm-model-gateway."""

import math
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, model_validator

LogprobsMode = Literal[
    "raw_logprobs",
    "processed_logprobs",
    "raw_logits",
    "processed_logits",
]


class GeneratorVersionSpan(BaseModel):
    """Generator version for one half-open completion-token interval."""

    start: int = Field(ge=0)
    end: int = Field(gt=0)
    weight_version: int | None = None

    @model_validator(mode="after")
    def _end_follows_start(self) -> "GeneratorVersionSpan":
        if self.end <= self.start:
            raise ValueError("generator version span end must be greater than start")
        return self


class TraceRecord(BaseModel):
    """A single captured LLM call with full token-level data."""

    trace_id: str
    session_id: str
    model: str = ""
    # Input
    messages: list[dict[str, Any]] = Field(default_factory=list)
    prompt_token_ids: list[int] = Field(default_factory=list)
    # Output
    response_message: dict[str, Any] = Field(default_factory=dict)
    completion_token_ids: list[int] = Field(default_factory=list)
    logprobs: list[float] | None = None
    # Semantics of ``logprobs`` as configured on the upstream vLLM server.
    # ``None`` means unknown and must not be assumed to be behavior-policy
    # probabilities by a training consumer.
    logprobs_mode: LogprobsMode | None = None
    # Forwarded request-side sampling controls used by the behavior policy.
    # Omitted controls still use the inference worker's defaults. These are
    # diagnostic inputs to distribution matching; they do not by themselves
    # guarantee that trainer recomputation applies the same logits processors
    # as vLLM.
    behavior_sampling_params: dict[str, Any] = Field(default_factory=dict)
    # Compact policy lineage for completion tokens. Offsets are half-open and
    # relative to ``completion_token_ids``. A turn interrupted by weight sync
    # may contain more than one span.
    generator_version_spans: list[GeneratorVersionSpan] = Field(default_factory=list)
    routing_matrices: list[str] | None = None
    finish_reason: str | None = None
    weight_version: int | None = None
    # Metadata
    latency_ms: float = 0.0
    token_counts: dict[str, int] = Field(default_factory=dict)
    timestamp: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    raw_request: dict[str, Any] | None = None
    raw_response: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _validate_token_lineage(self) -> "TraceRecord":
        # Old gateway records did not declare logprob semantics and may contain
        # sparse diagnostic values. Keep those records loadable. The strict
        # one-value-per-token contract is required only when a record claims
        # its values are processed behavior-policy logprobs.
        if self.logprobs_mode == "processed_logprobs" and self.logprobs is not None:
            if len(self.logprobs) != len(self.completion_token_ids):
                raise ValueError("processed trace must contain one logprob per completion token")
            if not all(math.isfinite(value) for value in self.logprobs):
                raise ValueError("processed trace generated-token logprobs must be finite")

        if not self.generator_version_spans:
            return self
        if not self.completion_token_ids:
            raise ValueError("generator version spans require completion tokens")

        expected_start = 0
        previous_version: int | None = None
        for index, span in enumerate(self.generator_version_spans):
            if span.start != expected_start:
                raise ValueError("generator version spans must be contiguous and start at token offset 0")
            if index > 0 and span.weight_version == previous_version:
                raise ValueError("adjacent generator version spans with the same version must be compacted")
            expected_start = span.end
            previous_version = span.weight_version
        if expected_start != len(self.completion_token_ids):
            raise ValueError("generator version spans must cover every completion token")
        return self


def _split_worker_url(raw: str) -> dict[str, str]:
    """Split ``http://host:port/v1`` into base URL + api_path.

    If the URL contains a path component (e.g. ``/v1``), it is separated
    out so that health checks can use the bare ``scheme://host:port`` while
    proxying uses ``scheme://host:port + api_path``.
    """
    parsed = urlparse(raw.rstrip("/"))
    if parsed.path and parsed.path != "/":
        base = f"{parsed.scheme}://{parsed.netloc}"
        return {"url": base, "api_path": parsed.path}
    return {"url": raw.rstrip("/"), "api_path": "/v1"}


class WorkerConfig(BaseModel):
    """Configuration for a single inference worker."""

    worker_id: str = ""
    url: str  # base URL, e.g. "http://localhost:4000"
    api_path: str = "/v1"  # API version prefix, appended for proxying
    model_name: str | None = None
    weight: int = 1

    @model_validator(mode="before")
    @classmethod
    def _auto_split_url(cls, values: Any) -> Any:
        """Backward compat: auto-split url with path into url + api_path."""
        if isinstance(values, dict):
            url = values.get("url", "")
            # Only auto-split if api_path was NOT explicitly provided
            if url and "api_path" not in values:
                parts = _split_worker_url(url)
                values["url"] = parts["url"]
                values["api_path"] = parts["api_path"]
        return values


class WorkerInfo(BaseModel):
    """Runtime info for a worker including health state."""

    worker_id: str
    url: str  # base URL
    api_path: str = "/v1"
    model_name: str | None = None
    weight: int = 1
    healthy: bool = True
    active_requests: int = 0

    @model_validator(mode="before")
    @classmethod
    def _auto_split_url(cls, values: Any) -> Any:
        """Auto-split url with path into url + api_path."""
        if isinstance(values, dict):
            url = values.get("url", "")
            if url and "api_path" not in values:
                parts = _split_worker_url(url)
                values["url"] = parts["url"]
                values["api_path"] = parts["api_path"]
        return values

    @property
    def api_url(self) -> str:
        """Full URL for API proxying: base + api_path."""
        return self.url.rstrip("/") + self.api_path


class SessionInfo(BaseModel):
    """Session metadata returned by session management APIs."""

    session_id: str
    trace_count: int = 0
    created_at: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GatewayConfig(BaseModel):
    """Top-level gateway configuration."""

    host: str = "0.0.0.0"
    port: int = 9090
    workers: list[WorkerConfig] = Field(default_factory=list)
    db_path: str | None = None
    store_worker: str = "memory"
    add_logprobs: bool = True
    # vLLM configures this at server startup, not per request. The gateway
    # records the declared mode so training can fail closed when the semantics
    # of captured values are unknown.
    logprobs_mode: LogprobsMode | None = None
    add_return_token_ids: bool = True
    strip_vllm_fields: bool = True
    routing_policy: str | None = None
    health_check_interval: float = 10.0
    log_level: str = "INFO"
    sync_traces: bool = False
    model: str | None = None  # When set, overrides ``body.model``
    # Optional local checkpoint used only for tokenizer loading. Keep this
    # separate from ``model`` so requests can retain the public served-model
    # name while offline workers read tokenizer files from a mounted volume.
    tokenizer_path: str | None = None
    cumulative_token_mode: bool = False
    # renderers family for the cumulative-mode bridge. Check supported model families
    # in MODEL_RENDERER_MAP of https://github.com/PrimeIntellect-ai/renderers/blob/main/renderers/base.py
    renderer_family: str = "auto"
    # Fully async VERL can abort a live vLLM request during weight sync. Resume
    # that same model turn from its raw token IDs before returning to the agent.
    resume_aborted_requests: bool = False
    abort_resume_tool_parser: str | None = None
    max_consecutive_no_progress_resumes: int = Field(default=120, gt=0)
