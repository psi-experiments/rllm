"""Unit tests for GatewayManager store-backend selection and validation."""

import pytest
from omegaconf import OmegaConf

from rllm.gateway.manager import GatewayManager, container_reachable_url


def _make_config(**gateway_overrides):
    """Build a minimal OmegaConf DictConfig with gateway overrides."""
    return OmegaConf.create({"rllm": {"gateway": gateway_overrides}})


class TestGatewayStoreSelection:
    def test_default_store_is_memory(self):
        gw = GatewayManager(_make_config(), mode="thread")
        assert gw.store == "memory"
        assert gw.db_path is None

    def test_explicit_memory_store(self):
        gw = GatewayManager(_make_config(store="memory"), mode="thread")
        assert gw.store == "memory"
        assert gw.db_path is None

    def test_sqlite_with_explicit_db_path(self):
        gw = GatewayManager(_make_config(store="sqlite", db_path="/tmp/x.db"), mode="thread")
        assert gw.store == "sqlite"
        assert gw.db_path == "/tmp/x.db"

    def test_sqlite_without_db_path_is_allowed(self):
        gw = GatewayManager(_make_config(store="sqlite"), mode="thread")
        assert gw.store == "sqlite"
        assert gw.db_path is None


class TestGatewayStoreValidation:
    def test_unknown_store_raises(self):
        with pytest.raises(ValueError, match="must be 'memory' or 'sqlite'"):
            GatewayManager(_make_config(store="postgres"), mode="thread")

    def test_memory_with_db_path_raises(self):
        with pytest.raises(ValueError, match="db_path is set but store='memory'"):
            GatewayManager(_make_config(store="memory", db_path="/tmp/x.db"), mode="thread")


class TestGatewayAbortResume:
    @staticmethod
    def _config(
        *,
        enabled: bool,
        partial_rollout: bool,
        gateway_override=None,
    ):
        gateway = {}
        if gateway_override is not None:
            gateway["resume_aborted_requests"] = gateway_override
        return OmegaConf.create(
            {
                "model": {"name": "Qwen/Qwen3-8B"},
                "rllm": {
                    "gateway": gateway,
                    "async_training": {
                        "enable": enabled,
                        "partial_rollout": partial_rollout,
                    },
                },
                "actor_rollout_ref": {
                    "rollout": {
                        "engine_kwargs": {
                            "vllm": {
                                "tool_call_parser": "psi_hermes_concurrent",
                            }
                        }
                    }
                },
            }
        )

    def test_enabled_automatically_for_fully_async_partial_rollouts(self):
        gateway = GatewayManager(
            self._config(enabled=True, partial_rollout=True),
            mode="process",
        )
        assert gateway.resume_aborted_requests is True
        assert gateway.abort_resume_tool_parser == "psi_hermes_concurrent"

    @pytest.mark.parametrize(
        ("enabled", "partial_rollout"),
        [(False, False), (True, False), (False, True)],
    )
    def test_disabled_unless_both_async_switches_are_on(
        self,
        enabled,
        partial_rollout,
    ):
        gateway = GatewayManager(
            self._config(enabled=enabled, partial_rollout=partial_rollout),
            mode="process",
        )
        assert gateway.resume_aborted_requests is False

    def test_explicit_override_can_disable_automatic_resume(self):
        gateway = GatewayManager(
            self._config(
                enabled=True,
                partial_rollout=True,
                gateway_override=False,
            ),
            mode="process",
        )
        assert gateway.resume_aborted_requests is False


class TestContainerReachableUrl:
    @pytest.mark.parametrize(
        ("url", "backend", "expected"),
        [
            # Docker containers can't reach the host's loopback — rewrite
            # to host.docker.internal, preserving port and path.
            ("http://127.0.0.1:8000/v1", "docker", "http://host.docker.internal:8000/v1"),
            ("http://localhost:9001/sessions/x/v1", "docker", "http://host.docker.internal:9001/sessions/x/v1"),
            # Non-docker backends (and unset) pass the URL through untouched.
            ("http://127.0.0.1:8000/v1", "modal", "http://127.0.0.1:8000/v1"),
            ("http://localhost:9000/v1", "local", "http://localhost:9000/v1"),
            ("http://127.0.0.1:8000/v1", None, "http://127.0.0.1:8000/v1"),
        ],
    )
    def test_loopback_rewrite_only_for_docker_backend(self, url, backend, expected):
        assert container_reachable_url(url, backend) == expected
