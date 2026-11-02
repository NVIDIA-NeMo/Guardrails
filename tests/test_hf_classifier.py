# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the hf_classifier rail."""

from __future__ import annotations

import asyncio
import json
import logging
import pickle
import threading
import time
from types import SimpleNamespace
from unittest import mock

import httpx
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import TypeAdapter, ValidationError
from pytest_httpx import HTTPXMock

from nemoguardrails.actions.rail_outcome import RailOutcome, TransformTarget
from nemoguardrails.http import (
    HTTPConnectionError,
    HTTPResponse,
    HTTPTimeoutError,
    InstrumentedHTTPClient,
    RetryingHTTPClient,
    RetryPolicy,
)
from nemoguardrails.library.hf_classifier import backends as backends_mod
from nemoguardrails.library.hf_classifier.actions import (
    _classify_and_check,
    _hf_classifier_retrieval_outcome,
    hf_classifier_check_input,
    hf_classifier_check_output,
    hf_classifier_check_retrieval,
)
from nemoguardrails.library.hf_classifier.backends import (
    ClassificationResult,
    FMSBackend,
    KServeBackend,
    LocalBackend,
    VLLMBackend,
    _build_headers,
    _build_ssl_config,
    _get_or_create_pipeline,
    _get_timeout,
    _pipeline_cache_key,
    _PipelineLoadError,
    _PipelineLoading,
    get_backend,
)
from nemoguardrails.rails.llm.config import (
    HFClassifierConfig,
    LocalHFClassifierConfig,
    RailsConfig,
    RemoteHFClassifierConfig,
)
from nemoguardrails.testing import RecordingHTTPClient

_REMOTE_DEFAULTS = dict(
    engine="vllm",
    model="test-model",
    base_url="http://localhost:8000",
    threshold=0.5,
    blocked_labels=["toxic"],
)


def _remote(**overrides) -> RemoteHFClassifierConfig:
    return RemoteHFClassifierConfig(**{**_REMOTE_DEFAULTS, **overrides})


def _local(**overrides) -> LocalHFClassifierConfig:
    defaults = dict(engine="local", model="test-model", blocked_labels=["toxic"])
    return LocalHFClassifierConfig(**{**defaults, **overrides})


def _rails_cfg(name, hf_config):
    return SimpleNamespace(rails=SimpleNamespace(config=SimpleNamespace(hf_classifier={name: hf_config})))


def _patch_backend(results):
    backend = mock.AsyncMock()
    backend.classify.return_value = results
    return mock.patch(
        "nemoguardrails.library.hf_classifier.actions.get_backend",
        return_value=backend,
    )


@pytest.fixture(autouse=True)
def _clear_caches():
    backends_mod._pipelines.clear()
    backends_mod._ssl_cache.clear()
    backends_mod._warned_env_vars.clear()
    backends_mod._backend_instances.clear()
    yield
    backends_mod._pipelines.clear()
    backends_mod._ssl_cache.clear()
    backends_mod._warned_env_vars.clear()
    backends_mod._backend_instances.clear()


@pytest.fixture
def http_otel():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test"), exporter


class TestConfig:
    def test_local_has_no_base_url(self):
        assert "base_url" not in LocalHFClassifierConfig.model_fields

    @pytest.mark.parametrize("engine", ["vllm", "kserve", "fms"])
    def test_remote_requires_base_url(self, engine):
        with pytest.raises(ValidationError, match="base_url"):
            RemoteHFClassifierConfig(engine=engine, model="m", blocked_labels=["x"])  # type: ignore[reportCallIssue]

    def test_invalid_base_url_scheme(self):
        with pytest.raises(ValidationError, match="http://"):
            _remote(base_url="ftp://host:8000")

    def test_api_key_requires_https_base_url(self):
        with pytest.raises(ValidationError, match="must use HTTPS"):
            _remote(base_url="http://host:8000", api_key_env_var="HF_TOKEN")

    def test_api_key_accepts_https_base_url(self):
        config = _remote(base_url="https://host:8000", api_key_env_var="HF_TOKEN")
        assert config.base_url == "https://host:8000"

    def test_http_base_url_remains_supported_without_api_key(self):
        assert _remote(base_url="http://host:8000").base_url == "http://host:8000"

    def test_aggregation_rejects_text_classification(self):
        with pytest.raises(ValidationError, match="aggregation_strategy"):
            _local(parameters={"aggregation_strategy": "simple"}, task="text-classification")

    def test_aggregation_accepts_token_classification(self):
        c = _local(task="token-classification", parameters={"aggregation_strategy": "simple"})
        assert c.parameters["aggregation_strategy"] == "simple"

    def test_remote_has_no_aggregation(self):
        assert "aggregation_strategy" not in RemoteHFClassifierConfig.model_fields

    @pytest.mark.parametrize("val", [-0.1, 1.1])
    def test_threshold_out_of_range(self, val):
        with pytest.raises(ValidationError):
            _remote(threshold=val)

    def test_empty_blocked_labels_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            _remote(blocked_labels=[])
        assert "blocked_labels is empty" in caplog.text

    def test_verify_ssl_false_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            _remote(parameters={"verify_ssl": False})
        assert "TLS verification is disabled" in caplog.text

    def test_base_url_trailing_slash_stripped(self):
        c = _remote(base_url="http://host:8000/")
        assert c.base_url == "http://host:8000"

    def test_unknown_parameters_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            _remote(parameters={"typo_param": 42})
        assert "unknown parameters ignored" in caplog.text
        assert "typo_param" in caplog.text


class TestHeaders:
    def test_defaults(self):
        assert _build_headers(_remote()) == {"Content-Type": "application/json"}

    def test_api_key(self, monkeypatch):
        monkeypatch.setenv("K", "secret")
        h = _build_headers(_remote(base_url="https://host:8000", api_key_env_var="K"))
        assert h["Authorization"] == "Bearer secret"

    def test_missing_key_warns_once(self, caplog):
        c = _remote(base_url="https://host:8000", api_key_env_var="MISSING")
        with caplog.at_level(logging.WARNING):
            _build_headers(c)
        assert "MISSING" in caplog.text
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            _build_headers(c)
        assert "MISSING" not in caplog.text

    def test_no_default_headers_support(self):
        h = _build_headers(_remote(parameters={"default_headers": {"X-Custom": "val"}}))
        assert "X-Custom" not in h


class TestSSLTimeout:
    def test_timeout_default(self):
        assert _get_timeout(_remote()) == 30.0

    def test_timeout_custom(self):
        assert _get_timeout(_remote(parameters={"timeout": 10.0})) == 10.0

    def test_ssl_default(self):
        cfg = _build_ssl_config(_remote())
        assert cfg.verify is True
        assert cfg.ca_bundle is None
        assert cfg.client_certificate is None

    def test_ssl_disabled(self):
        cfg = _build_ssl_config(_remote(parameters={"verify_ssl": False}))
        assert cfg.verify is False

    def test_ssl_cached(self):
        c = _remote(parameters={"ca_cert": "/ca.pem"})
        first = _build_ssl_config(c)
        second = _build_ssl_config(c)
        assert first is second


class TestLocalBackend:
    @pytest.mark.asyncio
    async def test_text_classification(self):
        c = _local()
        key = backends_mod._pipeline_cache_key(c.model, c.task, c.parameters)
        backends_mod._pipelines[key] = mock.MagicMock(return_value=[{"label": "toxic", "score": 0.9}])
        r = await LocalBackend(c).classify("text")
        assert r == [ClassificationResult(label="toxic", score=0.9)]

    @pytest.mark.asyncio
    async def test_token_entity_group(self):
        c = _local(
            task="token-classification",
            blocked_labels=["PER"],
            parameters={"aggregation_strategy": "simple"},
        )
        key = backends_mod._pipeline_cache_key(c.model, c.task, c.parameters)
        backends_mod._pipelines[key] = mock.MagicMock(return_value=[{"entity_group": "PER", "score": 0.85}])
        r = await LocalBackend(c).classify("John")
        assert r[0]["label"] == "PER"

    @pytest.mark.asyncio
    async def test_token_entity_fallback(self):
        c = _local(
            task="token-classification",
            blocked_labels=["LOC"],
            parameters={"aggregation_strategy": "simple"},
        )
        key = backends_mod._pipeline_cache_key(c.model, c.task, c.parameters)
        backends_mod._pipelines[key] = mock.MagicMock(return_value=[{"entity": "LOC", "score": 0.7}])
        r = await LocalBackend(c).classify("Paris")
        assert r[0]["label"] == "LOC"

    @pytest.mark.asyncio
    async def test_nested_results_flattened(self):
        c = _local()
        key = backends_mod._pipeline_cache_key(c.model, c.task, c.parameters)
        backends_mod._pipelines[key] = mock.MagicMock(
            return_value=[[{"label": "toxic", "score": 0.9}, {"label": "safe", "score": 0.1}]]
        )
        r = await LocalBackend(c).classify("text")
        assert len(r) == 2
        assert r[0] == ClassificationResult(label="toxic", score=0.9)
        assert r[1] == ClassificationResult(label="safe", score=0.1)


class TestVLLMBackend:
    _URL = "http://vllm:8000/classify"

    def _backend(self):
        return VLLMBackend(_remote(engine="vllm", base_url="http://vllm:8000"))

    @pytest.mark.asyncio
    async def test_success(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(url=self._URL, method="POST", json={"data": [{"label": "toxic", "probs": [0.9, 0.1]}]})
        r = await self._backend().classify("text")
        assert r[0] == ClassificationResult(label="toxic", score=0.9)

    @pytest.mark.asyncio
    async def test_empty_probs_defaults_to_zero(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(url=self._URL, method="POST", json={"data": [{"label": "safe", "probs": []}]})
        r = await self._backend().classify("text")
        assert r[0]["score"] == 0.0

    @pytest.mark.asyncio
    async def test_non_200(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(url=self._URL, method="POST", status_code=500, text="error")
        with pytest.raises(ValueError, match="500"):
            await self._backend().classify("text")

    @pytest.mark.asyncio
    async def test_missing_data_key(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(url=self._URL, method="POST", json={"wrong": []})
        with pytest.raises(ValueError, match="Unexpected vLLM"):
            await self._backend().classify("text")

    @pytest.mark.asyncio
    async def test_uses_injected_http_client(self):
        client = RecordingHTTPClient(
            [
                HTTPResponse(
                    status_code=200,
                    content=b'{"data":[{"label":"safe","probs":[0.95]}]}',
                )
            ]
        )
        backend = VLLMBackend(
            _remote(engine="vllm", base_url="http://vllm:8000"),
            http_client=client,
        )

        result = await backend.classify("text")

        assert result == [ClassificationResult(label="safe", score=0.95)]
        assert client.requests[0].url == self._URL
        assert client.close_calls == 0

    @pytest.mark.asyncio
    async def test_injected_http_client_preserves_transport_retry(self):
        client = RecordingHTTPClient(
            [
                HTTPConnectionError("offline"),
                HTTPResponse(
                    status_code=200,
                    content=b'{"data":[{"label":"safe","probs":[0.95]}]}',
                ),
            ]
        )
        backend = VLLMBackend(
            _remote(engine="vllm", base_url="http://vllm:8000"),
            http_client=client,
        )

        result = await backend.classify("text")

        assert result == [ClassificationResult(label="safe", score=0.95)]
        assert len(client.requests) == 2

    @pytest.mark.asyncio
    async def test_injected_instrumented_client_observes_all_retry_attempts(self, http_otel):
        tracer, exporter = http_otel
        transport = RecordingHTTPClient(
            [
                HTTPTimeoutError("timeout"),
                HTTPResponse(
                    status_code=200,
                    content=b'{"data":[{"label":"safe","probs":[0.95]}]}',
                ),
            ]
        )
        client = InstrumentedHTTPClient(transport, tracer)
        backend = VLLMBackend(
            _remote(engine="vllm", base_url="http://vllm:8000"),
            http_client=client,
        )

        result = await backend.classify("text")

        assert result == [ClassificationResult(label="safe", score=0.95)]
        assert len(transport.requests) == 2
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes["http.request.resend_count"] == 1

    @pytest.mark.asyncio
    async def test_injected_retry_policy_is_not_compounded(self, http_otel):
        tracer, exporter = http_otel
        transport = RecordingHTTPClient(
            [
                HTTPTimeoutError("first timeout"),
                HTTPTimeoutError("second timeout"),
                HTTPResponse(
                    status_code=200,
                    content=b'{"data":[{"label":"safe","probs":[0.95]}]}',
                ),
            ]
        )
        retrying = RetryingHTTPClient(
            transport,
            RetryPolicy(
                max_attempts=2,
                retryable_methods=frozenset({"POST"}),
                retryable_status_codes=frozenset(),
                initial_delay=0,
                max_delay=0,
            ),
        )
        client = InstrumentedHTTPClient(retrying, tracer)
        backend = VLLMBackend(
            _remote(engine="vllm", base_url="http://vllm:8000"),
            http_client=client,
        )

        with pytest.raises(HTTPTimeoutError, match="second timeout"):
            await backend.classify("text")

        assert len(transport.requests) == 2
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes["http.request.resend_count"] == 1

    @pytest.mark.asyncio
    async def test_injected_retrying_instrumented_client_is_normalized(self, http_otel):
        tracer, exporter = http_otel
        transport = RecordingHTTPClient(
            [
                HTTPTimeoutError("first timeout"),
                HTTPTimeoutError("second timeout"),
                HTTPResponse(
                    status_code=200,
                    content=b'{"data":[{"label":"safe","probs":[0.95]}]}',
                ),
            ]
        )
        instrumented = InstrumentedHTTPClient(transport, tracer)
        client = RetryingHTTPClient(
            instrumented,
            RetryPolicy(
                max_attempts=3,
                retryable_methods=frozenset({"POST"}),
                retryable_status_codes=frozenset(),
                initial_delay=0,
                max_delay=0,
            ),
        )
        backend = VLLMBackend(
            _remote(engine="vllm", base_url="http://vllm:8000"),
            http_client=client,
        )

        result = await backend.classify("text")

        assert result == [ClassificationResult(label="safe", score=0.95)]
        assert len(transport.requests) == 3
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes["http.request.resend_count"] == 2


class TestKServeBackend:
    _URL = "http://ks:8080/v1/models/m:predict"

    def _backend(self):
        return KServeBackend(_remote(engine="kserve", base_url="http://ks:8080", model="m"))

    @pytest.mark.asyncio
    async def test_dict_prediction(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(url=self._URL, method="POST", json={"predictions": [{"0": 0.1, "1": 0.9}]})
        r = await self._backend().classify("text")
        assert {x["label"] for x in r} == {"0", "1"}

    @pytest.mark.asyncio
    async def test_int_prediction(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(url=self._URL, method="POST", json={"predictions": [2]})
        r = await self._backend().classify("text")
        assert r == [ClassificationResult(label="2", score=1.0)]

    @pytest.mark.asyncio
    async def test_list_flattens_dedupes(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(url=self._URL, method="POST", json={"predictions": [[[0, 1, 2], [1, 0, 3]]]})
        r = await self._backend().classify("text")
        assert [x["label"] for x in r] == ["1", "2", "3"]

    @pytest.mark.asyncio
    async def test_empty_predictions(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(url=self._URL, method="POST", json={"predictions": []})
        assert await self._backend().classify("text") == []

    @pytest.mark.asyncio
    async def test_unknown_type_raises(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(url=self._URL, method="POST", json={"predictions": ["bad"]})
        with pytest.raises(ValueError, match="Unexpected KServe prediction type"):
            await self._backend().classify("text")

    @pytest.mark.asyncio
    async def test_non_200(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(url=self._URL, method="POST", status_code=500, text="error")
        with pytest.raises(ValueError, match="KServe predict returned 500"):
            await self._backend().classify("text")

    @pytest.mark.asyncio
    async def test_missing_predictions_key(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(url=self._URL, method="POST", json={"wrong": []})
        with pytest.raises(ValueError, match="Unexpected KServe predict response"):
            await self._backend().classify("text")


class TestFMSBackend:
    _URL = "http://fms:9000/api/v1/text/contents"

    def _backend(self):
        return FMSBackend(_remote(engine="fms", base_url="http://fms:9000"))

    @pytest.mark.asyncio
    async def test_success(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(url=self._URL, method="POST", json=[[{"detection_type": "harm", "score": 0.95}]])
        r = await self._backend().classify("text")
        assert r == [ClassificationResult(label="harm", score=0.95)]

    @pytest.mark.asyncio
    async def test_empty_detections(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(url=self._URL, method="POST", json=[[]])
        assert await self._backend().classify("text") == []

    @pytest.mark.asyncio
    async def test_non_200(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(url=self._URL, method="POST", status_code=503, text="down")
        with pytest.raises(ValueError, match="503"):
            await self._backend().classify("text")

    @pytest.mark.asyncio
    async def test_malformed_structure(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(url=self._URL, method="POST", json={"bad": []})
        with pytest.raises(ValueError, match="Unexpected FMS response"):
            await self._backend().classify("text")

    @pytest.mark.asyncio
    async def test_threshold_sent_in_payload(self, httpx_mock: HTTPXMock):
        cfg = _remote(engine="fms", base_url="http://fms:9000", threshold=0.75)
        backend = FMSBackend(cfg)
        httpx_mock.add_response(url=self._URL, method="POST", json=[[]])
        await backend.classify("hello")
        requests = httpx_mock.get_requests()
        assert len(requests) == 1
        body = json.loads(requests[0].content)
        assert body["detector_params"]["threshold"] == 0.75

    @pytest.mark.asyncio
    async def test_inner_not_a_list(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(url=self._URL, method="POST", json=[42])
        with pytest.raises(ValueError, match="Unexpected FMS response"):
            await self._backend().classify("text")

    @pytest.mark.asyncio
    async def test_malformed_detection_entry(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(url=self._URL, method="POST", json=[[{"wrong_key": "x"}]])
        with pytest.raises(ValueError, match="Unexpected FMS detection entry"):
            await self._backend().classify("text")


class TestGetBackend:
    @pytest.mark.parametrize(
        "name,cls",
        [
            ("local", LocalBackend),
            ("vllm", VLLMBackend),
            ("kserve", KServeBackend),
            ("fms", FMSBackend),
        ],
    )
    def test_known(self, name, cls):
        if name == "local":
            cfg = _local()
        else:
            cfg = _remote(engine=name)
        assert isinstance(get_backend(cfg), cls)

    def test_unknown_raises(self):
        c = RemoteHFClassifierConfig.model_construct(
            engine="bogus",
            model="x",
            base_url="http://x",
            threshold=0.5,
            blocked_labels=[],
            parameters={},
            api_key_env_var=None,
        )
        with pytest.raises(ValueError, match="Unknown hf_classifier engine"):
            get_backend(c)

    def test_remote_backend_uses_injected_http_client(self):
        client = RecordingHTTPClient()

        backend = get_backend(_remote(engine="vllm"), http_client=client)

        assert isinstance(backend, VLLMBackend)
        assert backend._http_client is client


class TestClassifyAndCheck:
    @pytest.mark.asyncio
    async def test_blocks_above_threshold(self):
        c = _remote(threshold=0.5, blocked_labels=["toxic"])
        with _patch_backend([ClassificationResult(label="toxic", score=0.8)]):
            assert await _classify_and_check("t", "bad", _rails_cfg("t", c)) is False

    @pytest.mark.asyncio
    async def test_allows_below_threshold(self):
        c = _remote(threshold=0.5, blocked_labels=["toxic"])
        with _patch_backend([ClassificationResult(label="toxic", score=0.3)]):
            assert await _classify_and_check("t", "ok", _rails_cfg("t", c)) is True

    @pytest.mark.asyncio
    async def test_blocks_at_exact_threshold(self):
        c = _remote(threshold=0.5, blocked_labels=["toxic"])
        with _patch_backend([ClassificationResult(label="toxic", score=0.5)]):
            assert await _classify_and_check("t", "edge", _rails_cfg("t", c)) is False

    @pytest.mark.asyncio
    async def test_allows_non_blocked_label(self):
        c = _remote(threshold=0.5, blocked_labels=["toxic"])
        with _patch_backend([ClassificationResult(label="safe", score=0.99)]):
            assert await _classify_and_check("t", "ok", _rails_cfg("t", c)) is True

    @pytest.mark.asyncio
    async def test_no_config_raises(self):
        with pytest.raises(ValueError, match="no 'hf_classifier' section"):
            await _classify_and_check("t", "text", None)

    @pytest.mark.asyncio
    async def test_unknown_classifier_raises(self):
        c = _remote()
        with pytest.raises(ValueError, match="Unknown classifier 'bad'"):
            await _classify_and_check("bad", "text", _rails_cfg("good", c))

    @pytest.mark.asyncio
    async def test_empty_results_warns(self, caplog):
        c = _local(threshold=0.5, blocked_labels=["toxic"])
        with _patch_backend([]):
            with caplog.at_level(logging.WARNING):
                result = await _classify_and_check("t", "text", _rails_cfg("t", c))
        assert result is True
        assert "returned no results" in caplog.text

    @pytest.mark.asyncio
    async def test_empty_text_no_warning(self, caplog):
        c = _remote(threshold=0.5, blocked_labels=["toxic"])
        with _patch_backend([]):
            with caplog.at_level(logging.WARNING):
                result = await _classify_and_check("t", "", _rails_cfg("t", c))
        assert result is True
        assert "returned no results" not in caplog.text

    @pytest.mark.asyncio
    async def test_remote_empty_results_no_warning(self, caplog):
        cfg = _remote(engine="fms", base_url="http://fms:9000", threshold=0.5, blocked_labels=["toxic"])
        with _patch_backend([]):
            with caplog.at_level(logging.WARNING):
                result = await _classify_and_check("t", "text", _rails_cfg("t", cfg))
        assert result is True
        assert "returned no results" not in caplog.text


class TestActionContextKeys:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "action_fn,context_key",
        [
            (hf_classifier_check_input, "user_message"),
            (hf_classifier_check_output, "bot_message"),
            (hf_classifier_check_retrieval, "relevant_chunks"),
        ],
    )
    async def test_reads_correct_context_key(self, action_fn, context_key):
        c = _remote(threshold=0.5, blocked_labels=["toxic"])
        cfg = _rails_cfg("t", c)
        with _patch_backend([ClassificationResult(label="toxic", score=0.9)]) as p:
            await action_fn(classifier="t", config=cfg, context={context_key: "bad"})
            p.return_value.classify.assert_called_once_with("bad")

    @pytest.mark.asyncio
    async def test_none_context_defaults_to_empty(self):
        c = _remote(threshold=0.5, blocked_labels=["toxic"])
        cfg = _rails_cfg("t", c)
        with _patch_backend([]):
            result = await hf_classifier_check_input(
                classifier="t",
                config=cfg,
                context=None,
            )
        assert result == RailOutcome.allow()


class TestOutputVerdict:
    def test_allow_outcome_is_not_blocked(self):
        outcome = RailOutcome.allow()

        assert outcome.is_blocked is False

    def test_block_outcome_is_blocked(self):
        outcome = RailOutcome.block()

        assert outcome.is_blocked is True


class TestRetrievalOutcome:
    def test_allowed_retrieval_maps_to_allow(self):
        assert _hf_classifier_retrieval_outcome(True) == RailOutcome.allow()

    def test_blocked_retrieval_maps_to_transform(self):
        assert _hf_classifier_retrieval_outcome(False) == RailOutcome.transform([(TransformTarget.RELEVANT_CHUNKS, "")])


class TestStreamingOutputFallback:
    @pytest.mark.asyncio
    async def test_dollar_classifier_falls_back_to_model_name(self):
        c = _remote(threshold=0.5, blocked_labels=["toxic"])
        cfg = _rails_cfg("hap", c)
        with _patch_backend([ClassificationResult(label="toxic", score=0.9)]):
            result = await hf_classifier_check_output(
                classifier="$classifier",
                config=cfg,
                context={"bot_message": "bad"},
                model_name="hap",
            )
        assert result == RailOutcome.block()

    @pytest.mark.asyncio
    async def test_dollar_classifier_without_model_name_raises(self):
        c = _remote(threshold=0.5, blocked_labels=["toxic"])
        cfg = _rails_cfg("hap", c)
        with _patch_backend([]):
            with pytest.raises(ValueError, match="Unknown classifier"):
                await hf_classifier_check_output(
                    classifier="$classifier",
                    config=cfg,
                    context={"bot_message": "text"},
                )


class TestSSLCerts:
    def test_ca_cert(self):
        cfg = _build_ssl_config(_remote(parameters={"ca_cert": "/ca.pem"}))
        assert cfg.verify is True
        assert cfg.ca_bundle == "/ca.pem"
        assert cfg.client_certificate is None

    def test_mtls(self):
        cfg = _build_ssl_config(_remote(parameters={"client_cert": "/cert.pem", "client_key": "/key.pem"}))
        assert cfg.verify is True
        assert cfg.client_certificate == "/cert.pem"
        assert cfg.client_key == "/key.pem"

    def test_ca_cert_plus_mtls(self):
        cfg = _build_ssl_config(
            _remote(
                parameters={
                    "ca_cert": "/ca.pem",
                    "client_cert": "/cert.pem",
                    "client_key": "/key.pem",
                }
            )
        )
        assert cfg.verify is True
        assert cfg.ca_bundle == "/ca.pem"
        assert cfg.client_certificate == "/cert.pem"
        assert cfg.client_key == "/key.pem"

    def test_client_cert_without_key_raises(self):
        with pytest.raises(ValueError, match="client_key"):
            _build_ssl_config(_remote(parameters={"client_cert": "/cert.pem"}))

    def test_client_key_without_cert_raises(self):
        with pytest.raises(ValueError, match="client_cert"):
            _build_ssl_config(_remote(parameters={"client_key": "/key.pem"}))


class TestLocalImportError:
    @pytest.mark.asyncio
    async def test_missing_transformers(self):
        c = _local()
        with mock.patch.dict("sys.modules", {"transformers": None}):
            with pytest.raises(ImportError, match="transformers"):
                await LocalBackend(c).classify("text")


class TestDiscriminatedUnion:
    _ta = TypeAdapter(HFClassifierConfig)

    def test_local_engine_resolves(self):
        c = self._ta.validate_python({"engine": "local", "model": "m", "blocked_labels": ["x"]})
        assert isinstance(c, LocalHFClassifierConfig)

    def test_remote_engine_resolves(self):
        c = self._ta.validate_python(
            {
                "engine": "vllm",
                "model": "m",
                "base_url": "http://x:8000",
                "blocked_labels": ["x"],
            }
        )
        assert isinstance(c, RemoteHFClassifierConfig)

    def test_invalid_engine_rejected(self):
        with pytest.raises(ValidationError, match="engine"):
            self._ta.validate_python({"engine": "bogus", "model": "m", "blocked_labels": ["x"]})

    def test_task_default(self):
        c = _local()
        assert c.task == "text-classification"

    def test_remote_has_no_task(self):
        assert "task" not in RemoteHFClassifierConfig.model_fields


class TestPipelineCacheKey:
    def test_http_params_excluded(self):
        base = {"device": "cpu"}
        with_http = {
            **base,
            "timeout": 10.0,
            "verify_ssl": False,
            "ca_cert": "/ca.pem",
            "client_cert": "/c.pem",
            "client_key": "/k.pem",
        }
        assert _pipeline_cache_key("m", "text-classification", base) == _pipeline_cache_key(
            "m", "text-classification", with_http
        )

    def test_model_params_included(self):
        k1 = _pipeline_cache_key("m", "text-classification", {"device": "cpu"})
        k2 = _pipeline_cache_key("m", "text-classification", {"device": "cuda"})
        assert k1 != k2


class TestPipelineConcurrency:
    def test_sentinel_not_overwritten_by_late_load(self):
        """If prewarm timeout stores a _PipelineLoadError while the pipeline is
        loading (outside the lock), the loaded pipeline must not overwrite it."""
        key = _pipeline_cache_key("slow-model", "text-classification", {})
        sentinel_placed = threading.Event()

        real_pipeline = mock.MagicMock(return_value=mock.MagicMock())

        def slow_pipeline(**kwargs):
            sentinel_placed.wait(timeout=5)
            return real_pipeline(**kwargs)

        mock_transformers = mock.MagicMock()
        mock_transformers.pipeline = slow_pipeline

        with mock.patch.dict(
            "sys.modules",
            {"transformers": mock_transformers, "nemoguardrails.library.hf_classifier.chunking": mock.MagicMock()},
        ):
            t = threading.Thread(
                target=_get_or_create_pipeline,
                args=("slow-model", "text-classification", {}),
            )
            t.start()

            for _ in range(200):
                time.sleep(0.01)
                if isinstance(backends_mod._pipelines.get(key), _PipelineLoading):
                    break
            assert isinstance(backends_mod._pipelines.get(key), _PipelineLoading)

            backends_mod._pipelines[key] = _PipelineLoadError("timed out")
            sentinel_placed.set()
            t.join(timeout=5)

        sentinel = backends_mod._pipelines.get(key)
        assert isinstance(sentinel, _PipelineLoadError), "Sentinel was overwritten by late pipeline load"

    def test_concurrent_callers_wait_for_first_load(self):
        """A second caller seeing _PipelineLoading should wait for the first
        loader to finish and receive the same pipeline."""
        key = _pipeline_cache_key("shared-model", "text-classification", {})
        load_started = threading.Event()
        mock_pipe = mock.MagicMock()

        def slow_pipeline(**kwargs):
            load_started.set()
            time.sleep(0.1)
            return mock_pipe

        mock_transformers = mock.MagicMock()
        mock_transformers.pipeline = slow_pipeline

        results = [None, None]

        def loader(idx):
            results[idx] = _get_or_create_pipeline("shared-model", "text-classification", {})

        with mock.patch.dict(
            "sys.modules",
            {"transformers": mock_transformers, "nemoguardrails.library.hf_classifier.chunking": mock.MagicMock()},
        ):
            t1 = threading.Thread(target=loader, args=(0,))
            t1.start()
            load_started.wait(timeout=5)

            t2 = threading.Thread(target=loader, args=(1,))
            t2.start()

            t1.join(timeout=5)
            t2.join(timeout=5)

        assert results[0] is mock_pipe
        assert results[1] is mock_pipe


class TestSentinelThroughLocalBackend:
    @pytest.mark.asyncio
    async def test_classify_raises_on_sentinel(self):
        cfg = _local()
        key = _pipeline_cache_key("test-model", "text-classification", {})
        backends_mod._pipelines[key] = _PipelineLoadError("download failed")
        with pytest.raises(RuntimeError, match="failed to load at startup"):
            await LocalBackend(cfg).classify("text")

    def test_sentinel_prevents_lazy_retry(self):
        key = _pipeline_cache_key("broken-model", "text-classification", {})
        backends_mod._pipelines[key] = _PipelineLoadError("download timed out")
        with pytest.raises(RuntimeError, match="failed to load at startup"):
            _get_or_create_pipeline("broken-model", "text-classification", {})

    def test_load_failure_stores_sentinel(self):
        mock_transformers = mock.MagicMock()
        mock_transformers.pipeline.side_effect = OSError("model not found")
        with mock.patch.dict(
            "sys.modules",
            {"transformers": mock_transformers, "nemoguardrails.library.hf_classifier.chunking": mock.MagicMock()},
        ):
            with pytest.raises(OSError, match="model not found"):
                _get_or_create_pipeline("missing-model", "text-classification", {})
        key = _pipeline_cache_key("missing-model", "text-classification", {})
        assert isinstance(backends_mod._pipelines.get(key), _PipelineLoadError)
        with pytest.raises(RuntimeError, match="failed to load at startup"):
            _get_or_create_pipeline("missing-model", "text-classification", {})


class TestConnectionError:
    @pytest.mark.asyncio
    async def test_vllm_connection_error(self, httpx_mock: HTTPXMock):
        backend = VLLMBackend(_remote(engine="vllm", base_url="http://vllm:8000"))
        httpx_mock.add_exception(httpx.ConnectError("Connection refused"), url="http://vllm:8000/classify")
        httpx_mock.add_exception(httpx.ConnectError("Connection refused"), url="http://vllm:8000/classify")
        with pytest.raises(HTTPConnectionError):
            await backend.classify("text")

    @pytest.mark.asyncio
    async def test_fms_connection_error(self, httpx_mock: HTTPXMock):
        backend = FMSBackend(_remote(engine="fms", base_url="http://fms:9000"))
        httpx_mock.add_exception(httpx.ConnectError("Connection refused"), url="http://fms:9000/api/v1/text/contents")
        httpx_mock.add_exception(httpx.ConnectError("Connection refused"), url="http://fms:9000/api/v1/text/contents")
        with pytest.raises(HTTPConnectionError):
            await backend.classify("text")

    @pytest.mark.asyncio
    async def test_retry_recovers_on_second_attempt(self, httpx_mock: HTTPXMock):
        backend = VLLMBackend(_remote(engine="vllm", base_url="http://vllm:8000"))
        httpx_mock.add_exception(httpx.ConnectError("Connection refused"), url="http://vllm:8000/classify")
        httpx_mock.add_response(
            url="http://vllm:8000/classify",
            json={"data": [{"label": "safe", "probs": [0.95]}]},
        )
        results = await backend.classify("text")
        assert len(results) == 1
        assert results[0]["label"] == "safe"

    @pytest.mark.asyncio
    async def test_timeout_retried(self, httpx_mock: HTTPXMock):
        backend = VLLMBackend(_remote(engine="vllm", base_url="http://vllm:8000"))
        httpx_mock.add_exception(httpx.ReadTimeout("timed out"), url="http://vllm:8000/classify")
        httpx_mock.add_response(
            url="http://vllm:8000/classify",
            json={"data": [{"label": "safe", "probs": [0.9]}]},
        )
        results = await backend.classify("text")
        assert results[0]["label"] == "safe"

    @pytest.mark.asyncio
    async def test_timeout_raises_after_retry(self, httpx_mock: HTTPXMock):
        backend = VLLMBackend(_remote(engine="vllm", base_url="http://vllm:8000"))
        httpx_mock.add_exception(httpx.ReadTimeout("timed out"), url="http://vllm:8000/classify")
        httpx_mock.add_exception(httpx.ReadTimeout("timed out"), url="http://vllm:8000/classify")
        with pytest.raises(HTTPTimeoutError):
            await backend.classify("text")


class TestBackendCaching:
    def test_local_backend_reports_ignored_http_client(self, caplog):
        caplog.set_level(logging.DEBUG, logger=backends_mod.__name__)

        backend = get_backend(_local(), http_client=RecordingHTTPClient())

        assert isinstance(backend, LocalBackend)
        assert "Ignoring injected HTTP client for local HF classifier backend." in caplog.text

    def test_get_backend_returns_same_instance(self):
        cfg = _remote(engine="vllm")
        first = get_backend(cfg, name="my_classifier")
        second = get_backend(cfg, name="my_classifier")
        assert first is second

    def test_different_names_return_different_instances(self):
        cfg = _remote(engine="vllm")
        a = get_backend(cfg, name="classifier_a")
        b = get_backend(cfg, name="classifier_b")
        assert a is not b


class TestLocalPipeline:
    """Offline pipeline checks with tiny models; other tests need no HF dependencies."""

    @pytest.fixture(autouse=True)
    def hf_modules(self):
        self.torch = pytest.importorskip("torch")
        self.transformers = pytest.importorskip("transformers")
        self.tokenizers = pytest.importorskip("tokenizers")
        from nemoguardrails.library.hf_classifier import chunking

        self.chunking = chunking

    @pytest.fixture
    def tokenizer(self):
        vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *[f"w{i}" for i in range(30)], "café", "危"]
        backend = self.tokenizers.Tokenizer(
            self.tokenizers.models.WordPiece(dict(zip(vocabulary, range(len(vocabulary)))), unk_token="[UNK]")
        )
        backend.pre_tokenizer = self.tokenizers.pre_tokenizers.Whitespace()
        backend.decoder = self.tokenizers.decoders.WordPiece()
        backend.post_processor = self.tokenizers.processors.TemplateProcessing(
            single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
        )
        return self.transformers.PreTrainedTokenizerFast(
            tokenizer_object=backend,
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            model_max_length=16,
        )

    @pytest.fixture(params=["Bert"])
    def model_family(self, request):
        if not hasattr(self.transformers, f"{request.param}Config"):
            pytest.skip(f"{request.param} is unavailable in Transformers {self.transformers.__version__}")
        return request.param

    @pytest.fixture(params=["text-classification"])
    def classifiers(self, request, tokenizer, model_family):
        task = request.param
        options = dict(
            vocab_size=len(tokenizer),
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=32,
            id2label={0: "O", 1: "B-toxic"},
            label2id={"O": 0, "B-toxic": 1},
        )
        if model_family == "Bert":
            options.update(hidden_dropout_prob=0, attention_probs_dropout_prob=0)
        else:
            tokenizer.model_input_names = ["input_ids", "attention_mask"]
            options.update(
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.cls_token_id,
                eos_token_id=tokenizer.sep_token_id,
                cls_token_id=tokenizer.cls_token_id,
                sep_token_id=tokenizer.sep_token_id,
                num_hidden_layers=2,
                local_attention=8,
                reference_compile=False,
                attn_implementation="eager",
            )
        config = getattr(self.transformers, f"{model_family}Config")(**options)
        model_type = "Sequence" if task == "text-classification" else "Token"
        with self.torch.random.fork_rng(devices=[]):
            self.torch.manual_seed(0)
            model = getattr(self.transformers, f"{model_family}For{model_type}Classification")(config)
        adapter = (
            self.chunking.ChunkedTextClassificationPipeline
            if task == "text-classification"
            else self.transformers.TokenClassificationPipeline
        )
        model.eval()
        original = self.transformers.pipeline(task, model=model, tokenizer=tokenizer, device="cpu")
        chunked = self.transformers.pipeline(
            task, model=model, tokenizer=tokenizer, device="cpu", pipeline_class=adapter
        )
        return original, chunked

    @pytest.mark.parametrize(
        "size,options",
        [
            (0, {}),
            (25, {}),
            (25, {"top_k": None, "function_to_apply": "sigmoid"}),
            (25, {"top_k": 2}),
            (60, {"truncation": True}),
            (60, {"max_length": 12}),
            (60, {"max_length": 12, "truncation": True, "stride": 3, "return_overflowing_tokens": False}),
            (60, {"max_length": 12, "truncation": "only_first", "padding": "max_length"}),
            (25, {"truncation": False, "padding": True}),
        ],
    )
    def test_native_settings_match_original(self, classifiers, size, options):
        original, chunked = classifiers
        original = type(original)(model=original.model, tokenizer=original.tokenizer, device="cpu", **options)
        chunked = type(chunked)(model=chunked.model, tokenizer=chunked.tokenizer, device="cpu", **options)
        text = " ".join(f"w{i % 30}" for i in range(size))
        assert chunked(text) == original(text)

    @pytest.mark.parametrize("top_k", [None, 1])
    def test_window_scores_match_native_hf(self, classifiers, top_k):
        original, chunked = classifiers
        text = " ".join(f"w{i % 30}" for i in range(60))
        encoded = original.tokenizer(text, max_length=32, stride=3, truncation=True, return_overflowing_tokens=True)
        options = {"top_k": top_k, "function_to_apply": "sigmoid"}
        expected = [
            score
            for ids in encoded["input_ids"]
            for score in original(original.tokenizer.decode(ids, skip_special_tokens=True), **options)
        ]
        assert chunked(text, stride=3, **options) == expected

    @pytest.mark.parametrize("stride,special_tokens,overlap", [(None, True, 7), (0, True, 0), (3, False, 3)])
    def test_windows_cover_all_tokens(self, classifiers, stride, special_tokens, overlap):
        _, chunked = classifiers
        text = " ".join([*[f"w{i}" for i in range(30)], *["café", "危"] * 20])
        options = {"add_special_tokens": special_tokens}
        if stride is not None:
            options["stride"] = stride
        windows = list(chunked.preprocess(text, **options))
        content = []
        for window in windows:
            assert window["input_ids"].shape[1] <= 32
            ids = window["input_ids"][0][window["attention_mask"][0].bool()].tolist()
            content.append(ids[1:-1] if special_tokens else ids)
        assert [window["is_last"] for window in windows] == [False] * (len(windows) - 1) + [True]
        reconstructed = content[0] + [token for window in content[1:] for token in window[overlap:]]
        assert reconstructed == chunked.tokenizer.encode(text, add_special_tokens=False)

    @pytest.mark.parametrize("truncation", [None, False, "do_not_truncate"])
    def test_previously_failing_input_succeeds(self, classifiers, truncation):
        original, chunked = classifiers
        chunked.tokenizer.pad_token = None
        text = " ".join(["w0"] * 60)
        options = {} if truncation is None else {"truncation": truncation}
        with pytest.raises((RuntimeError, IndexError)):
            original(text, **options)
        with mock.patch.object(chunked, "_forward", wraps=chunked._forward) as forward:
            chunked(text, **options)
        assert forward.call_count > 1
        assert all(call.args[0]["input_ids"].shape[-1] <= 32 for call in forward.call_args_list)

    @pytest.mark.parametrize("model_family", ["ModernBert"], indirect=True)
    def test_rotary_architecture_handles_short_and_long_inputs(self, classifiers):
        original, chunked = classifiers
        assert chunked("w0 w1") == original("w0 w1")
        with mock.patch.object(chunked, "_forward", wraps=chunked._forward) as forward:
            assert chunked(" ".join(["w0"] * 60))
        assert forward.call_count > 1
        assert all(call.args[0]["input_ids"].shape[-1] <= 32 for call in forward.call_args_list)

    def test_configured_truncation_never_rescans_original_text(self, classifiers):
        original, chunked = classifiers
        text = " ".join(["w0"] * 60)
        options = {"max_length": 40, "truncation": True}
        windows = list(chunked.preprocess(text, **options))
        assert len(windows) == 1
        assert self.torch.equal(windows[0]["input_ids"], original.preprocess(text, **options)["input_ids"])

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "classifiers,parameters,checks_tail",
        [
            (
                "text-classification",
                {"max_length": 12, "stride": 3, "truncation": True, "return_overflowing_tokens": False},
                False,
            ),
            ("token-classification", {}, False),
            ("token-classification", {"stride": 0, "aggregation_strategy": "simple"}, True),
            ("token-classification", {"stride": 3, "aggregation_strategy": "max", "batch_size": 2}, True),
        ],
        indirect=["classifiers"],
    )
    async def test_factory_preserves_configuration(self, classifiers, parameters, checks_tail, tmp_path, monkeypatch):
        original, adapted = classifiers
        model_path = tmp_path / "tiny"
        original.save_pretrained(model_path)
        monkeypatch.setattr(backends_mod, "_pipelines", {})
        config = RailsConfig.from_content(
            config={
                "rails": {
                    "config": {
                        "hf_classifier": {
                            "test": {
                                "engine": "local",
                                "model": str(model_path),
                                "task": original.task,
                                "parameters": {**parameters, "device": "cpu"},
                            }
                        }
                    }
                }
            }
        ).rails.config.hf_classifier["test"]
        loaded = backends_mod._get_or_create_pipeline(config.model, config.task, config.parameters)
        assert type(loaded) is type(adapted)
        assert loaded is backends_mod._get_or_create_pipeline(config.model, config.task, config.parameters)
        text = " ".join(["w0"] * 59 + ["w24"])
        with mock.patch.object(loaded, "_forward", wraps=loaded._forward) as forward:
            actual = await backends_mod.LocalBackend(config).classify(text)
        expected = original(text, **parameters)
        if original.task == "token-classification":
            expected = [
                {"label": item.get("entity_group") or item.get("entity", ""), "score": item["score"]}
                for item in expected
            ]
        assert len(actual) == len(expected)
        for result, reference in zip(actual, expected):
            assert result == pytest.approx(reference)
        tail_token = loaded.tokenizer.convert_tokens_to_ids("w24")
        assert any((call.args[0]["input_ids"] == tail_token).any() for call in forward.call_args_list) == checks_tail

    @pytest.fixture
    def rail_config(self):
        return RailsConfig.from_content(
            yaml_content="""
    rails:
      config:
        hf_classifier:
          test:
            engine: local
            model: tiny
            blocked_labels: [B-toxic]
    """
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "action,blocked",
        [
            (hf_classifier_check_input, RailOutcome.block()),
            (hf_classifier_check_output, RailOutcome.block()),
            (hf_classifier_check_retrieval, RailOutcome.transform([(TransformTarget.RELEVANT_CHUNKS, "")])),
        ],
    )
    async def test_final_window_detection_blocks(self, classifiers, rail_config, monkeypatch, action, blocked):
        _, chunked = classifiers
        forward = chunked._forward
        target = chunked.tokenizer.convert_tokens_to_ids("w24")

        def detect_target(inputs):
            hits = (inputs["input_ids"] == target).any(dim=1)
            output = forward(inputs)
            output["logits"] = self.torch.stack([(~hits).float() * 10, hits.float() * 10], dim=-1)
            return output

        monkeypatch.setattr(chunked, "_forward", detect_target)
        monkeypatch.setattr(backends_mod, "_get_or_create_pipeline", lambda *args: chunked)
        text = " ".join(["w0"] * 59 + ["w24"])
        context = {"user_message": text, "bot_message": text, "relevant_chunks": text}
        assert await action(context=context, config=rail_config, classifier="test") == blocked

    def test_overlap_preserves_boundary_detection(self, classifiers, monkeypatch):
        _, chunked = classifiers
        forward = chunked._forward
        left, right = chunked.tokenizer.convert_tokens_to_ids(["w28", "w29"])

        def detect_pair(inputs):
            ids = inputs["input_ids"]
            hits = (ids == left).any(dim=1) & (ids == right).any(dim=1)
            output = forward(inputs)
            output["logits"] = self.torch.stack([(~hits).float() * 10, hits.float() * 10], dim=-1)
            return output

        monkeypatch.setattr(chunked, "_forward", detect_pair)
        text = " ".join(["w0"] * 29 + ["w28", "w29"] + ["w0"] * 29)
        assert not any(item["label"] == "B-toxic" for item in chunked(text, stride=0)[0])
        assert any(item["label"] == "B-toxic" for item in chunked(text)[0])

    def test_window_failure_propagates(self, classifiers):
        _, chunked = classifiers
        forward = chunked._forward
        calls = 0

        def fail_second(inputs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("window failed")
            return forward(inputs)

        with mock.patch.object(chunked, "_forward", side_effect=fail_second):
            with pytest.raises(RuntimeError, match="window failed"):
                chunked(" ".join(["w0"] * 60))
        assert calls == 2

    def test_unrelated_native_inference_error_is_not_retried(self, classifiers):
        _, chunked = classifiers
        with mock.patch.object(chunked, "_forward", side_effect=RuntimeError("inference failed")) as forward:
            with pytest.raises(RuntimeError, match="inference failed"):
                chunked("w0")
        assert forward.call_count == 1

    def test_unknown_limit_preserves_native_path_and_accepts_existing_max_length(self, classifiers):
        original, chunked = classifiers
        chunked.tokenizer.model_max_length = int(1e30)
        chunked.model.config.max_position_embeddings = int(1e30)
        assert chunked("w0") == original("w0")
        text = " ".join(["w0"] * 60)
        windows = list(chunked.preprocess(text, max_length=12, truncation=False))
        assert len(windows) > 1
        assert max(window["input_ids"].shape[-1] for window in windows) == 12
        chunked(text, max_length=12, truncation=False)

    @pytest.mark.parametrize(
        "model_limit,tokenizer_limit,expected", [(32, int(1e30), 32), (32, 512, 32), (int(1e30), 16, 16)]
    )
    def test_capacity_metadata(self, classifiers, model_limit, tokenizer_limit, expected):
        _, chunked = classifiers
        chunked.model.config.max_position_embeddings = model_limit
        chunked.tokenizer.model_max_length = tokenizer_limit
        windows = list(chunked.preprocess(" ".join(["w0"] * 60)))
        assert max(window["input_ids"].shape[-1] for window in windows) == expected
        chunked(" ".join(["w0"] * 60))

    def test_reserved_position_embeddings_are_excluded(self, tokenizer):
        tokenizer.model_max_length = int(1e30)
        config = self.transformers.RobertaConfig(
            vocab_size=len(tokenizer),
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=18,
        )
        pipe = self.chunking.ChunkedTextClassificationPipeline(
            model=self.transformers.RobertaForSequenceClassification(config).eval(), tokenizer=tokenizer, device="cpu"
        )
        windows = list(pipe.preprocess(" ".join(["w0"] * 60)))
        assert max(window["input_ids"].shape[1] for window in windows) == 16
        pipe(" ".join(["w0"] * 60))

    def test_slow_tokenizer_native_path_preserved(self, classifiers, monkeypatch):
        original, chunked = classifiers
        monkeypatch.setattr(type(chunked.tokenizer), "is_fast", property(lambda self: False))
        assert chunked("w0") == original("w0")
        assert chunked(" ".join(["w0"] * 60), truncation=True) == original(" ".join(["w0"] * 60), truncation=True)
        with pytest.raises(ValueError, match="fast tokenizer"):
            chunked(" ".join(["w0"] * 60))

    def test_overlap_must_leave_room_for_content(self, classifiers):
        _, chunked = classifiers
        with pytest.raises(ValueError, match="content token capacity"):
            chunked(" ".join(["w0"] * 60), stride=30)

    def test_padding_cannot_exceed_window_budget(self, classifiers):
        _, chunked = classifiers
        with pytest.raises(ValueError, match="padding"):
            chunked(" ".join(["w0"] * 60), padding=True, pad_to_multiple_of=64)

    def test_fallback_respects_requested_batching(self, classifiers):
        _, chunked = classifiers
        text = " ".join(["w0"] * 60)
        expected = chunked(text)[0]
        with mock.patch.object(chunked, "_forward", wraps=chunked._forward) as forward:
            actual = chunked(text, batch_size=2)[0]
        assert any(call.args[0]["input_ids"].shape[0] > 1 for call in forward.call_args_list)
        assert len(actual) == len(expected)
        for result, reference in zip(actual, expected):
            assert result == pytest.approx(reference)

    def test_postprocessing_does_not_retain_all_window_logits(self, classifiers):
        _, chunked = classifiers
        order = []
        forward = chunked._forward
        parent = self.transformers.TextClassificationPipeline
        score = parent.postprocess

        def infer(inputs):
            order.append("forward")
            return forward(inputs)

        def score_window(self, outputs, **kwargs):
            order.append("score")
            return score(self, outputs, **kwargs)

        with mock.patch.object(chunked, "_forward", infer), mock.patch.object(parent, "postprocess", score_window):
            chunked(" ".join(["w0"] * 60))
        assert len(order) > 2
        assert order[::2] == ["forward"] * (len(order) // 2)
        assert order[1::2] == ["score"] * (len(order) // 2)

    def test_pipeline_can_be_pickled_for_spawned_workers(self, classifiers):
        _, chunked = classifiers
        restored = pickle.loads(pickle.dumps(chunked))
        assert restored("w0 w1") == chunked("w0 w1")

    @pytest.mark.asyncio
    async def test_concurrent_short_and_long_calls_share_pipeline(self, classifiers, rail_config, monkeypatch):
        _, chunked = classifiers
        monkeypatch.setattr(backends_mod, "_get_or_create_pipeline", lambda *args: chunked)
        backend = backends_mod.LocalBackend(rail_config.rails.config.hf_classifier["test"])
        texts = ["w0", " ".join(["w0"] * 60)]
        expected = [await backend.classify(text) for text in texts]
        results = await asyncio.gather(*(backend.classify(text) for _ in range(4) for text in texts))
        assert results == expected * 4
        assert chunked.tokenizer.model_max_length == 16
