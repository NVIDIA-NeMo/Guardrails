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

import json
import logging
import threading
import time
from types import SimpleNamespace
from unittest import mock

import httpx
import pytest
from pydantic import TypeAdapter, ValidationError
from pytest_httpx import HTTPXMock

from nemoguardrails.actions.output_mapping import is_output_blocked
from nemoguardrails.library.hf_classifier import backends as backends_mod
from nemoguardrails.library.hf_classifier.actions import (
    _classify_and_check,
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
    RemoteHFClassifierConfig,
)

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


class TestConfig:
    def test_local_has_no_base_url(self):
        assert "base_url" not in LocalHFClassifierConfig.model_fields

    @pytest.mark.parametrize("engine", ["vllm", "kserve", "fms"])
    def test_remote_requires_base_url(self, engine):
        with pytest.raises(ValidationError, match="base_url"):
            RemoteHFClassifierConfig(engine=engine, model="m", blocked_labels=["x"])

    def test_invalid_base_url_scheme(self):
        with pytest.raises(ValidationError, match="http://"):
            _remote(base_url="ftp://host:8000")

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
        h = _build_headers(_remote(api_key_env_var="K"))
        assert h["Authorization"] == "Bearer secret"

    def test_missing_key_warns_once(self, caplog):
        c = _remote(api_key_env_var="MISSING")
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
        assert _get_timeout(_remote()) == httpx.Timeout(30.0)

    def test_timeout_custom(self):
        assert _get_timeout(_remote(parameters={"timeout": 10.0})) == httpx.Timeout(10.0)

    def test_ssl_default(self):
        cfg = _build_ssl_config(_remote())
        assert cfg.verify is True
        assert cfg.cert is None

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
        assert result is True


class TestOutputMapping:
    def test_allowed_maps_to_not_blocked(self):
        assert is_output_blocked(True, hf_classifier_check_output) is False

    def test_blocked_maps_to_blocked(self):
        assert is_output_blocked(False, hf_classifier_check_output) is True

    def test_has_explicit_output_mapping(self):
        meta = getattr(hf_classifier_check_output, "action_meta", {})
        assert meta.get("output_mapping") is not None


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
        assert result is False

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
        assert cfg.verify == "/ca.pem"
        assert cfg.cert is None

    def test_mtls(self):
        cfg = _build_ssl_config(_remote(parameters={"client_cert": "/cert.pem", "client_key": "/key.pem"}))
        assert cfg.verify is True
        assert cfg.cert == ("/cert.pem", "/key.pem")

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
        assert cfg.verify == "/ca.pem"
        assert cfg.cert == ("/cert.pem", "/key.pem")

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

        with mock.patch.dict("sys.modules", {"transformers": mock_transformers}):
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

        with mock.patch.dict("sys.modules", {"transformers": mock_transformers}):
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
        with mock.patch.dict("sys.modules", {"transformers": mock_transformers}):
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
        with pytest.raises(httpx.ConnectError):
            await backend.classify("text")

    @pytest.mark.asyncio
    async def test_fms_connection_error(self, httpx_mock: HTTPXMock):
        backend = FMSBackend(_remote(engine="fms", base_url="http://fms:9000"))
        httpx_mock.add_exception(httpx.ConnectError("Connection refused"), url="http://fms:9000/api/v1/text/contents")
        httpx_mock.add_exception(httpx.ConnectError("Connection refused"), url="http://fms:9000/api/v1/text/contents")
        with pytest.raises(httpx.ConnectError):
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
        with pytest.raises(httpx.ReadTimeout):
            await backend.classify("text")


class TestBackendCaching:
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
