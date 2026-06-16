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

"""Pluggable inference backends for HuggingFace classifier rails."""

from __future__ import annotations

import abc
import asyncio
import json
import logging
import os
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TypedDict, Union

import httpx

if TYPE_CHECKING:
    from nemoguardrails.rails.llm.config import (
        HFClassifierConfig,
        LocalHFClassifierConfig,
        RemoteHFClassifierConfig,
    )

log = logging.getLogger(__name__)


class ClassificationResult(TypedDict):
    label: str
    score: float


class ClassifierBackend(abc.ABC):
    """Abstract interface for HuggingFace classifier inference."""

    @abc.abstractmethod
    async def classify(self, text: str) -> List[ClassificationResult]:
        """Classify a single text. Returns list of label/score detections."""
        ...

    async def close(self) -> None:
        """Release resources (e.g. HTTP sessions). Override in subclasses."""


_DEFAULT_TIMEOUT = 30.0
_warned_env_vars: set = set()


def _build_headers(config: RemoteHFClassifierConfig) -> Dict[str, str]:
    """Build HTTP request headers from classifier config."""
    headers: Dict[str, str] = {"Content-Type": "application/json"}

    if config.api_key_env_var:
        api_key = os.environ.get(config.api_key_env_var)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        elif config.api_key_env_var not in _warned_env_vars:
            _warned_env_vars.add(config.api_key_env_var)
            log.warning(
                "api_key_env_var '%s' is configured but not set in the environment.",
                config.api_key_env_var,
            )

    return headers


def _get_timeout(config: RemoteHFClassifierConfig) -> httpx.Timeout:
    total = config.parameters.get("timeout", _DEFAULT_TIMEOUT)
    return httpx.Timeout(total)


class _HTTPXSSLConfig:
    """Stores httpx-compatible SSL parameters (verify, cert)."""

    def __init__(
        self,
        verify: Union[str, bool] = True,
        cert: Optional[tuple] = None,
    ) -> None:
        self.verify = verify
        self.cert = cert


_ssl_cache: Dict[tuple, _HTTPXSSLConfig] = {}


def _build_ssl_config(config: RemoteHFClassifierConfig) -> _HTTPXSSLConfig:
    """Build httpx SSL params from config parameters (cached).

    Reads from ``config.parameters``:
      - ``verify_ssl`` (bool, default True): set to False to skip TLS verification.
      - ``ca_cert`` (str): path to a CA bundle file for custom/internal CAs.
      - ``client_cert`` (str) + ``client_key`` (str): paths for mTLS client auth.
    """
    params = config.parameters
    verify = params.get("verify_ssl", True)
    ca_cert: Optional[str] = params.get("ca_cert")
    client_cert: Optional[str] = params.get("client_cert")
    client_key: Optional[str] = params.get("client_key")

    cache_key = (verify, ca_cert, client_cert, client_key)
    if cache_key in _ssl_cache:
        return _ssl_cache[cache_key]

    if bool(client_cert) != bool(client_key):
        provided, missing = ("client_cert", "client_key") if client_cert else ("client_key", "client_cert")
        raise ValueError(f"mTLS requires both 'client_cert' and 'client_key'; got '{provided}' without '{missing}'.")

    cert = (client_cert, client_key) if client_cert else None
    if verify is False:
        result = _HTTPXSSLConfig(verify=False, cert=cert)
    elif ca_cert:
        result = _HTTPXSSLConfig(verify=ca_cert, cert=cert)
    else:
        result = _HTTPXSSLConfig(verify=True, cert=cert)

    _ssl_cache[cache_key] = result
    return result


_pipelines: Dict[str, Any] = {}
_pipelines_lock = threading.Lock()
_HTTP_ONLY_PARAMS = frozenset(
    {
        "timeout",
        "verify_ssl",
        "ca_cert",
        "client_cert",
        "client_key",
    }
)


class _PipelineLoadError:
    """Sentinel stored in _pipelines when a pipeline fails to load at startup.
    Prevents silent retry — classify() raises immediately with the original error."""

    def __init__(self, error: str) -> None:
        self.error = error


class _PipelineLoading:
    """Sentinel stored while a pipeline is being loaded.
    Prevents duplicate concurrent loads — a second caller waits on the event."""

    def __init__(self) -> None:
        self.ready = threading.Event()
        self.result: Any = None
        self.error: Optional[BaseException] = None

    _WAIT_TIMEOUT = 600.0

    def wait(self) -> Any:
        if not self.ready.wait(timeout=self._WAIT_TIMEOUT):
            raise RuntimeError(
                f"Timed out after {self._WAIT_TIMEOUT}s waiting for another thread to finish loading an HF pipeline."
            )
        if self.error is not None:
            raise self.error
        return self.result


_FAILED_LOAD_MSG = (
    "HF pipeline '{model}' failed to load at startup: {error}. "
    "Check server logs for details. To use a local model, set "
    "model to a filesystem path. For air-gapped environments, "
    "set HF_HUB_OFFLINE=1 and ensure the model is cached."
)


def _pipeline_cache_key(model_name: str, task: str, parameters: Dict[str, Any]) -> str:
    filtered = {k: v for k, v in sorted(parameters.items()) if k not in _HTTP_ONLY_PARAMS}
    return f"{task}:{model_name}:{json.dumps(filtered, sort_keys=True, default=str)}"


def _check_cached(cache_key: str, model_name: str) -> Any:
    """Return cached pipeline, wait on loading, or raise on error. Returns None on miss."""
    cached = _pipelines.get(cache_key)
    if isinstance(cached, _PipelineLoadError):
        raise RuntimeError(_FAILED_LOAD_MSG.format(model=model_name, error=cached.error))
    if isinstance(cached, _PipelineLoading):
        return cached.wait()
    return cached


def _get_or_create_pipeline(
    model_name: str,
    task: str,
    parameters: Dict[str, Any],
) -> Any:
    cache_key = _pipeline_cache_key(model_name, task, parameters)

    hit = _check_cached(cache_key, model_name)
    if hit is not None:
        return hit

    with _pipelines_lock:
        cached = _pipelines.get(cache_key)
        if isinstance(cached, _PipelineLoadError):
            raise RuntimeError(_FAILED_LOAD_MSG.format(model=model_name, error=cached.error))
        if isinstance(cached, _PipelineLoading):
            pending = cached
        elif cached is not None:
            return cached
        else:
            pending = None
            loading = _PipelineLoading()
            _pipelines[cache_key] = loading

    if pending is not None:
        return pending.wait()

    try:
        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError(
                "The 'transformers' package is required for the local HF classifier "
                "backend. Install it with: pip install nemoguardrails[hf-classifier]"
            )
        kwargs = {k: v for k, v in parameters.items() if k not in _HTTP_ONLY_PARAMS}
        pipe = pipeline(task=task, model=model_name, **kwargs)
    except BaseException as exc:
        loading.error = exc
        loading.ready.set()
        with _pipelines_lock:
            if _pipelines.get(cache_key) is loading:
                _pipelines[cache_key] = _PipelineLoadError(str(exc))
        raise

    with _pipelines_lock:
        current = _pipelines.get(cache_key)
        if isinstance(current, _PipelineLoadError):
            log.warning(
                "HF pipeline '%s' loaded after prewarm timeout; discarding to preserve sentinel.",
                model_name,
            )
            loading.result = pipe
            loading.ready.set()
            return pipe
        _pipelines[cache_key] = pipe
        loading.result = pipe
        loading.ready.set()

    log.info("Loaded HF pipeline: task=%s model=%s", task, model_name)
    return pipe


class LocalBackend(ClassifierBackend):
    """Local HF Transformers pipeline backend.

    Tested against: ``transformers >= 4.35`` pipeline API for
    ``text-classification`` and ``token-classification`` tasks.
    """

    def __init__(self, config: LocalHFClassifierConfig) -> None:
        self._config = config

    async def classify(self, text: str) -> List[ClassificationResult]:
        loop = asyncio.get_running_loop()

        def _run():
            pipe = _get_or_create_pipeline(
                self._config.model,
                self._config.task,
                self._config.parameters,
            )
            return pipe(text)

        raw = await loop.run_in_executor(None, _run)

        # Transformers may wrap results in an extra list for single-text input
        if raw and isinstance(raw[0], list):
            raw = raw[0]

        results: List[ClassificationResult] = []
        for item in raw:
            if self._config.task == "text-classification":
                results.append(ClassificationResult(label=item["label"], score=item["score"]))
            else:
                label = item.get("entity_group") or item.get("entity", "")
                results.append(ClassificationResult(label=label, score=item["score"]))
        return results


_TRANSIENT_ERRORS = (httpx.NetworkError, httpx.TimeoutException, httpx.RemoteProtocolError, OSError, ValueError)


class _RemoteBackend(ClassifierBackend):
    """Base class for remote HTTP classifier backends.

    Uses a fresh httpx client per request — no connection pool state to manage.
    Classifier calls are infrequent and latency-tolerant (inference >> TLS
    handshake), so connection reuse provides negligible benefit while introducing
    stale-connection complexity.
    """

    def __init__(self, config: RemoteHFClassifierConfig) -> None:
        self._config = config
        self._timeout = _get_timeout(config)
        self._ssl = _build_ssl_config(config)

    async def _post(self, url: str, json: Dict[str, Any]) -> httpx.Response:
        """POST with one automatic retry on transient failures."""
        headers = _build_headers(self._config)
        for attempt in range(2):
            try:
                async with httpx.AsyncClient(
                    timeout=self._timeout, verify=self._ssl.verify, cert=self._ssl.cert
                ) as client:
                    return await client.post(url, json=json, headers=headers)
            except _TRANSIENT_ERRORS:
                if attempt > 0:
                    raise
                log.debug("Classifier request to %s failed, retrying.", url)
        raise RuntimeError("unreachable")


class VLLMBackend(_RemoteBackend):
    """vLLM ``/classify`` endpoint backend.

    Tested against: vLLM v0.6.x ``/classify`` API. Expects response shape::

        {"data": [{"label": "...", "probs": [float, ...]}]}

    Raises ``ValueError`` if the response is missing required keys (``data``,
    ``label``), indicating an API change.
    """

    def __init__(self, config: RemoteHFClassifierConfig) -> None:
        super().__init__(config)
        self._url = config.base_url + "/classify"
        self._model_name = config.model

    async def classify(self, text: str) -> List[ClassificationResult]:
        payload = {"model": self._model_name, "input": text}

        resp = await self._post(self._url, json=payload)
        if resp.status_code != 200:
            raise ValueError(f"vLLM /classify returned {resp.status_code}: {resp.text[:500]}")
        data = resp.json()

        try:
            items = data["data"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Unexpected vLLM /classify response structure: {exc}. Raw: {str(data)[:500]}") from exc

        results: List[ClassificationResult] = []
        for item in items:
            try:
                label = item["label"]
            except (KeyError, TypeError) as exc:
                raise ValueError(f"vLLM /classify item missing 'label': {exc}. Raw item: {str(item)[:200]}") from exc
            probs = item.get("probs", [])
            if not probs:
                log.warning("vLLM /classify item for label '%s' has no probs; treating as score=0.0", label)
            results.append(
                ClassificationResult(
                    label=label,
                    score=max(probs) if probs else 0.0,
                )
            )
        return results


class KServeBackend(_RemoteBackend):
    """KServe v1 inference predict backend.

    Tested against: KServe v1 predict API (``/v1/models/{name}:predict``).
    Handles three ``predictions`` shapes:

    - ``dict``: class-index → probability (``--return_probabilities``)
    - ``int/float``: argmax class index, score assumed 1.0
    - nested ``list``: token-level class indices, flattened and deduplicated

    Raises ``ValueError`` if ``predictions`` key is missing or the prediction
    has an unrecognised type.
    """

    def __init__(self, config: RemoteHFClassifierConfig) -> None:
        super().__init__(config)
        self._url = f"{config.base_url}/v1/models/{config.model}:predict"

    async def classify(self, text: str) -> List[ClassificationResult]:
        payload: Dict[str, Any] = {"instances": [text]}

        resp = await self._post(self._url, json=payload)
        if resp.status_code != 200:
            raise ValueError(f"KServe predict returned {resp.status_code}: {resp.text[:500]}")
        data = resp.json()

        try:
            predictions = data["predictions"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Unexpected KServe predict response structure: {exc}. Raw: {str(data)[:500]}") from exc

        if not predictions:
            return []

        pred = predictions[0]
        if isinstance(pred, dict):
            return [ClassificationResult(label=cls_idx, score=float(score)) for cls_idx, score in pred.items()]
        if isinstance(pred, (int, float)):
            return [ClassificationResult(label=str(int(pred)), score=1.0)]
        if isinstance(pred, list):
            flat = _flatten_ints(pred)
            return [ClassificationResult(label=str(cls), score=1.0) for cls in sorted(set(flat)) if cls != 0]
        raise ValueError(f"Unexpected KServe prediction type: {type(pred).__name__}. Raw: {str(data)[:500]}")


def _flatten_ints(nested: Any) -> List[int]:
    out: List[int] = []
    stack = [nested]
    while stack:
        item = stack.pop()
        if isinstance(item, (int, float)):
            out.append(int(item))
        elif isinstance(item, list):
            stack.extend(item)
        else:
            raise ValueError(f"Unexpected type in KServe prediction list: {type(item).__name__}")
    return out


class FMSBackend(_RemoteBackend):
    """FMS guardrails-detectors ``/api/v1/text/contents`` backend.

    Tested against: FMS guardrails-detectors API v1. Expects response shape::

        [[{"detection_type": "...", "score": float}, ...]]

    Raises ``ValueError`` if the response is not a list-of-lists or detection
    entries lack required keys.
    """

    def __init__(self, config: RemoteHFClassifierConfig) -> None:
        super().__init__(config)
        self._url = config.base_url + "/api/v1/text/contents"
        self._threshold = config.threshold

    async def classify(self, text: str) -> List[ClassificationResult]:
        payload: Dict[str, Any] = {
            "contents": [text],
            "detector_params": {"threshold": self._threshold},
        }

        resp = await self._post(self._url, json=payload)
        if resp.status_code != 200:
            raise ValueError(f"FMS detectors returned {resp.status_code}: {resp.text[:500]}")
        data = resp.json()

        try:
            (detections,) = data
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Unexpected FMS response structure (expected [[...]] for single content). Raw: {str(data)[:500]}"
            ) from exc

        if not isinstance(detections, list):
            raise ValueError(f"Unexpected FMS response structure (inner element is not a list). Raw: {str(data)[:500]}")
        if not detections:
            return []

        try:
            return [ClassificationResult(label=d["detection_type"], score=d["score"]) for d in detections]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Unexpected FMS detection entry structure: {exc}. Raw: {str(data)[:500]}") from exc


_BACKENDS = {
    "local": LocalBackend,
    "vllm": VLLMBackend,
    "kserve": KServeBackend,
    "fms": FMSBackend,
}

_backend_instances: Dict[str, ClassifierBackend] = {}


def get_backend(config: HFClassifierConfig, name: str = "") -> ClassifierBackend:
    """Get or create a cached backend instance from classifier config."""
    cache_key = json.dumps({"name": name, "config": config.model_dump(mode="json")}, sort_keys=True)
    cached = _backend_instances.get(cache_key)
    if cached is not None:
        return cached
    cls = _BACKENDS.get(config.engine)
    if cls is None:
        raise ValueError(f"Unknown hf_classifier engine: '{config.engine}'. Supported: {', '.join(_BACKENDS)}")
    _backend_instances[cache_key] = cls(config)
    return _backend_instances[cache_key]
