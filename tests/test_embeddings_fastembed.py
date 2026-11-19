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

import os

import pytest
from onnxruntime.capi.onnxruntime_pybind11_state import NoSuchFile

from nemoguardrails.embeddings.providers.fastembed import FastEmbedEmbeddingModel

LIVE_TEST_MODE = os.environ.get("LIVE_TEST")


class _FakeEmbeddingVector:
    def tolist(self):
        return [0.1, 0.2]


def test_recovers_from_missing_onnxruntime_model_cache(monkeypatch):
    calls = []

    class FakeTextEmbedding:
        def __init__(self, model_name, **kwargs):
            calls.append((model_name, kwargs))
            if len(calls) == 1:
                raise NoSuchFile(
                    "[ONNXRuntimeError] : 3 : NO_SUCHFILE : Load model from /tmp/model.onnx failed. File doesn't exist"
                )

        def embed(self, documents):
            return [_FakeEmbeddingVector()]

    monkeypatch.setattr("fastembed.TextEmbedding", FakeTextEmbedding)

    model = FastEmbedEmbeddingModel("all-MiniLM-L6-v2")

    assert model.embedding_size == 2
    assert calls == [
        ("sentence-transformers/all-MiniLM-L6-v2", {}),
        ("sentence-transformers/all-MiniLM-L6-v2", {"cache_dir": ".cache"}),
    ]


def test_reraises_unrelated_fastembed_errors(monkeypatch):
    class FakeTextEmbedding:
        def __init__(self, model_name, **kwargs):
            raise ValueError("unrelated failure")

    monkeypatch.setattr("fastembed.TextEmbedding", FakeTextEmbedding)

    with pytest.raises(ValueError, match="unrelated failure"):
        FastEmbedEmbeddingModel("all-MiniLM-L6-v2")


@pytest.mark.skipif(not LIVE_TEST_MODE, reason="Not in live mode.")
def test_sync_embeddings():
    model = FastEmbedEmbeddingModel("all-MiniLM-L6-v2")

    result = model.encode(["test"])

    assert len(result[0]) == 384


@pytest.mark.skipif(not LIVE_TEST_MODE, reason="Not in live mode.")
def test_additional_params_with_fastembed():
    model = FastEmbedEmbeddingModel("all-MiniLM-L6-v2", max_length=512, lazy_load=True)
    result = model.encode(["test"])

    assert len(result[0]) == 384


@pytest.mark.skipif(not LIVE_TEST_MODE, reason="Not in live mode.")
@pytest.mark.asyncio
async def test_async_embeddings():
    model = FastEmbedEmbeddingModel("all-MiniLM-L6-v2")

    result = await model.encode_async(["test"])

    assert len(result[0]) == 384
