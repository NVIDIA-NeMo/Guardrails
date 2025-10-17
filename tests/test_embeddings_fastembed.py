# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest

from nemoguardrails.embeddings.providers.fastembed import FastEmbedEmbeddingModel


def test_sync_embeddings():
    model = FastEmbedEmbeddingModel("all-MiniLM-L6-v2")

    result = model.encode(["test"])

    assert len(result[0]) == 384


def test_additional_params_with_fastembed():
    model = FastEmbedEmbeddingModel("all-MiniLM-L6-v2", max_length=512, lazy_load=True)
    result = model.encode(["test"])

    assert len(result[0]) == 384


@pytest.mark.asyncio
async def test_async_embeddings():
    model = FastEmbedEmbeddingModel("all-MiniLM-L6-v2")

    result = await model.encode_async(["test"])

    assert len(result[0]) == 384
