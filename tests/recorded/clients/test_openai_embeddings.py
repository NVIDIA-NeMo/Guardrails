# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nemoguardrails.embeddings.providers.openai import OpenAIEmbeddingModel
from tests.recorded.utils import DUMMY_OPENAI_API_KEY, api_key_for_record_mode


@pytest.mark.recorded
@pytest.mark.vcr
@pytest.mark.default_cassette("openai_embeddings_sync.yaml")
def test_openai_embeddings_sync(record_mode):
    api_key = api_key_for_record_mode("OPENAI_API_KEY", DUMMY_OPENAI_API_KEY, record_mode)
    model = OpenAIEmbeddingModel("text-embedding-3-small", api_key=api_key, max_retries=0)

    result = model.encode(["test"])

    assert len(result) == 1
    assert len(result[0]) == 1536
