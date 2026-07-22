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


@pytest.fixture(autouse=True)
def _langchain_tests_use_langchain(langchain_framework):
    pass


@pytest.fixture(autouse=True)
def _restore_langchain_provider_registries():
    from nemoguardrails.integrations.langchain.providers.providers import (
        _chat_providers,
        _llm_providers,
    )

    chat_providers = _chat_providers.copy()
    llm_providers = _llm_providers.copy()
    yield
    _chat_providers.clear()
    _chat_providers.update(chat_providers)
    _llm_providers.clear()
    _llm_providers.update(llm_providers)
