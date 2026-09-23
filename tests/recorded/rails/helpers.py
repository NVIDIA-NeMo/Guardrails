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

from __future__ import annotations

from typing import Any, AsyncIterator

from nemoguardrails import LLMRails
from tests.recorded.rails_config import RailsConfigSource, enable_streaming, load_config


async def async_chunks(values: list[str]) -> AsyncIterator[str]:
    for value in values:
        yield value


def build_rails(
    source: RailsConfigSource,
    *,
    llm: Any = None,
    streaming: bool = False,
    verbose: bool = False,
) -> LLMRails:
    config = load_config(source)
    if streaming:
        config = enable_streaming(config)
    return LLMRails(config, llm=llm, verbose=verbose)
