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

import asyncio

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.library.sensitive_data_detection import actions


@pytest.fixture(scope="module", autouse=True)
def require_presidio():
    pytest.importorskip("presidio_analyzer")
    pytest.importorskip("presidio_anonymizer")
    spacy = pytest.importorskip("spacy")
    if not spacy.util.is_package("en_core_web_lg"):
        pytest.skip("The en_core_web_lg model must already be installed")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text, expected",
    [
        ("My name is John Smith.", RailOutcome.block(metadata={"has_sensitive_data": True})),
        ("Discuss data security.", RailOutcome.allow(metadata={"has_sensitive_data": False})),
    ],
)
async def test_detection_yields_to_event_loop(text, expected):
    config = RailsConfig.from_content(
        yaml_content="""
rails:
  config:
    sensitive_data_detection:
      input:
        entities: [PERSON]
"""
    )
    # Repeat so yielding cannot depend only on first-call model initialization.
    for _ in range(2):
        task = asyncio.create_task(actions.detect_sensitive_data("input", text, config))
        try:
            await asyncio.sleep(0)
            assert not task.done(), "Detection completed without yielding to the event loop"
            assert await task == expected
        finally:
            await asyncio.gather(task, return_exceptions=True)
