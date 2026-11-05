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

"""Audit captured traffic even when pytest selects only a compatibility test."""

from pathlib import Path

import pytest
import pytest_asyncio

from tests.recorded.server.experimental.openai.contract_audit import audit_exchange


@pytest_asyncio.fixture(autouse=True)
async def strict_proxy_contract(request):
    if request.node.get_closest_marker("vcr") is None:
        yield
        return
    cassette = request.getfixturevalue("vcr")
    if cassette is None:
        pytest.fail("Strict proxy contract auditing requires VCR recording or replay.", pytrace=False)
    path = Path(request.getfixturevalue("recorded_cassette_path"))
    yield
    assert len(cassette.requests) == len(cassette.responses)
    for outbound, response in zip(cassette.requests, cassette.responses):
        await audit_exchange(path, outbound.uri, outbound.body, response)
