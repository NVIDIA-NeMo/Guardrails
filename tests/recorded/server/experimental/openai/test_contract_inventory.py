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

"""Audit every stored proxy exchange, including compatibility recordings."""

import pytest
import yaml

from tests.recorded.cassette import cassette_with_rehydrated_bodies
from tests.recorded.server.experimental.openai.contract_audit import CASSETTES, audit_exchange

pytestmark = [pytest.mark.recorded, pytest.mark.asyncio]


@pytest.mark.parametrize("path", sorted(CASSETTES.rglob("*.yaml")), ids=lambda path: str(path.relative_to(CASSETTES)))
async def test_every_proxy_recording_matches_the_reviewed_contract(path):
    cassette = cassette_with_rehydrated_bodies(yaml.safe_load(path.read_text(encoding="utf-8")))
    assert cassette["interactions"], "A provider recording must retain its exchanges"
    for exchange in cassette["interactions"]:
        request = exchange["request"]
        await audit_exchange(path, request["uri"], request["body"], exchange["response"])
