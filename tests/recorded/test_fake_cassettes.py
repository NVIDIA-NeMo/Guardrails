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

from pathlib import Path

import pytest

from tests.recorded.fake_cassettes import fake_cassette_header, validate_fake_cassette_metadata

pytestmark = [pytest.mark.recorded]

RECORDED_DIR = Path(__file__).parent


def test_fake_cassette_header_metadata_validation(tmp_path):
    cassette = tmp_path / "fake.yaml"
    cassette.write_text(
        """
# fake_cassette:
#   reason: stream error path cannot be refreshed deterministically
#   frozen_fields:
#     - response.body.parsed_body
#   fake_llm_model_considered: true
version: 1
interactions: []
""",
        encoding="utf-8",
    )

    assert fake_cassette_header(cassette)["fake_cassette"]["reason"]
    validate_fake_cassette_metadata(cassette)


def test_committed_fake_cassettes_have_metadata():
    for cassette in RECORDED_DIR.rglob("cassettes/**/fake/**/*.yaml"):
        validate_fake_cassette_metadata(cassette)
