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
from typing import Any

import yaml

REQUIRED_FAKE_CASSETTE_FIELDS = {"reason", "frozen_fields", "fake_llm_model_considered"}


def fake_cassette_header(path: Path) -> dict[str, Any]:
    lines = []
    seen_header = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not seen_header and not line.strip():
            continue
        if not line.startswith("#"):
            break
        seen_header = True
        content = line.removeprefix("#")
        if content.startswith(" "):
            content = content[1:]
        lines.append(content)
    data = yaml.safe_load("\n".join(lines)) if lines else None
    return data if isinstance(data, dict) else {}


def validate_fake_cassette_metadata(path: Path) -> None:
    metadata = fake_cassette_header(path).get("fake_cassette")
    assert isinstance(metadata, dict), f"{path} is missing fake_cassette header metadata"
    missing = REQUIRED_FAKE_CASSETTE_FIELDS - set(metadata)
    assert not missing, f"{path} missing fake_cassette fields: {sorted(missing)}"
    assert isinstance(metadata["frozen_fields"], list) and metadata["frozen_fields"]
    assert isinstance(metadata["reason"], str) and metadata["reason"].strip()
    assert isinstance(metadata["fake_llm_model_considered"], bool)
