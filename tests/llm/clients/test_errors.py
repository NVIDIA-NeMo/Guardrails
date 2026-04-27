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

from nemoguardrails.llm.clients._errors import _redact_secrets


class TestRedactSecrets:
    @pytest.mark.parametrize(
        "raw,redacted_marker",
        [
            ("Auth failed for sk-proj-AbCdEfG12345", "sk-***"),
            ("Auth failed for sk-ant-api03-AbCdEfG", "sk-***"),
            ("Token: nvapi-XYZ_abc123", "nvapi-***"),
            ("Authorization: Bearer eyJhbGciOiJIUzI1Ni", "Bearer ***"),
            ("Google key AIzaSyD-x9K8z7QWERTYuiopasdfgh-1234567890 leaked", "AIza***"),
        ],
    )
    def test_known_prefixes_redacted(self, raw, redacted_marker):
        out = _redact_secrets(raw)
        assert redacted_marker in out

    def test_aiza_lowercase_also_caught(self):
        out = _redact_secrets("aizasydddd-key")
        assert "***" in out
        assert "aizasydddd-key" not in out

    def test_no_secrets_unchanged(self):
        msg = "Model returned an empty response"
        assert _redact_secrets(msg) == msg

    def test_partial_key_not_leaked(self):
        out = _redact_secrets("Auth failed for sk-proj-AbCdEfG12345")
        assert "AbCdEfG12345" not in out
