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

import os

import pytest

DUMMY_OPENAI_API_KEY = "sk-recorded-replay"
DUMMY_NVIDIA_API_KEY = "nvapi-recorded-replay"


def api_key_for_record_mode(env_name: str, dummy_value: str, record_mode: str) -> str:
    """Return dummy replay credentials, but require real credentials to record."""
    if record_mode == "none":
        return dummy_value

    value = os.environ.get(env_name)
    if not value:
        pytest.fail(f"{env_name} is required to refresh cassette", pytrace=False)
    return value


def set_api_key_for_record_mode(
    monkeypatch: pytest.MonkeyPatch, env_name: str, dummy_value: str, record_mode: str
) -> str:
    value = api_key_for_record_mode(env_name, dummy_value, record_mode)
    monkeypatch.setenv(env_name, value)
    return value
