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

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from nemoguardrails import RailsConfig
from nemoguardrails.server import api
from nemoguardrails.server.api import GuardrailsApp, lifespan

MAIN_MODEL_CONFIG = """\
models:
  - type: main
    engine: openai
    model: gpt-4o-mini
"""

UNDEFINED_RAIL_MODEL_CONFIG = (
    MAIN_MODEL_CONFIG
    + """\
rails:
  input:
    flows:
      - content safety check input $model=content_safety

prompts:
  - task: content_safety_check_input $model=content_safety
    content: Check content safety
"""
)


def _write_config(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _server_app(config_path: Path) -> GuardrailsApp:
    server_app = GuardrailsApp(lifespan=lifespan)
    server_app.rails_config_path = str(config_path)
    return server_app


@pytest.mark.asyncio
@pytest.mark.parametrize("config_file", ["config.yml", "config/config.yml"])
async def test_single_config_is_loaded_at_startup(tmp_path, config_file):
    """Startup in single-config mode loads the config into app.single_config, in both supported layouts."""
    config_path = tmp_path / "my_config"
    _write_config(config_path / config_file, MAIN_MODEL_CONFIG)
    server_app = _server_app(config_path)

    async with lifespan(server_app):
        assert server_app.single_config_mode is True
        assert server_app.single_config_id == "my_config"
        assert isinstance(server_app.single_config, RailsConfig)
        assert server_app.single_config.models[0].model == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_invalid_single_config_fails_startup(tmp_path):
    """An invalid single config makes startup raise its validation error and leaves the single-config state unset."""
    _write_config(tmp_path / "config.yml", UNDEFINED_RAIL_MODEL_CONFIG)
    server_app = _server_app(tmp_path)

    with pytest.raises(
        ValueError,
        match="Input flow 'content safety check input' references model type 'content_safety' that is not defined",
    ):
        async with lifespan(server_app):
            pass

    assert server_app.single_config_mode is False
    assert server_app.single_config is None


@pytest.mark.asyncio
async def test_multi_config_directory_loads_no_config_at_startup(tmp_path):
    """Startup in multi-config mode leaves app.single_config unset and doesn't load the sub-configs, even an invalid one."""
    _write_config(tmp_path / "invalid" / "config.yml", UNDEFINED_RAIL_MODEL_CONFIG)
    server_app = _server_app(tmp_path)

    async with lifespan(server_app):
        assert server_app.single_config_mode is False
        assert server_app.single_config is None


def test_multi_config_server_starts_with_an_invalid_sub_config(tmp_path, monkeypatch):
    """A multi-config server with an invalid sub-config still starts and lists it, and a request for that sub-config gets 400."""
    _write_config(tmp_path / "valid" / "config.yml", MAIN_MODEL_CONFIG)
    _write_config(tmp_path / "invalid" / "config.yml", UNDEFINED_RAIL_MODEL_CONFIG)
    monkeypatch.setattr(api.app, "rails_config_path", str(tmp_path))
    monkeypatch.setattr(api, "llm_rails_instances", {})

    with TestClient(api.app, raise_server_exceptions=False) as client:
        configs = client.get("/v1/rails/configs")
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Hello"}],
                "guardrails": {"config_id": "invalid"},
            },
        )

    assert sorted(config["id"] for config in configs.json()) == ["invalid", "valid"]
    assert response.status_code == 400
    assert response.json()["error"]["message"].startswith("Could not load the requested guardrails configuration")


@pytest.mark.asyncio
async def test_shutdown_resets_single_config_state(tmp_path):
    """Shutdown resets single_config_mode, single_config_id and single_config to their defaults."""
    _write_config(tmp_path / "config.yml", MAIN_MODEL_CONFIG)
    server_app = _server_app(tmp_path)

    async with lifespan(server_app):
        assert server_app.single_config is not None

    assert server_app.single_config_mode is False
    assert server_app.single_config_id is None
    assert server_app.single_config is None
