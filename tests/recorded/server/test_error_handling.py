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

from collections.abc import Callable, Iterator

import pytest
from fastapi.testclient import TestClient
from openai import NotFoundError, OpenAI

from nemoguardrails.llm.frameworks import (
    _areset_frameworks,
    get_default_framework,
    set_default_framework,
)
from nemoguardrails.server import api
from tests.recorded.rails.library.configs import OPENAI_MULTI_SELF_CHECK_INVALID_MODEL_CONFIG
from tests.recorded.rails.public_api.configs import (
    OPENAI_INVALID_MODEL,
    OPENAI_INVALID_MODEL_CONFIG,
    OPENAI_MODEL,
)
from tests.recorded.rails_config import RailsConfigSource, load_config

pytestmark = [pytest.mark.recorded, pytest.mark.vcr]

_CONFIGS = {
    source: load_config(source)
    for source in (OPENAI_INVALID_MODEL_CONFIG, OPENAI_MULTI_SELF_CHECK_INVALID_MODEL_CONFIG)
}


@pytest.fixture
def recorded_server(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[TestClient, Callable[[RailsConfigSource], None]]]:
    selected_config = None

    def serve(source: RailsConfigSource) -> None:
        nonlocal selected_config
        selected_config = _CONFIGS[source].model_copy(deep=True)

    def load_selected_config(path: str):
        assert selected_config is not None
        return selected_config.model_copy(deep=True)

    original_single_config_mode = api.app.single_config_mode
    original_framework = get_default_framework()
    api.app.single_config_mode = False
    set_default_framework("default")
    api.llm_rails_instances.clear()
    api.llm_rails_events_history_cache.clear()
    monkeypatch.setattr(api.RailsConfig, "from_path", staticmethod(load_selected_config))

    try:
        with TestClient(api.app, raise_server_exceptions=False) as client:
            try:
                yield client, serve
            finally:
                assert client.portal is not None
                client.portal.call(_areset_frameworks)
    finally:
        api.llm_rails_instances.clear()
        api.llm_rails_events_history_cache.clear()
        api.app.single_config_mode = original_single_config_mode
        set_default_framework(original_framework)


def _chat_request(*, stream: bool) -> dict:
    return {
        "model": OPENAI_INVALID_MODEL,
        "messages": [{"role": "user", "content": "Say a short safe greeting."}],
        "stream": stream,
        "guardrails": {"config_id": "recorded"},
    }


def test_chat_completion_real_provider_error_maps_to_openai_exception(openai_api_key, recorded_server):
    server, serve = recorded_server
    serve(OPENAI_INVALID_MODEL_CONFIG)
    client = OpenAI(
        api_key="recorded-client",
        base_url="http://testserver/v1",
        http_client=server,
        max_retries=0,
    )

    with pytest.raises(NotFoundError) as exc_info:
        client.chat.completions.create(
            model=OPENAI_INVALID_MODEL,
            messages=[{"role": "user", "content": "Say a short safe greeting."}],
            extra_body={"guardrails": {"config_id": "recorded"}},
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.type == "not_found_error"
    assert exc_info.value.code == "model_not_found"
    assert exc_info.value.param is None
    assert OPENAI_INVALID_MODEL in exc_info.value.message


def test_streaming_chat_completion_initial_provider_error_preserves_http_status(openai_api_key, recorded_server):
    server, serve = recorded_server
    serve(OPENAI_INVALID_MODEL_CONFIG)

    response = server.post("/v1/chat/completions", json=_chat_request(stream=True))

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "message": f"The model `{OPENAI_INVALID_MODEL}` does not exist or you do not have access to it.",
            "type": "not_found_error",
            "param": None,
            "code": 404,
        }
    }


def test_checks_real_provider_error_preserves_status_and_code(openai_api_key, recorded_server):
    server, serve = recorded_server
    serve(OPENAI_MULTI_SELF_CHECK_INVALID_MODEL_CONFIG)

    response = server.post(
        "/v1/checks",
        json={
            "model": OPENAI_MODEL,
            "messages": [{"role": "user", "content": "hello"}],
            "guardrails": {"config_id": "recorded"},
        },
    )

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "message": "The model `nonexistent-self-check-model` does not exist or you do not have access to it.",
            "type": "not_found_error",
            "param": None,
            "code": "model_not_found",
        }
    }
