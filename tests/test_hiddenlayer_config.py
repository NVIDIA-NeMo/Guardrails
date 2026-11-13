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
from pydantic import ValidationError

from nemoguardrails import RailsConfig
from nemoguardrails.library.hiddenlayer.actions import _main_model, _rail_config, _settings
from nemoguardrails.library.hiddenlayer.rail_config import (
    DEFAULT_API_URL,
    DEFAULT_REQUESTER_ID,
    HiddenLayerRailConfig,
    build_config_spec,
)

_ENV_VARS = (
    "HIDDENLAYER_CLIENT_ID",
    "HIDDENLAYER_CLIENT_SECRET",
    "HIDDENLAYER_BASE_URL",
    "HIDDENLAYER_PROJECT_ID",
    "HIDDENLAYER_REQUESTER_ID",
    "HIDDENLAYER_SEND_SESSION_ID",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_defaults():
    settings = _settings(HiddenLayerRailConfig())

    assert settings.api_url == DEFAULT_API_URL
    assert settings.project_id is None
    assert settings.requester_id == DEFAULT_REQUESTER_ID == "hiddenlayer-nemo"
    assert settings.fail_open is False
    assert (settings.max_retries, settings.max_retry_after_seconds, settings.retry_backoff_seconds) == (2, 30.0, 1.0)
    assert settings.send_session_id is True


@pytest.mark.parametrize(
    ("field", "env_name", "attribute"),
    [
        ("api_url", "HIDDENLAYER_BASE_URL", "api_url"),
        ("project_id", "HIDDENLAYER_PROJECT_ID", "project_id"),
        ("requester_id", "HIDDENLAYER_REQUESTER_ID", "requester_id"),
    ],
)
class TestPrecedence:
    @staticmethod
    def _value(field: str, source: str) -> str:
        return f"https://{source}.example" if field == "api_url" else f"{source}-value"

    def test_field_wins_over_env(self, monkeypatch, field, env_name, attribute):
        monkeypatch.setenv(env_name, self._value(field, "env"))
        settings = _settings(HiddenLayerRailConfig(**{field: self._value(field, "field")}))
        assert getattr(settings, attribute) == self._value(field, "field")

    def test_env_used_when_field_unset(self, monkeypatch, field, env_name, attribute):
        monkeypatch.setenv(env_name, self._value(field, "env"))
        assert getattr(_settings(HiddenLayerRailConfig()), attribute) == self._value(field, "env")

    def test_empty_field_counts_as_unset(self, monkeypatch, field, env_name, attribute):
        monkeypatch.setenv(env_name, self._value(field, "env"))
        settings = _settings(HiddenLayerRailConfig(**{field: ""}))
        assert getattr(settings, attribute) == self._value(field, "env")

    def test_empty_env_counts_as_unset(self, monkeypatch, field, env_name, attribute):
        monkeypatch.setenv(env_name, "")
        default = {"api_url": DEFAULT_API_URL, "project_id": None, "requester_id": DEFAULT_REQUESTER_ID}[field]
        assert getattr(_settings(HiddenLayerRailConfig()), attribute) == default


def test_api_url_trailing_slash_is_stripped(monkeypatch):
    monkeypatch.setenv("HIDDENLAYER_BASE_URL", "https://api.eu.hiddenlayer.ai/")
    assert _settings(HiddenLayerRailConfig()).api_url == "https://api.eu.hiddenlayer.ai"


def test_unknown_key_raises():
    with pytest.raises(ValidationError, match="projectid"):
        HiddenLayerRailConfig.model_validate({"projectid": "my-project"})


@pytest.mark.parametrize("knob", ["max_retries", "max_retry_after_seconds", "retry_backoff_seconds"])
def test_negative_retry_knobs_raise(knob):
    with pytest.raises(ValidationError):
        HiddenLayerRailConfig.model_validate({knob: -1})


def test_fail_open_description_warns_about_security_posture():
    description = HiddenLayerRailConfig.model_fields["fail_open"].description
    assert description is not None and "security posture" in description


class TestApiUrlScheme:
    @pytest.mark.parametrize(
        "url",
        [
            "http://evil.example",
            "http://api.us.hiddenlayer.ai",
            "http://localhost@evil.example",
            "http://127.0.0.1:9@evil.example",
            "http://localhost.evil.example",
            "ftp://api.us.hiddenlayer.ai",
            "api.us.hiddenlayer.ai",
            "https://",
        ],
    )
    def test_field_rejects_non_https_at_load(self, url):
        with pytest.raises(ValidationError, match="https://"):
            HiddenLayerRailConfig(api_url=url)

    def test_env_rejected_from_settings(self, monkeypatch):
        monkeypatch.setenv("HIDDENLAYER_BASE_URL", "http://evil.example")
        with pytest.raises(ValueError, match="https://") as exc_info:
            _settings(HiddenLayerRailConfig(fail_open=True))
        assert "evil.example" not in str(exc_info.value)

    @pytest.mark.parametrize(
        "url", ["http://127.0.0.1:9", "http://localhost:8080/v1", "http://[::1]:9", "HTTP://LOCALHOST"]
    )
    def test_loopback_http_accepted_from_field_and_env(self, monkeypatch, url):
        assert _settings(HiddenLayerRailConfig(api_url=url)).api_url == url
        monkeypatch.setenv("HIDDENLAYER_BASE_URL", url)
        assert _settings(HiddenLayerRailConfig()).api_url == url

    def test_https_any_host_accepted(self):
        assert _settings(HiddenLayerRailConfig(api_url="https://api.eu.hiddenlayer.ai")).api_url == (
            "https://api.eu.hiddenlayer.ai"
        )


def test_config_spec_exports_the_model():
    spec = build_config_spec()
    assert spec.exports == {"HiddenLayerRailConfig": HiddenLayerRailConfig}


def test_rail_config_defaults_without_a_config():
    assert _rail_config(None) == HiddenLayerRailConfig()


class TestMainModel:
    def test_main_model_and_engine(self):
        config = RailsConfig.from_content(
            yaml_content="""
                models:
                  - type: embeddings
                    engine: nim
                    model: nvidia/nv-embedqa-e5-v5
                  - type: main
                    engine: openai
                    model: gpt-5-nano
            """
        )
        assert _main_model(config) == ("gpt-5-nano", "openai")

    def test_no_placeholder_without_a_main_model(self):
        assert _main_model(RailsConfig.from_content(yaml_content="models: []")) == (None, None)

    def test_no_placeholder_without_a_config(self):
        assert _main_model(None) == (None, None)


class TestHeaderIdentityValidation:
    """requester_id and project_id travel as headers; one that cannot be sent fails every check."""

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("requester_id", "x" * 257),
            ("requester_id", "trailing-newline\n"),
            ("requester_id", " leading-space"),
            ("project_id", "p" * 257),
            ("project_id", "proyecto-se\u00f1al"),
            ("project_id", "has\rcarriage-return"),
        ],
    )
    def test_an_unsendable_identity_is_rejected_at_load(self, field, value):
        with pytest.raises(ValidationError):
            HiddenLayerRailConfig(**{field: value})

    @pytest.mark.parametrize(
        ("field", "value"),
        [("requester_id", "hiddenlayer-nemo"), ("project_id", "01a1b486-86a8-7c33-87f8-63f3ad713327")],
    )
    def test_a_sendable_identity_is_accepted(self, field, value):
        assert getattr(HiddenLayerRailConfig(**{field: value}), field) == value
