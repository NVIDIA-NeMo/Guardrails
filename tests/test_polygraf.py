# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Any, Dict, List, Optional

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.actions.actions import ActionResult, action
from nemoguardrails.library.polygraf.actions import (
    FAILSAFE_MASK_PLACEHOLDER,
    polygraf_detect_pii,
    polygraf_mask_pii,
)
from nemoguardrails.library.polygraf.request import polygraf_request
from tests.utils import TestChat


def create_polygraf_mock_response(
    text: str,
    entities_to_detect: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Create a mock Polygraf response based on the input text and entities to detect."""
    detected_entities = []

    entity_patterns = {
        "Person": ["John"],
        "Email": ["test@gmail.com"],
    }

    for entity_type, patterns in entity_patterns.items():
        if entities_to_detect and entity_type not in entities_to_detect:
            continue

        for pattern in patterns:
            start = 0
            while True:
                pos = text.find(pattern, start)
                if pos == -1:
                    break
                detected_entities.append(
                    {
                        "entity_type": entity_type,
                        "entity_text": pattern,
                        "start": pos,
                        "end": pos + len(pattern),
                        "score": 0.99,
                    }
                )
                start = pos + 1

    return detected_entities


def create_mock_polygraf_detect_pii(entities_to_detect: Optional[List[str]] = None):
    """Create a mock polygraf_detect_pii action that returns True when PII is detected."""

    async def mock_polygraf_detect_pii(source: str, text: str, config, **kwargs):
        entities = create_polygraf_mock_response(text, entities_to_detect)
        return len(entities) > 0

    return mock_polygraf_detect_pii


def create_mock_polygraf_mask_pii(entities_to_detect: Optional[List[str]] = None):
    """Create a mock polygraf_mask_pii action that masks PII in text."""

    async def mock_polygraf_mask_pii(source: str, text: str, config, **kwargs):
        entities = create_polygraf_mock_response(text, entities_to_detect)
        if not entities:
            return text

        masked_text = text
        for entity in sorted(entities, key=lambda x: x["start"], reverse=True):
            start = entity["start"]
            end = entity["end"]
            entity_type = entity["entity_type"]
            masked_text = masked_text[:start] + f"<{entity_type}>" + masked_text[end:]

        return masked_text

    return mock_polygraf_mask_pii


@action()
def retrieve_relevant_chunks():
    context_updates = {"relevant_chunks": "Mock retrieved context."}

    return ActionResult(
        return_value=context_updates["relevant_chunks"],
        context_updates=context_updates,
    )


def _polygraf_config():
    return RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                polygraf:
                  server_endpoint: http://localhost:8000/v1/pii/text-detect
                  input:
                    entities:
                      - Email
                      - Person
                  output:
                    entities:
                      - Email
                      - Person
                  retrieval:
                    entities:
                      - Email
                      - Person
        """,
    )


class _FakeResponse:
    def __init__(self, status=200, payload=None, text=""):
        self.status = status
        self._payload = payload if payload is not None else []
        self._text = text

    async def json(self):
        return self._payload

    async def text(self):
        return self._text


class _FakePostContextManager:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeSession:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def post(self, server_endpoint, json, headers):
        self.requests.append(
            {
                "server_endpoint": server_endpoint,
                "json": json,
                "headers": headers,
            }
        )
        return _FakePostContextManager(self.response)


@pytest.mark.asyncio
async def test_polygraf_request_uses_shared_session_and_bearer_auth():
    session = _FakeSession(
        _FakeResponse(
            payload=[
                {
                    "entity_type": "Person",
                    "entity_text": "John",
                    "start": 0,
                    "end": 4,
                    "score": 0.99,
                }
            ]
        )
    )

    entities = await polygraf_request("John", "http://polygraf.example/pii", "secret", session=session)

    assert entities[0]["entity_type"] == "Person"
    assert session.requests[0]["headers"]["Authorization"] == "Bearer secret"
    assert session.requests[0]["json"]["detect_pid"] is True
    assert session.requests[0]["json"]["aggregate_entities"] is True


@pytest.mark.asyncio
async def test_polygraf_request_accepts_wrapped_entities_response():
    session = _FakeSession(_FakeResponse(payload={"entities": [{"entity_type": "Email"}]}))

    entities = await polygraf_request("test@gmail.com", "http://polygraf.example/pii", None, session=session)

    assert entities == [{"entity_type": "Email"}]


@pytest.mark.asyncio
async def test_polygraf_request_accepts_null_entities_as_empty_response():
    session = _FakeSession(_FakeResponse(payload={"entities": None}))

    entities = await polygraf_request("hello", "http://polygraf.example/pii", None, session=session)

    assert entities == []


@pytest.mark.asyncio
async def test_polygraf_request_raises_for_invalid_response_shape():
    session = _FakeSession(_FakeResponse(payload={"unexpected": []}))

    with pytest.raises(ValueError, match="Invalid response from Polygraf service"):
        await polygraf_request("John", "http://polygraf.example/pii", None, session=session)


@pytest.mark.asyncio
async def test_polygraf_request_raises_for_non_200_response():
    session = _FakeSession(_FakeResponse(status=401, text="missing token"))

    with pytest.raises(ValueError, match="Polygraf call failed with status code 401"):
        await polygraf_request("John", "http://polygraf.example/pii", None, session=session)


class _FakeSessionWithTimeoutKwarg:
    def __init__(self, response):
        self.response = response
        self.timeouts = []

    def post(self, server_endpoint, json, headers, timeout=None):
        self.timeouts.append(timeout)
        return _FakePostContextManager(self.response)


@pytest.mark.asyncio
async def test_polygraf_request_forwards_timeout_to_post():
    import aiohttp

    session = _FakeSessionWithTimeoutKwarg(_FakeResponse(payload=[]))

    await polygraf_request("hello", "http://polygraf.example/pii", None, session=session, timeout=7)

    assert isinstance(session.timeouts[0], aiohttp.ClientTimeout)
    assert session.timeouts[0].total == 7


class _FakeRaisingSession:
    """Test double whose .post() raises a configurable exception when entered."""

    def __init__(self, exc: BaseException):
        self.exc = exc

    def post(self, *args, **kwargs):
        async def _raise():
            raise self.exc

        class _Ctx:
            def __init__(self, raise_fn):
                self._raise_fn = raise_fn

            async def __aenter__(self_inner):
                await self_inner._raise_fn()

            async def __aexit__(self_inner, *a):
                return False

        return _Ctx(_raise)


@pytest.mark.asyncio
async def test_polygraf_request_normalizes_timeout_as_value_error():
    import asyncio

    session = _FakeRaisingSession(asyncio.TimeoutError())

    with pytest.raises(ValueError, match="timed out"):
        await polygraf_request("hello", "http://polygraf.example/pii", None, session=session, timeout=3)


@pytest.mark.asyncio
async def test_polygraf_request_normalizes_client_error_as_value_error():
    import aiohttp

    session = _FakeRaisingSession(aiohttp.ClientConnectionError("dns failure"))

    with pytest.raises(ValueError, match="Polygraf call failed"):
        await polygraf_request("hello", "http://polygraf.example/pii", None, session=session)


def test_polygraf_config_rejects_unknown_keys():
    """Unknown Polygraf config keys must be rejected (extra='forbid')."""

    with pytest.raises(Exception) as excinfo:
        RailsConfig.from_content(
            yaml_content="""
                models: []
                rails:
                  config:
                    polygraf:
                      server_endpoint: http://localhost:8000/v1/pii/text-detect
                      unknown_field: 42
            """,
        )
    assert "unknown_field" in str(excinfo.value) or "extra" in str(excinfo.value).lower()


# ---------------------------------------------------------------------------
# Colang 2.0 flow coverage
# ---------------------------------------------------------------------------


def _load_polygraf_v2_flows():
    """Parse the shipped Polygraf Colang 2.x flow file and return flow dicts."""
    import importlib.resources as resources

    from nemoguardrails.colang import parse_colang_file

    flows_path = resources.files("nemoguardrails.library.polygraf").joinpath("flows.co")
    content = flows_path.read_text(encoding="utf-8")
    parsed = parse_colang_file(filename="flows.co", content=content, version="2.x", include_source_mapping=False)
    return [flow.to_dict() for flow in parsed["flows"]]


def _flow_global_vars(flow_dict):
    """Return the set of variable names declared `global` in a parsed Colang 2 flow."""
    globals_found = set()
    for el in flow_dict.get("elements", []):
        if el.get("_type") == "global":
            name = el.get("var_name")
            if name:
                globals_found.add(name)
        # Some parser variants attach `global` as a spec_op; collect those too.
        if el.get("_type") == "spec_op" and el.get("op") == "global":
            spec = el.get("spec") or {}
            name = spec.get("var_name") or spec.get("name")
            if name:
                globals_found.add(name)
    return globals_found


def test_polygraf_v2_flows_parse_successfully():
    """flows.co must be valid Colang 2.x and define all six expected flows."""

    flows = _load_polygraf_v2_flows()
    flow_names = sorted(f.get("name") or "" for f in flows)
    assert flow_names == [
        "polygraf detect pii on input",
        "polygraf detect pii on output",
        "polygraf detect pii on retrieval",
        "polygraf mask pii on input",
        "polygraf mask pii on output",
        "polygraf mask pii on retrieval",
    ]


@pytest.mark.unit
def test_polygraf_v2_input_flow_passes_actual_user_message_to_action():
    """End-to-end Colang 2 regression test.

    Reproduces the bug Pouyanpi flagged: a Colang 2 flow that reads a rails
    variable (``$user_message``) without a ``global`` declaration ends up
    sending ``text=null`` to the action. By registering a mock that records
    the ``text`` the masking action received and running it through the
    actual shipped ``polygraf mask pii on input`` flow body, we lock in the
    fix end-to-end.

    If the ``global $user_message`` line is removed from ``flows.co``, the
    Colang 2 runtime sends ``text=None`` and this test fails.
    """
    captured = {}

    async def fake_polygraf_mask_pii(source: str, text, **kwargs):
        captured["source"] = source
        captured["text"] = text
        return f"<masked:{text}>"

    # We use the SHIPPED polygraf flow body verbatim (read from flows.co)
    # and wire it into a v2 input rail using the standard guardrails pattern
    # used in tests/v2_x/test_input_output_rails_transformations.py. We do
    # not go through `rails.input.flows` here because that codepath emits a
    # deprecation warning and pulls in the whole library namespace, making
    # the test less direct.
    import importlib.resources as resources

    flows_co = resources.files("nemoguardrails.library.polygraf").joinpath("flows.co").read_text(encoding="utf-8")

    # Sanity check: this test relies on the shipped flow text being present.
    assert "flow polygraf mask pii on input" in flows_co
    assert "global $user_message" in flows_co

    colang_content = (
        """
import core
import guardrails

"""
        + flows_co
        + """

flow input rails $input_text
    polygraf mask pii on input

flow main
    await user said "John"
    bot say "done"
"""
    )

    config = RailsConfig.from_content(
        colang_content=colang_content,
        yaml_content="""
            colang_version: "2.x"
            models: []
        """,
    )

    chat = TestChat(config, llm_completions=[])
    chat.app.register_action(fake_polygraf_mask_pii, "polygraf_mask_pii")

    chat >> "John"
    chat << "done"

    # The action must have been called with the actual user text, not None.
    # If the `global $user_message` declaration is missing from flows.co, the
    # Colang 2 runtime would have sent text=None and this assertion would fail.
    assert captured.get("text") == "John", (
        f"Polygraf v2 input rail invoked action with text={captured.get('text')!r} "
        "instead of the actual user message. The most likely cause is a missing "
        "`global $user_message` declaration in flows.co."
    )
    assert captured.get("source") == "input"


def test_polygraf_v2_flows_declare_required_globals():
    """Each Polygraf v2 flow must declare the rails variable it reads as `global`.

    Without this, the Colang 2 runtime sends ``text: null`` to the action,
    letting PII through (regression guarded by this test).
    """

    flows = _load_polygraf_v2_flows()
    expected = {
        "polygraf detect pii on input": "$user_message",
        "polygraf detect pii on output": "$bot_message",
        "polygraf detect pii on retrieval": "$relevant_chunks",
        "polygraf mask pii on input": "$user_message",
        "polygraf mask pii on output": "$bot_message",
        "polygraf mask pii on retrieval": "$relevant_chunks",
    }
    by_name = {f["name"]: f for f in flows}

    for flow_name, required_var in expected.items():
        flow = by_name.get(flow_name)
        assert flow is not None, f"Flow {flow_name!r} missing from flows.co"

        # Serialize the flow YAML and check the global declaration appears
        # before any action invocation that reads the variable. We use a
        # textual search because the parser emits global declarations in a
        # few different shapes depending on the Colang 2 lexer state.
        import yaml as _yaml

        from nemoguardrails.utils import CustomDumper

        flow_yaml = _yaml.dump(flow, sort_keys=False, Dumper=CustomDumper, width=1000)
        assert required_var in flow_yaml, f"Flow {flow_name!r} does not reference {required_var}"

        # The text "global" should appear in the flow body. This is a
        # smoke check that the declaration is present in some form.
        assert "global" in flow_yaml.lower(), (
            f"Flow {flow_name!r} is missing a `global` declaration for {required_var}; "
            "Colang 2 would otherwise send text=null to the Polygraf action."
        )


@pytest.mark.asyncio
async def test_polygraf_mask_pii_fails_closed_on_malformed_selected_entity(monkeypatch, caplog):
    """A configured (selected) entity with bad offsets must fail closed, not silently skip."""

    sensitive_email = "test@gmail.com"
    sensitive_input = f"John lives here. Email: {sensitive_email}"

    async def mock_request(text, server_endpoint, api_key, session=None):
        return [
            {"entity_type": "Person", "start": 0, "end": 4},
            # Email is in the configured entities and has malformed offsets ->
            # the action must fail closed instead of returning partially masked text.
            {"entity_type": "Email", "entity_text": sensitive_email},
        ]

    monkeypatch.setenv("POLYGRAF_API_KEY", "secret")
    monkeypatch.setattr("nemoguardrails.library.polygraf.actions.polygraf_request", mock_request)
    caplog.set_level("WARNING")

    result = await polygraf_mask_pii("input", sensitive_input, _polygraf_config())

    assert result == FAILSAFE_MASK_PLACEHOLDER
    # The original sensitive value must never appear in the returned text.
    assert sensitive_email not in result
    # Log warnings must only carry structural metadata, not the PII value.
    assert sensitive_email not in caplog.text
    assert "Skipping malformed Polygraf entity" in caplog.text
    assert "invalid_fields" in caplog.text


@pytest.mark.asyncio
async def test_polygraf_mask_pii_skips_unselected_malformed_entity(monkeypatch, caplog):
    """A *known-type* malformed entity that does NOT match the entity filter is skipped, not failed."""

    async def mock_request(text, server_endpoint, api_key, session=None):
        return [
            {"entity_type": "Person", "start": 0, "end": 4},
            # CreditCard is not in the configured entities for `input`; even though
            # it's malformed, it must not trigger a fail-closed.
            {"entity_type": "CreditCard"},
        ]

    monkeypatch.setenv("POLYGRAF_API_KEY", "secret")
    monkeypatch.setattr("nemoguardrails.library.polygraf.actions.polygraf_request", mock_request)
    caplog.set_level("WARNING")

    result = await polygraf_mask_pii("input", "John lives here", _polygraf_config())

    assert result == "<Person> lives here"


@pytest.mark.asyncio
async def test_polygraf_mask_pii_fails_closed_on_missing_entity_type(monkeypatch, caplog):
    """An entity with no entity_type cannot be safely classified -> fail closed even with a filter set."""

    sensitive = "John lives here"

    async def mock_request(text, server_endpoint, api_key, session=None):
        return [
            {"start": 0, "end": 4, "entity_text": "John"},
        ]

    monkeypatch.setenv("POLYGRAF_API_KEY", "secret")
    monkeypatch.setattr("nemoguardrails.library.polygraf.actions.polygraf_request", mock_request)
    caplog.set_level("WARNING")

    result = await polygraf_mask_pii("input", sensitive, _polygraf_config())

    assert result == FAILSAFE_MASK_PLACEHOLDER
    assert "John" not in caplog.text


@pytest.mark.parametrize(
    "bad_offsets",
    [
        {"start": True, "end": 4},  # bool start (subclass of int) must be rejected
        {"start": 0, "end": False},  # bool end
        {"start": -1, "end": 4},  # negative start
        {"start": 5, "end": 3},  # reversed
        {"start": 0, "end": 9999},  # end past text length
        {"start": 0, "end": 0},  # empty span
    ],
)
@pytest.mark.asyncio
async def test_polygraf_mask_pii_fails_closed_on_out_of_range_offsets(monkeypatch, bad_offsets):
    """Invalid offsets (bool, negative, reversed, beyond text, empty) must fail closed for a selected entity."""

    async def mock_request(text, server_endpoint, api_key, session=None):
        return [{"entity_type": "Person", **bad_offsets}]

    monkeypatch.setenv("POLYGRAF_API_KEY", "secret")
    monkeypatch.setattr("nemoguardrails.library.polygraf.actions.polygraf_request", mock_request)

    result = await polygraf_mask_pii("input", "John lives here", _polygraf_config())

    assert result == FAILSAFE_MASK_PLACEHOLDER


@pytest.mark.asyncio
async def test_polygraf_detect_pii_fails_closed_on_missing_entity_type(monkeypatch):
    """detect must block when an entity has no entity_type (cannot prove it's safe)."""

    async def mock_request(text, server_endpoint, api_key, session=None):
        return [{"start": 0, "end": 4, "entity_text": "John"}]

    monkeypatch.setenv("POLYGRAF_API_KEY", "secret")
    monkeypatch.setattr("nemoguardrails.library.polygraf.actions.polygraf_request", mock_request)

    result = await polygraf_detect_pii("input", "John lives here", _polygraf_config())

    assert result is True


@pytest.mark.asyncio
async def test_polygraf_mask_pii_fails_closed_on_provider_error(monkeypatch, caplog):
    """A timeout / network error from the request layer must redact the entire payload."""

    sensitive_text = "John lives at 1 Main St; email test@gmail.com"

    async def mock_request(text, server_endpoint, api_key, session=None):
        raise ValueError("Polygraf call timed out after 30 seconds.")

    monkeypatch.setenv("POLYGRAF_API_KEY", "secret")
    monkeypatch.setattr("nemoguardrails.library.polygraf.actions.polygraf_request", mock_request)
    caplog.set_level("WARNING")

    result = await polygraf_mask_pii("input", sensitive_text, _polygraf_config())

    assert result == FAILSAFE_MASK_PLACEHOLDER
    # Even on failure, the caller's text must not leak into logs.
    assert sensitive_text not in caplog.text
    assert "test@gmail.com" not in caplog.text
    assert "Polygraf masking failed" in caplog.text


@pytest.mark.asyncio
async def test_polygraf_detect_pii_fails_closed_on_provider_error(monkeypatch, caplog):
    """A request-layer ValueError must cause detect to block (return True)."""

    async def mock_request(text, server_endpoint, api_key, session=None):
        raise ValueError("Polygraf call failed: ClientConnectorError: ...")

    monkeypatch.setenv("POLYGRAF_API_KEY", "secret")
    monkeypatch.setattr("nemoguardrails.library.polygraf.actions.polygraf_request", mock_request)
    caplog.set_level("WARNING")

    result = await polygraf_detect_pii("input", "John lives here", _polygraf_config())

    assert result is True
    assert "Polygraf detection failed" in caplog.text


@pytest.mark.asyncio
async def test_polygraf_detect_pii_fails_closed_on_malformed_selected_entity(monkeypatch, caplog):
    """detect must block when a configured entity is reported with bad shape."""

    async def mock_request(text, server_endpoint, api_key, session=None):
        return [
            # Email is in the configured filter and missing offsets -> fail closed.
            {"entity_type": "Email"},
        ]

    monkeypatch.setenv("POLYGRAF_API_KEY", "secret")
    monkeypatch.setattr("nemoguardrails.library.polygraf.actions.polygraf_request", mock_request)
    caplog.set_level("WARNING")

    result = await polygraf_detect_pii("input", "Some text", _polygraf_config())

    assert result is True
    assert "Polygraf returned a malformed selected entity" in caplog.text


@pytest.mark.asyncio
async def test_polygraf_actions_warn_when_api_key_missing(monkeypatch, caplog):
    async def mock_request(text, server_endpoint, api_key, session=None):
        assert api_key is None
        return []

    monkeypatch.delenv("POLYGRAF_API_KEY", raising=False)
    monkeypatch.setattr("nemoguardrails.library.polygraf.actions.polygraf_request", mock_request)
    caplog.set_level("WARNING")

    result = await polygraf_detect_pii("input", "John", _polygraf_config())

    assert result is False
    assert "POLYGRAF_API_KEY environment variable is not set" in caplog.text


@pytest.mark.asyncio
async def test_polygraf_mask_pii_accepts_extra_kwargs_and_shared_session(monkeypatch):
    sentinel_session = object()

    async def mock_request(text, server_endpoint, api_key, session=None):
        assert api_key == "secret"
        assert session is sentinel_session
        return [{"entity_type": "Person", "entity_text": "John", "start": 0, "end": 4, "score": 0.99}]

    monkeypatch.setenv("POLYGRAF_API_KEY", "secret")
    monkeypatch.setattr("nemoguardrails.library.polygraf.actions.polygraf_request", mock_request)

    result = await polygraf_mask_pii("input", "John", _polygraf_config(), session=sentinel_session, extra="ignored")

    assert result == "<Person>"


@pytest.mark.unit
def test_polygraf_pii_detection_no_active_pii_detection():
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                polygraf:
                  server_endpoint: http://localhost:8000/v1/pii/text-detect
        """,
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define bot inform answer unknown
              "I can't answer that."
        """,
    )

    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            '  "Hi! My name is John as well."',
        ],
    )

    chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")
    chat.app.register_action(create_mock_polygraf_detect_pii(), "polygraf_detect_pii")
    chat.app.register_action(create_mock_polygraf_mask_pii(), "polygraf_mask_pii")

    chat >> "Hi! I am Mr. John! And my email is test@gmail.com"
    chat << "Hi! My name is John as well."


@pytest.mark.unit
def test_polygraf_pii_detection_input():
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                polygraf:
                  server_endpoint: http://localhost:8000/v1/pii/text-detect
                  input:
                    entities:
                      - Email
                      - Person
              input:
                flows:
                  - polygraf detect pii on input
        """,
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define bot inform answer unknown
              "I can't answer that."
        """,
    )

    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            '  "Hi! My name is John as well."',
        ],
    )

    chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")
    chat.app.register_action(
        create_mock_polygraf_detect_pii(["Email", "Person"]),
        "polygraf_detect_pii",
    )
    chat.app.register_action(
        create_mock_polygraf_mask_pii(["Email", "Person"]),
        "polygraf_mask_pii",
    )

    chat >> "Hi! I am Mr. John! And my email is test@gmail.com"
    chat << "I can't answer that."


@pytest.mark.unit
def test_polygraf_pii_detection_output():
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                polygraf:
                  server_endpoint: http://localhost:8000/v1/pii/text-detect
                  output:
                    entities:
                      - Email
                      - Person
              output:
                flows:
                  - polygraf detect pii on output
        """,
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define bot inform answer unknown
              "I can't answer that."
        """,
    )

    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            '  "Hi! My name is John as well."',
        ],
    )

    chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")
    chat.app.register_action(
        create_mock_polygraf_detect_pii(["Email", "Person"]),
        "polygraf_detect_pii",
    )
    chat.app.register_action(
        create_mock_polygraf_mask_pii(["Email", "Person"]),
        "polygraf_mask_pii",
    )

    chat >> "Hi!"
    chat << "I can't answer that."


@pytest.mark.unit
def test_polygraf_pii_detection_retrieval_with_no_pii():
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                polygraf:
                  server_endpoint: http://localhost:8000/v1/pii/text-detect
                  retrieval:
                    entities:
                      - Email
                      - Person
              retrieval:
                flows:
                  - polygraf detect pii on retrieval
        """,
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define bot inform answer unknown
              "I can't answer that."
        """,
    )

    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            '  "Hi! My name is John as well."',
        ],
    )

    chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")
    chat.app.register_action(
        create_mock_polygraf_detect_pii(["Email", "Person"]),
        "polygraf_detect_pii",
    )
    chat.app.register_action(
        create_mock_polygraf_mask_pii(["Email", "Person"]),
        "polygraf_mask_pii",
    )

    chat >> "Hi!"
    chat << "Hi! My name is John as well."


@pytest.mark.unit
def test_polygraf_pii_masking_on_output():
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                polygraf:
                  server_endpoint: http://localhost:8000/v1/pii/text-detect
                  output:
                    entities:
                      - Email
                      - Person
              output:
                flows:
                  - polygraf mask pii on output
        """,
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define bot inform answer unknown
              "I can't answer that."
        """,
    )

    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            '  "Hi! I am John."',
        ],
    )

    chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")
    chat.app.register_action(
        create_mock_polygraf_detect_pii(["Email", "Person"]),
        "polygraf_detect_pii",
    )
    chat.app.register_action(
        create_mock_polygraf_mask_pii(["Email", "Person"]),
        "polygraf_mask_pii",
    )

    chat >> "Hi!"
    response = chat.app.generate(messages=[{"role": "user", "content": "Hi!"}])
    assert "John" not in response["content"]
    assert "<Person>" in response["content"]


@pytest.mark.unit
def test_polygraf_pii_masking_on_input():
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                polygraf:
                  server_endpoint: http://localhost:8000/v1/pii/text-detect
                  input:
                    entities:
                      - Email
                      - Person
              input:
                flows:
                  - polygraf mask pii on input
                  - check user message
        """,
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define bot inform answer unknown
              "I can't answer that."

            define flow check user message
              execute check_user_message(user_message=$user_message)
        """,
    )

    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            '  "Hi! Nice to meet you.',
        ],
    )

    @action()
    def check_user_message(user_message: str):
        """Check if the user message has PII masked."""
        assert "John" not in user_message
        assert "<Person>" in user_message

    chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")
    chat.app.register_action(check_user_message, "check_user_message")
    chat.app.register_action(
        create_mock_polygraf_detect_pii(["Email", "Person"]),
        "polygraf_detect_pii",
    )
    chat.app.register_action(
        create_mock_polygraf_mask_pii(["Email", "Person"]),
        "polygraf_mask_pii",
    )

    chat >> "Hi there! Are you John?"


@pytest.mark.unit
def test_polygraf_pii_masking_on_retrieval():
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                polygraf:
                  server_endpoint: http://localhost:8000/v1/pii/text-detect
                  retrieval:
                    entities:
                      - Email
                      - Person
              retrieval:
                flows:
                  - polygraf mask pii on retrieval
                  - check relevant chunks
        """,
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define bot inform answer unknown
              "I can't answer that."

            define flow check relevant chunks
              execute check_relevant_chunks(relevant_chunks=$relevant_chunks)
        """,
    )

    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            "  Sorry, I don't have that in my knowledge base.",
        ],
    )

    @action()
    def check_relevant_chunks(relevant_chunks: str):
        """Check if the relevant chunks have PII masked."""
        assert "test@gmail.com" not in relevant_chunks
        assert "<Email>" in relevant_chunks

    @action()
    def retrieve_relevant_chunk_for_masking():
        context_updates = {"relevant_chunks": "John's Email: test@gmail.com"}
        return ActionResult(
            return_value=context_updates["relevant_chunks"],
            context_updates=context_updates,
        )

    chat.app.register_action(retrieve_relevant_chunk_for_masking, "retrieve_relevant_chunks")
    chat.app.register_action(check_relevant_chunks)
    chat.app.register_action(
        create_mock_polygraf_detect_pii(["Email", "Person"]),
        "polygraf_detect_pii",
    )
    chat.app.register_action(
        create_mock_polygraf_mask_pii(["Email", "Person"]),
        "polygraf_mask_pii",
    )

    chat >> "Hey! Can you help me get John's email?"
