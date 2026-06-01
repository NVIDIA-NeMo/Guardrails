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

import pytest

import nemoguardrails.rails.llm.startup.colang_flows as colang_flows
from nemoguardrails.exceptions import InvalidRailsConfigurationError
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.rails.llm.startup.colang_flows import (
    load_default_colang_1_flows,
    load_guardrails_library_flows_and_bot_messages,
    mark_rail_flows_as_system_subflows,
    validate_rail_flow_names,
)


def test_load_default_colang_1_flows_marks_loaded_flows_as_system():
    config = RailsConfig(models=[])

    load_default_colang_1_flows(config)

    assert config.flows
    assert "process user input" in {flow.get("id") for flow in config.flows}
    assert all(flow.get("is_system_flow") is True for flow in config.flows)


def test_load_guardrails_library_flows_and_bot_messages_loads_package_resources():
    config = RailsConfig(models=[])

    load_guardrails_library_flows_and_bot_messages(config)

    flow_ids = {flow.get("id") for flow in config.flows}
    assert "content safety check input" in flow_ids
    assert "regex check input" in flow_ids
    assert all(flow.get("is_system_flow") is True for flow in config.flows)
    assert config.bot_messages["refuse to respond"] == ["I'm sorry, I can't respond to that."]
    assert config.bot_messages["inform cannot engage with sensitive content"] == [
        "I will not engage with sensitive content."
    ]


def test_load_guardrails_library_flows_and_bot_messages_keeps_existing_bot_messages():
    config = RailsConfig(
        models=[],
        bot_messages={"refuse to respond": ["Custom refusal."]},
    )

    load_guardrails_library_flows_and_bot_messages(config)

    assert config.bot_messages["refuse to respond"] == ["Custom refusal."]
    assert config.bot_messages["inform cannot engage with sensitive content"] == [
        "I will not engage with sensitive content."
    ]


def test_colang_flow_loading_does_not_depend_on_colang_flows_file_location(monkeypatch, tmp_path):
    monkeypatch.setattr(colang_flows, "__file__", str(tmp_path / "moved" / "colang_flows.py"))

    default_config = RailsConfig(models=[])
    colang_flows.load_default_colang_1_flows(default_config)

    library_config = RailsConfig(models=[])
    colang_flows.load_guardrails_library_flows_and_bot_messages(library_config)

    assert "process user input" in {flow.get("id") for flow in default_config.flows}
    assert "content safety check input" in {flow.get("id") for flow in library_config.flows}
    assert library_config.bot_messages["inform cannot engage with sensitive content"] == [
        "I will not engage with sensitive content."
    ]


def test_mark_rail_flows_as_system_subflows_mutates_matching_flow_configs():
    config = RailsConfig(
        models=[],
        flows=[
            {"id": "check input", "is_system_flow": False},
            {"id": "regular flow", "is_system_flow": False},
        ],
    )
    config.rails.input.flows = ["check input"]

    mark_rail_flows_as_system_subflows(config)

    assert config.flows[0]["is_system_flow"] is True
    assert config.flows[0]["is_subflow"] is True
    assert config.flows[1]["is_system_flow"] is False
    assert "is_subflow" not in config.flows[1]


class _FakeResource:
    """Minimal Traversable stand-in whose ``iterdir`` order we control."""

    def __init__(self, name, *, is_dir=False, children=None):
        self._name = name
        self._is_dir = is_dir
        self._children = children or []

    @property
    def name(self):
        return self._name

    def is_file(self):
        return not self._is_dir

    def is_dir(self):
        return self._is_dir

    def iterdir(self):
        return iter(self._children)


def test_iter_colang_resources_yields_sorted_filesystem_independent_order():
    # iterdir() returns children in an arbitrary (here: unsorted) order; the traversal
    # must still yield .co files in a deterministic, sorted order so the library load
    # order does not depend on the filesystem (the dropped os.walk + sort behavior).
    resource = _FakeResource(
        "library",
        is_dir=True,
        children=[
            _FakeResource("zebra.co"),
            _FakeResource("alpha.co"),
            _FakeResource("not_colang.txt"),
            _FakeResource(
                "subpack",
                is_dir=True,
                children=[_FakeResource("yray.co"), _FakeResource("beta.co")],
            ),
            _FakeResource("middle.co"),
        ],
    )

    names = [item.name for item in colang_flows._iter_colang_resources(resource)]

    # Files in the directory come first (sorted), then sorted subdirectories are
    # descended into (their files sorted); non-.co files are skipped.
    assert names == ["alpha.co", "middle.co", "zebra.co", "beta.co", "yray.co"]


def test_iter_colang_resources_order_does_not_depend_on_iterdir_order():
    # The same children in two different iterdir() orders must yield the same sequence.
    files_forward = [_FakeResource(f"{name}.co") for name in ("a", "b", "c")]
    files_reversed = list(reversed(files_forward))

    forward = [
        r.name for r in colang_flows._iter_colang_resources(_FakeResource("d", is_dir=True, children=files_forward))
    ]
    backward = [
        r.name for r in colang_flows._iter_colang_resources(_FakeResource("d", is_dir=True, children=files_reversed))
    ]

    assert forward == backward == ["a.co", "b.co", "c.co"]


def test_validate_rail_flow_names_keeps_existing_flow_validation_behavior():
    config = RailsConfig(
        models=[],
        flows=[{"id": "content safety check input"}],
    )
    config.rails.input.flows = ["content safety check input $model=content_safety"]

    validate_rail_flow_names(config)

    config.rails.output.flows = ["missing output rail"]
    with pytest.raises(InvalidRailsConfigurationError, match="`missing output rail` does not exist"):
        validate_rail_flow_names(config)
