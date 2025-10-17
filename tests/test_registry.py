# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Any

import pytest

from nemoguardrails.registry import Registry


class StubRegistry(Registry):
    def validate(self, name: str, item: Any) -> None:
        pass


@pytest.fixture()
def registry():
    # Create a new registry before each test
    registry = StubRegistry(enable_validation=False)
    # Yield the registry to the test
    yield registry
    # Reset the registry after each test as it is a singleton
    registry.reset()


def test_add_and_get_item(registry):
    registry.add("item1", "value1")
    assert registry.get("item1") == "value1"


def test_add_existing_item_raises_error(registry):
    registry.add("item1", "value1")
    with pytest.raises(ValueError):
        registry.add("item1", "value2")


def test_get_non_existent_item_raises_error(registry):
    with pytest.raises(KeyError):
        registry.get("non_existent_item")


def test_list_items(registry):
    registry.add("item1", "value1")
    registry.add("item2", "value2")
    assert set(registry.list()) == {"item1", "item2"}


def test_len(registry):
    registry.add("item1", "value1")
    registry.add("item2", "value2")
    assert len(registry) == 2


def test_contains(registry):
    registry.add("item1", "value1")
    assert "item1" in registry
    assert "non_existent_item" not in registry


def test_get_item(registry):
    registry.add("item1", "value1")
    assert registry["item1"] == "value1"


def test_reset(registry):
    registry.add("item1", "value1")
    registry.reset()
    assert len(registry) == 0
