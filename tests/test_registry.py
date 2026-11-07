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

import threading
import time
from typing import Any

import pytest

from nemoguardrails.registry import Registry
from nemoguardrails.singleton import Singleton


class StubRegistry(Registry):
    def validate(self, name: str, item: Any) -> None:
        pass


class SlowValidateRegistry(Registry):
    def validate(self, name: str, item: Any) -> None:
        time.sleep(0.005)


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


class TestThreadSafety:
    def test_singleton_under_concurrent_construction(self):
        class _SlowSingleton(metaclass=Singleton):
            construction_count = 0

            def __init__(self):
                time.sleep(0.01)
                type(self).construction_count += 1

        Singleton._instances.pop(_SlowSingleton, None)

        start = threading.Event()
        instances = []

        def worker():
            start.wait()
            instances.append(_SlowSingleton())

        threads = [threading.Thread(target=worker) for _ in range(16)]
        for t in threads:
            t.start()
        start.set()
        for t in threads:
            t.join()

        assert len(instances) == 16
        assert all(inst is instances[0] for inst in instances)
        assert _SlowSingleton.construction_count == 1

    def test_concurrent_add_no_duplicates(self):
        Singleton._instances.pop(SlowValidateRegistry, None)
        reg = SlowValidateRegistry(enable_validation=True)
        reg.reset()

        start = threading.Event()
        succeeded = []
        duplicate_rejected = []

        def worker():
            start.wait()
            try:
                reg.add("contested", "value")
                succeeded.append(True)
            except ValueError:
                duplicate_rejected.append(True)

        threads = [threading.Thread(target=worker) for _ in range(16)]
        for t in threads:
            t.start()
        start.set()
        for t in threads:
            t.join()

        assert len(succeeded) == 1
        assert len(duplicate_rejected) == 15

    def test_iter_during_concurrent_mutation(self, registry):
        registry.reset()
        for i in range(50):
            registry.add(f"initial-{i}", i)

        stop = threading.Event()
        errors = []

        def writer():
            i = 0
            while not stop.is_set():
                try:
                    registry.add(f"w-{i}", i)
                except ValueError:
                    pass
                i += 1

        def reader():
            try:
                for _ in range(100):
                    for _name in registry:
                        pass
            except Exception as e:
                errors.append(e)

        w = threading.Thread(target=writer)
        r = threading.Thread(target=reader)
        w.start()
        r.start()
        r.join()
        stop.set()
        w.join()

        assert errors == []
