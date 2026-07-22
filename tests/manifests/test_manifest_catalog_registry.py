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

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

import nemoguardrails.manifests as manifests_pkg
from nemoguardrails.manifests import RailCatalog, registry


def test_package_accessors_with_built_in_rails():
    registry._reset_rail_manifest_cache()
    try:
        catalog = manifests_pkg.default_rail_catalog()
        assert isinstance(catalog, RailCatalog)
        # a second call returns the cached instance rather than rediscovering
        assert manifests_pkg.default_rail_catalog() is catalog

        assert manifests_pkg.all_rail_manifests() == dict(catalog.manifests)
        assert manifests_pkg.all_rail_manifests()
    finally:
        registry._reset_rail_manifest_cache()


def test_default_catalog_serializes_concurrent_discovery(monkeypatch):
    registry._reset_rail_manifest_cache()
    catalog = RailCatalog(())
    discovery_started = threading.Event()
    release_discovery = threading.Event()
    second_lock_attempted = threading.Event()
    calls = 0

    class TrackingLock:
        def __init__(self):
            self._lock = threading.RLock()

        def __enter__(self):
            if discovery_started.is_set():
                second_lock_attempted.set()
            self._lock.acquire()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self._lock.release()

    def discover_built_ins():
        nonlocal calls
        calls += 1
        discovery_started.set()
        assert release_discovery.wait(timeout=5)
        return catalog

    monkeypatch.setattr(RailCatalog, "discover_built_ins", discover_built_ins)
    monkeypatch.setattr(registry, "_lock", TrackingLock())
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(registry.default_rail_catalog)
            assert discovery_started.wait(timeout=5)
            second = executor.submit(registry.default_rail_catalog)
            assert second_lock_attempted.wait(timeout=5)
            release_discovery.set()

            assert first.result(timeout=5) is catalog
            assert second.result(timeout=5) is catalog
        assert calls == 1
    finally:
        registry._reset_rail_manifest_cache()


def test_default_catalog_rejects_same_thread_reentry(monkeypatch):
    registry._reset_rail_manifest_cache()

    def discover_built_ins():
        return registry.default_rail_catalog()

    monkeypatch.setattr(RailCatalog, "discover_built_ins", discover_built_ins)
    try:
        with pytest.raises(RuntimeError, match="re-entered"):
            registry.default_rail_catalog()
    finally:
        registry._reset_rail_manifest_cache()
