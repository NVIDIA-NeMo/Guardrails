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

"""Auto-reload cache eviction for ``nemoguardrails server --auto-reload``.

``_get_rails`` caches instances as ``config_id:model_name``. The watchdog used
to look up the bare ``config_id``, so real /v1/checks traffic never evicted.
"""

import os

import pytest

from nemoguardrails.server import api


class _FakeRails:
    def __init__(self, history=None):
        self.events_history_cache = history if history is not None else {}


@pytest.fixture(autouse=True)
def reset_server_state():
    original_path = api.app.rails_config_path
    original_single_config_mode = api.app.single_config_mode
    original_single_config_id = api.app.single_config_id
    api.app.single_config_mode = False
    api.app.single_config_id = None
    api.llm_rails_instances.clear()
    api.llm_rails_events_history_cache.clear()
    yield
    api.llm_rails_instances.clear()
    api.llm_rails_events_history_cache.clear()
    api.app.rails_config_path = original_path
    api.app.single_config_mode = original_single_config_mode
    api.app.single_config_id = original_single_config_id


def test_generate_cache_key_appends_model_name():
    assert api._generate_cache_key(["regex"]) == "regex"
    assert api._generate_cache_key(["regex"], "gpt-4o") == "regex:gpt-4o"
    assert api._generate_cache_key(["a", "b"], "gpt-4o") == "a-b:gpt-4o"


@pytest.mark.parametrize(
    ("cache_key", "config_id", "expected"),
    [
        ("regex", "regex", True),
        ("regex:gpt-4o", "regex", True),
        ("a-b:gpt-4o", "a", True),
        ("a-b:gpt-4o", "b", True),
        ("regex:gpt-4o", "other", False),
        ("regex2:gpt-4o", "regex", False),
    ],
)
def test_cache_key_matches_config_id(cache_key, config_id, expected):
    assert api._cache_key_matches_config_id(cache_key, config_id) is expected


def test_evict_clears_model_suffixed_keys_and_preserves_other_configs():
    """The reported bug: eviction looked up bare ``regex`` while the live key is ``regex:gpt-4o``."""
    stale = _FakeRails(history={"thread": ["old"]})
    other = _FakeRails(history={"thread": ["keep"]})
    api.llm_rails_instances["regex:gpt-4o"] = stale
    api.llm_rails_instances["other:gpt-4o"] = other

    evicted = api._evict_cached_rails_for_config("regex")

    assert evicted is True
    assert "regex:gpt-4o" not in api.llm_rails_instances
    assert api.llm_rails_instances["other:gpt-4o"] is other
    assert api.llm_rails_events_history_cache["regex:gpt-4o"] == {"thread": ["old"]}


def test_evict_clears_bare_and_multi_config_keys():
    api.llm_rails_instances["regex"] = _FakeRails()
    api.llm_rails_instances["regex-extra:gpt-4o"] = _FakeRails()
    api.llm_rails_instances["unrelated"] = _FakeRails()

    assert api._evict_cached_rails_for_config("regex") is True
    assert set(api.llm_rails_instances) == {"unrelated"}


def test_evict_is_noop_when_config_is_not_cached():
    api.llm_rails_instances["other:gpt-4o"] = _FakeRails()

    assert api._evict_cached_rails_for_config("regex") is False
    assert "other:gpt-4o" in api.llm_rails_instances


def test_config_id_for_watched_path_uses_first_directory(tmp_path):
    api.app.rails_config_path = str(tmp_path)
    config_file = tmp_path / "regex" / "config.yml"
    config_file.parent.mkdir()
    config_file.write_text("rails: {}\n", encoding="utf-8")

    assert api._config_id_for_watched_path(str(config_file)) == "regex"


def test_config_id_for_watched_path_uses_single_config_id(tmp_path):
    api.app.rails_config_path = str(tmp_path)
    api.app.single_config_mode = True
    api.app.single_config_id = "myconfig"
    config_file = tmp_path / "config.yml"
    config_file.write_text("rails: {}\n", encoding="utf-8")

    assert api._config_id_for_watched_path(str(config_file)) == "myconfig"


def test_should_ignore_hidden_and_checkpoint_paths():
    assert api._should_ignore_watched_path(os.path.join("regex", ".hidden.yml")) is True
    assert api._should_ignore_watched_path(os.path.join("regex", ".ipynb_checkpoints", "config.yml")) is True
    assert api._should_ignore_watched_path(os.path.join("regex", "config.yml")) is False


def test_mtime_polling_evicts_suffixed_cache_after_edit(tmp_path):
    """Fallback path for Docker bind mounts where inotify can stop delivering events."""
    api.app.rails_config_path = str(tmp_path)
    config_dir = tmp_path / "regex"
    config_dir.mkdir()
    config_file = config_dir / "config.yml"
    config_file.write_text("rails: {}\n", encoding="utf-8")

    api.llm_rails_instances["regex:gpt-4o"] = _FakeRails()
    previous = api._snapshot_config_mtimes(str(tmp_path))

    config_file.write_text("rails: { input: {} }\n", encoding="utf-8")
    os.utime(config_file, (os.path.getmtime(config_file) + 5, os.path.getmtime(config_file) + 5))
    current = api._snapshot_config_mtimes(str(tmp_path))

    api._evict_configs_for_mtime_changes(previous, current)

    assert "regex:gpt-4o" not in api.llm_rails_instances


def test_mtime_polling_evicts_when_file_is_removed(tmp_path):
    api.app.rails_config_path = str(tmp_path)
    config_dir = tmp_path / "regex"
    config_dir.mkdir()
    config_file = config_dir / "config.yml"
    config_file.write_text("rails: {}\n", encoding="utf-8")

    api.llm_rails_instances["regex:gpt-4o"] = _FakeRails()
    previous = api._snapshot_config_mtimes(str(tmp_path))
    config_file.unlink()
    current = api._snapshot_config_mtimes(str(tmp_path))

    api._evict_configs_for_mtime_changes(previous, current)

    assert "regex:gpt-4o" not in api.llm_rails_instances


def test_watchdog_handler_evicts_model_suffixed_key(tmp_path):
    """End-to-end handler path: a modified config.yml must drop ``config:model`` keys."""
    pytest.importorskip("watchdog")
    from watchdog.events import FileModifiedEvent

    api.app.rails_config_path = str(tmp_path)
    config_file = tmp_path / "regex" / "config.yml"
    config_file.parent.mkdir()
    config_file.write_text("rails: {}\n", encoding="utf-8")
    api.llm_rails_instances["regex:gpt-4o"] = _FakeRails()

    handler = api._make_reload_event_handler()
    handler.on_any_event(FileModifiedEvent(str(config_file)))

    assert "regex:gpt-4o" not in api.llm_rails_instances
