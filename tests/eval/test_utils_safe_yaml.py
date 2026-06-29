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


import importlib
import json

import pytest
import yaml

import nemoguardrails.eval.utils as eval_utils
from nemoguardrails.eval.utils import (
    load_dict_from_file,
    load_dict_from_path,
    save_dict_to_file,
)


# Create an arbitrary object and make sure we don't serialize it for security
class _CustomObject:
    pass


def test_load_dict_from_file_rejects_python_object_tags(tmp_path):
    """A YAML file with a !!python/object/apply tag must be rejected by the
    safe loader instead of instantiating the object, and the side-effecting
    call it encodes must never run."""
    sentinel = tmp_path / "pwned"
    config_file = tmp_path / "config.yaml"
    config_file.write_text(f"models:\n- !!python/object/apply:os.makedirs [{json.dumps(str(sentinel))}]\n")

    with pytest.raises(yaml.constructor.ConstructorError):
        load_dict_from_file(str(config_file))

    assert not sentinel.exists()


def test_load_dict_from_path_rejects_python_object_tags(tmp_path):
    """The recursive directory loader must also reject !!python/object tags in
    any file it walks rather than executing them at load time."""
    sentinel = tmp_path / "pwned"
    (tmp_path / "config.yaml").write_text(
        f"models:\n- !!python/object/apply:os.makedirs [{json.dumps(str(sentinel))}]\n"
    )

    with pytest.raises(yaml.constructor.ConstructorError):
        load_dict_from_path(str(tmp_path))

    assert not sentinel.exists()


def test_load_dict_from_file_loads_benign_yaml(tmp_path):
    """The safe loader must still parse ordinary YAML content unchanged."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("models:\n- name: main\n  engine: openai\nkey: value\n")

    result = load_dict_from_file(str(config_file))

    assert result == {"models": [{"name": "main", "engine": "openai"}], "key": "value"}


def test_load_dict_from_file_loads_json(tmp_path):
    """A .json file must be parsed through the json branch of load_dict_from_file."""
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"models": [{"name": "main"}], "key": "value"}))

    result = load_dict_from_file(str(config_file))

    assert result == {"models": [{"name": "main"}], "key": "value"}


def test_dumper_falls_back_to_safe_dumper_without_libyaml(monkeypatch):
    """When the optional libyaml CSafeDumper is unavailable, the import must
    fall back to the pure-Python SafeDumper, never the unsafe Dumper."""
    monkeypatch.delattr(yaml, "CSafeDumper", raising=False)
    try:
        importlib.reload(eval_utils)
        assert eval_utils.Dumper is yaml.SafeDumper
    finally:
        monkeypatch.undo()
        importlib.reload(eval_utils)


def test_save_dict_to_file_refuses_python_object_tags(tmp_path):
    """The dumper must refuse to serialize arbitrary Python objects"""
    output = tmp_path / "out.yaml"

    with pytest.raises(yaml.representer.RepresenterError):
        save_dict_to_file({"value": _CustomObject()}, str(output))


def test_save_dict_to_file_roundtrips_benign_data(tmp_path):
    """Ordinary primitive data must still serialize and load back unchanged."""
    data = {"results": [{"id": "a", "score": 1}], "logs": ["ok"]}
    output = tmp_path / "out.yaml"

    save_dict_to_file(data, str(output))

    assert load_dict_from_file(str(output)) == data
