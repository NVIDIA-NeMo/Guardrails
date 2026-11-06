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

import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml

from tests.server.experimental.conformance.openai.chat_completions.fixtures import extract_source as chat
from tests.server.experimental.conformance.openai.responses.fixtures import extract_source as responses


@pytest.mark.parametrize("extractor", [chat, responses], ids=["chat", "responses"])
def test_fixture_regeneration_preserves_bytes_across_platform_newlines(tmp_path, monkeypatch, extractor):
    document = {
        "openapi": "3.1.0",
        "info": {"title": "test source", "version": "1", "license": {"name": "MIT"}},
        "paths": {
            "/chat/completions": {"post": {"schema": {"$ref": "#/components/schemas/Response"}}},
            "/responses": {
                "post": {"x-oaiMeta": {"examples": [{"title": "Text input", "response": json.dumps({"output": []})}]}}
            },
        },
        "components": {
            "schemas": {name: {"properties": {"field": {"type": "string"}}} for name in responses.COMPONENTS}
        },
    }
    source = tmp_path / "source.yaml"
    source.write_bytes(yaml.safe_dump(document).encode("utf-8"))
    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_bytes(
        yaml.safe_dump({"revision": "test", "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}).encode("utf-8")
    )
    original_open = Path.open
    outputs = []
    for platform_newline in ("\n", "\r\n"):

        def platform_open(path, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
            if "w" in mode and "b" not in mode and newline is None:
                newline = platform_newline
            return original_open(path, mode, buffering, encoding, errors, newline)

        output = tmp_path / f"output-{len(outputs)}"
        with monkeypatch.context() as patch:
            patch.setattr(Path, "open", platform_open)
            patch.setattr(sys, "argv", ["extract_source.py", str(source), str(descriptor), str(output)])
            extractor.main()
        outputs.append(output.read_bytes())

    assert outputs[0].endswith(b"\n")
    assert b"\r\n" not in outputs[0]
    assert outputs[0] == outputs[1]
