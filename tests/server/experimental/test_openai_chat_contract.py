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

"""Validate pinned provider metadata without contacting the provider."""

import re
from pathlib import Path
from urllib.parse import urlparse

import pytest
import yaml

from nemoguardrails.server.experimental.providers.openai import source as openai_pin

REPOSITORY_ROOT = Path(__file__).parents[3]
SOURCE_PATH = REPOSITORY_ROOT / "nemoguardrails/server/experimental/contracts/openai/source.yaml"


@pytest.fixture(scope="module")
def openai_source() -> dict[str, str]:
    """Load source provenance, not a separately authored operation policy."""
    source = yaml.safe_load(SOURCE_PATH.read_text(encoding="utf-8"))
    assert isinstance(source, dict)
    return source


def test_openai_source_declares_revision_and_digest(openai_source: dict[str, str]) -> None:
    """Pin metadata has the expected shape; this does not verify upstream bytes."""
    assert set(openai_source) == {
        "document_url",
        "download_url",
        "revision",
        "document_version",
        "sha256",
    }
    assert re.fullmatch(r"[0-9a-f]{40}", openai_source["revision"])
    assert re.fullmatch(r"[0-9a-f]{64}", openai_source["sha256"])
    assert openai_source["document_version"].strip()


@pytest.mark.parametrize(
    ("key", "host", "prefix"),
    [
        ("document_url", "github.com", "/openai/openai-openapi/blob"),
        ("download_url", "raw.githubusercontent.com", "/openai/openai-openapi"),
    ],
)
def test_openai_source_urls_use_the_declared_revision(
    openai_source: dict[str, str], key: str, host: str, prefix: str
) -> None:
    """Both locations identify the same pinned file rather than a moving branch."""
    url = urlparse(openai_source[key])

    assert url.scheme == "https"
    assert url.netloc == host
    assert url.path == f"{prefix}/{openai_source['revision']}/openapi.yaml"
    assert not url.query
    assert not url.fragment


def test_python_pin_matches_source_metadata(openai_source: dict[str, str]) -> None:
    """The importable pin repeats source.yaml exactly, so the two cannot drift."""
    assert {
        "document_url": openai_pin.PROVIDER_DOCUMENT_URL,
        "download_url": openai_pin.PROVIDER_DOWNLOAD_URL,
        "revision": openai_pin.PROVIDER_REVISION,
        "document_version": openai_pin.PROVIDER_DOCUMENT_VERSION,
        "sha256": openai_pin.PROVIDER_DOCUMENT_SHA256,
    } == openai_source
