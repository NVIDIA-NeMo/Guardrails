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

from nemoguardrails.manifests import all_rail_manifests
from scripts.generate_rail_requirements_docs import (
    BLOCK_END,
    BLOCK_START,
    DOCUMENT_PATH,
    catalog_documents_are_current,
    document_is_current,
    render_document,
    render_page_requirements,
    update_page_requirements,
    write_document,
)


def test_generated_rail_installation_document_covers_manifest_requirements():
    manifests = all_rail_manifests()

    document = render_document(manifests.values())

    assert all(f"(`{rail_name}`)" in document for rail_name in manifests)
    assert "`presidio-analyzer>=2.2; python_version < '3.13'`" in document
    assert "`NVIDIA_API_KEY` (optional)" in document
    assert "`spacy:en_core_web_lg` (required)" in document
    assert "pip install cleanlab-studio" in document


def test_generated_rail_installation_document_detects_stale_output(tmp_path):
    path = tmp_path / "rail-installation-requirements.mdx"

    assert not document_is_current(path)

    write_document(path)

    assert document_is_current(path)
    path.write_text("stale\n", encoding="utf-8")
    assert not document_is_current(path)


def test_generated_page_requirements_preserve_manual_content_and_are_idempotent():
    manifest = all_rail_manifests()["cleanlab"]
    document = "---\ntitle: Cleanlab\n---\n\nIntroduction.\n\n## Setup\n\nManual setup.\n"

    generated = update_page_requirements(document, (manifest,))

    assert generated.index("Introduction.") < generated.index(BLOCK_START)
    assert generated.index(BLOCK_END) < generated.index("## Setup")
    assert "pip install cleanlab-studio" in generated
    assert "Manual setup." in generated
    assert update_page_requirements(generated, (manifest,)) == generated


def test_generated_page_requirements_show_shared_page_provenance():
    manifests = all_rail_manifests()

    generated = render_page_requirements((manifests["self_check.input_check"], manifests["self_check.output_check"]))

    assert "`llm` (required; rails: `self_check.input_check`, `self_check.output_check`)" in generated
    assert "--rail self_check.input_check --rail self_check.output_check" in generated


def test_committed_rail_installation_document_is_current():
    assert DOCUMENT_PATH.exists()
    assert document_is_current()
    assert catalog_documents_are_current(all_rail_manifests().values())
