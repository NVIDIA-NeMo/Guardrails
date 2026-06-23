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

"""Offline tests for the LLM-backed extractor.

The model is injected as a `Chat` callable, so the whole extractor is exercised
without any network or LLM. These tests pin the security-critical behavior: a
hostile or broken model response is parsed defensively, clamped back onto the
closed taxonomy and the real tool registry, and fails closed when it cannot be
trusted.
"""

import json

import pytest
from scanner.llm_extractor import (
    ExtractionError,
    LLMExtractor,
    _build_system,
    _chunk,
    _merge,
    _strip_to_json,
)
from scanner.scan import ArgSpec, ExtractedTechnique, ScanContext, SourceDoc
from synthesis.catalog import CLASS_DESCRIPTIONS, CLASS_TO_FACTORY

REGISTRY = {
    "read_account": "Read an account's balance",
    "transfer_funds": "Move money between accounts",
}

CTX = ScanContext(
    docs_dir="(unused)",
    tool_registry=REGISTRY,
    taxonomy=tuple(CLASS_TO_FACTORY),
    class_definitions=dict(CLASS_DESCRIPTIONS),
)


def _doc(text: str = "a document") -> SourceDoc:
    return SourceDoc(id="doc1", url="https://example.org/x", text=text)


def _chat_returning(payload) -> object:
    """A `Chat` that always returns the same response (raw string or JSON-able)."""
    text = payload if isinstance(payload, str) else json.dumps(payload)

    def chat(system: str, user: str) -> str:
        return text

    return chat


def test_bogus_attack_class_is_clamped_to_novel():
    chat = _chat_returning(
        {
            "has_technique": True,
            "summary": "x",
            "attack_class": "totally-made-up",
            "affected_tools": ["transfer_funds"],
            "suggested_params": {},
        }
    )
    [tech] = LLMExtractor(chat).extract(_doc(), CTX)
    assert tech.attack_class == "novel"
    assert tech.tool_mentions == ["transfer_funds"]


def test_unknown_tool_is_dropped():
    chat = _chat_returning(
        {
            "has_technique": True,
            "summary": "x",
            "attack_class": "unbounded-arg",
            "affected_tools": ["ghost_tool"],
            "suggested_params": {},
        }
    )
    [tech] = LLMExtractor(chat).extract(_doc(), CTX)
    assert tech.attack_class == "unbounded-arg"
    assert tech.tool_mentions == []  # ghost_tool clamped away


def test_novel_with_no_tools_yields_nothing():
    chat = _chat_returning(
        {
            "has_technique": True,
            "summary": "x",
            "attack_class": "made-up",
            "affected_tools": ["ghost_tool"],
            "suggested_params": {},
        }
    )
    assert LLMExtractor(chat).extract(_doc(), CTX) == []


def test_non_mapping_params_clamped_to_empty_dict():
    chat = _chat_returning(
        {
            "has_technique": True,
            "summary": "x",
            "attack_class": "unbounded-arg",
            "affected_tools": ["transfer_funds"],
            "suggested_params": ["not", "a", "mapping"],
        }
    )
    [tech] = LLMExtractor(chat).extract(_doc(), CTX)
    assert tech.suggested_params == {}


def test_malformed_model_output_fails_closed():
    assert LLMExtractor(_chat_returning("this is not json at all")).extract(_doc(), CTX) == []


def test_has_technique_false_yields_nothing():
    chat = _chat_returning({"has_technique": False})
    assert LLMExtractor(chat).extract(_doc(), CTX) == []


def test_strip_to_json_tolerates_code_fences():
    raw = '```json\n{"has_technique": false}\n```'
    assert json.loads(_strip_to_json(raw)) == {"has_technique": False}


def test_chunking_splits_long_text_and_bounds_each_chunk():
    text = "\n\n".join(f"paragraph number {i} with some filler" for i in range(20))
    chunks = _chunk(text, size=80, overlap=10)
    assert len(chunks) > 1
    assert all(len(c) <= 80 for c in chunks)


def test_max_chunks_bounds_model_calls():
    calls = []

    def counting_chat(system: str, user: str) -> str:
        calls.append(user)
        return json.dumps({"has_technique": False})

    text = "\n\n".join(f"paragraph {i} padded out a bit" for i in range(12))
    extractor = LLMExtractor(counting_chat, chunk_chars=40, overlap_chars=0, max_chunks=2)
    extractor.extract(_doc(text), CTX)
    assert len(calls) == 2  # capped, even though the doc chunks into more


def test_system_prompt_lists_required_params_per_class():
    ctx = ScanContext(
        docs_dir="(unused)",
        tool_registry=REGISTRY,
        taxonomy=("unbounded-arg",),
        class_definitions={"unbounded-arg": "cap a numeric argument"},
        class_params={"unbounded-arg": ("arg_name", "ceiling")},
    )
    system = _build_system(ctx)
    assert "REQUIRED suggested_params: arg_name, ceiling" in system


def test_system_prompt_lists_tool_arguments():
    ctx = ScanContext(
        docs_dir="(unused)",
        tool_registry={"transfer_funds": "move money"},
        taxonomy=("unbounded-arg",),
        tool_schemas={"transfer_funds": [ArgSpec("amount", "number", "amount to move")]},
    )
    assert "arg amount (number): amount to move" in _build_system(ctx)


def _schema_ctx() -> ScanContext:
    return ScanContext(
        docs_dir="(unused)",
        tool_registry={"transfer_funds": "move money"},
        taxonomy=tuple(CLASS_TO_FACTORY),
        class_definitions=dict(CLASS_DESCRIPTIONS),
        tool_schemas={"transfer_funds": [ArgSpec("amount", "number", "amount to move")]},
        principal_attrs=["mfa_verified"],
    )


def test_hallucinated_arg_name_is_dropped():
    # The model names an argument that does not exist on the affected tool; it is
    # dropped so the candidate fails closed downstream rather than enforcing a
    # wrong-but-valid rule.
    chat = _chat_returning(
        {
            "has_technique": True,
            "summary": "x",
            "attack_class": "unbounded-arg",
            "affected_tools": ["transfer_funds"],
            "suggested_params": {"arg_name": "totally_wrong", "ceiling": 5000},
        }
    )
    [tech] = LLMExtractor(chat).extract(_doc(), _schema_ctx())
    assert "arg_name" not in tech.suggested_params  # dropped: not a real argument
    assert tech.suggested_params["ceiling"] == 5000  # value params are untouched


def test_real_arg_name_is_kept():
    chat = _chat_returning(
        {
            "has_technique": True,
            "summary": "x",
            "attack_class": "unbounded-arg",
            "affected_tools": ["transfer_funds"],
            "suggested_params": {"arg_name": "amount", "ceiling": 5000},
        }
    )
    [tech] = LLMExtractor(chat).extract(_doc(), _schema_ctx())
    assert tech.suggested_params["arg_name"] == "amount"


def test_hallucinated_principal_attr_is_dropped():
    chat = _chat_returning(
        {
            "has_technique": True,
            "summary": "x",
            "attack_class": "privilege-escalation",
            "affected_tools": ["transfer_funds"],
            "suggested_params": {"attr_name": "is_admin", "expected": True},
        }
    )
    [tech] = LLMExtractor(chat).extract(_doc(), _schema_ctx())
    assert "attr_name" not in tech.suggested_params  # not a recognized attribute


def _technique_json() -> str:
    return json.dumps(
        {
            "has_technique": True,
            "summary": "x",
            "attack_class": "unbounded-arg",
            "affected_tools": ["transfer_funds"],
            "suggested_params": {},
        }
    )


def test_backend_unreachable_for_whole_doc_raises_not_silent():
    # A single short doc is one chunk; if that chunk's backend call fails, the
    # extractor must NOT return [] (which would read as "scanned, found nothing").
    def dead_chat(system: str, user: str) -> str:
        raise ConnectionError("connection refused")

    with pytest.raises(ExtractionError):
        LLMExtractor(dead_chat).extract(_doc(), CTX)


def test_partial_chunk_failure_warns_but_returns_results():
    # First chunk's call fails (transient), the rest succeed: degrade, don't abort.
    state = {"n": 0}

    def flaky_chat(system: str, user: str) -> str:
        state["n"] += 1
        if state["n"] == 1:
            raise TimeoutError("slow")
        return _technique_json()

    text = "\n\n".join(f"paragraph {i} with enough text" for i in range(8))
    extractor = LLMExtractor(flaky_chat, chunk_chars=40, overlap_chars=0)
    techniques = extractor.extract(_doc(text), CTX)
    assert state["n"] > 1  # more than one chunk was attempted
    assert any(t.attack_class == "unbounded-arg" for t in techniques)


def test_merge_collapses_per_class_and_unions_tools():
    a = ExtractedTechnique(summary="first", attack_class="unbounded-arg", tool_mentions=["transfer_funds"])
    b = ExtractedTechnique(summary="", attack_class="unbounded-arg", tool_mentions=["read_account"])
    [merged] = _merge([a, b])
    assert merged.summary == "first"
    assert set(merged.tool_mentions) == {"transfer_funds", "read_account"}
