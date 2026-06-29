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

from __future__ import annotations

import json
from typing import Any

from nemoguardrails.rails.llm.options import GenerationResponse, RailsResult
from tests.recorded.cassette import normalize_body


def normalize_rails_result(result: RailsResult) -> dict[str, Any]:
    return normalize_body(
        {
            "status": result.status.value,
            "rail": result.rail,
            "content": result.content,
        }
    )


def normalize_llm_calls(result: GenerationResponse) -> list[dict[str, Any]]:
    if result.log is None or result.log.llm_calls is None:
        return []
    calls = []
    for call in result.log.llm_calls:
        calls.append(
            {
                "task": getattr(call, "task", None),
                "provider": call.llm_provider_name,
                "model": call.llm_model_name,
                "completion": call.completion,
                "prompt_tokens": getattr(call, "prompt_tokens", None),
                "completion_tokens": getattr(call, "completion_tokens", None),
                "total_tokens": getattr(call, "total_tokens", None),
            }
        )
    return normalize_body(calls)


def normalize_generation_response(result: GenerationResponse) -> dict[str, Any]:
    activated_rails = []
    if result.log is not None:
        activated_rails = [
            {
                "type": rail.type,
                "name": rail.name,
                "decisions": rail.decisions,
                "stop": rail.stop,
            }
            for rail in result.log.activated_rails
        ]
    return normalize_body(
        {
            "response": result.response,
            "activated_rails": activated_rails,
            "llm_calls": normalize_llm_calls(result),
        }
    )


def normalize_stream_chunks(chunks: list[Any]) -> dict[str, Any]:
    content_parts = []
    errors = []
    normalized_chunks = []
    for chunk in chunks:
        if isinstance(chunk, str):
            if chunk.startswith('{"error":'):
                errors.append(json.loads(chunk))
            else:
                content_parts.append(chunk)
            normalized_chunks.append(chunk)
        elif isinstance(chunk, dict):
            text_value = chunk.get("text")
            content_value = chunk.get("content")
            text = (
                text_value if isinstance(text_value, str) else content_value if isinstance(content_value, str) else None
            )
            if isinstance(text, str):
                content_parts.append(text)
            metadata = chunk.get("metadata") or {}
            normalized = {key: chunk[key] for key in ("text", "content") if key in chunk}
            if metadata.get("usage"):
                normalized["usage"] = metadata["usage"]
            normalized_chunks.append(normalized)
    return normalize_body(
        {
            "content": "".join(content_parts),
            "chunks": normalized_chunks,
            "errors": errors,
        }
    )
