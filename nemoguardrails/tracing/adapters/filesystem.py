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


from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from nemoguardrails.tracing import InteractionLog

from nemoguardrails.tracing.adapters.base import InteractionLogAdapter
from nemoguardrails.tracing.span_formatting import (
    format_span_for_filesystem,
    get_schema_version_for_filesystem,
)


class FileSystemAdapter(InteractionLogAdapter):
    name = "FileSystem"

    def __init__(self, filepath: Optional[str] = None):
        if not filepath:
            self.filepath = "./.traces/trace.jsonl"
        else:
            self.filepath = os.path.abspath(filepath)
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

    def transform(self, interaction_log: "InteractionLog"):
        """Transforms the InteractionLog into a JSON string."""
        spans = [format_span_for_filesystem(span_data) for span_data in interaction_log.trace]

        if not interaction_log.trace:
            schema_version = None
        else:
            schema_version = get_schema_version_for_filesystem(interaction_log.trace[0])

        log_dict = {
            "schema_version": schema_version,
            "trace_id": interaction_log.id,
            "spans": spans,
        }

        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_dict) + "\n")

    async def transform_async(self, interaction_log: "InteractionLog"):
        try:
            import aiofiles
        except ImportError:
            raise ImportError(
                "aiofiles is required for async file writing. Please install it using `pip install aiofiles`"
            )

        spans = [format_span_for_filesystem(span_data) for span_data in interaction_log.trace]

        if not interaction_log.trace:
            schema_version = None
        else:
            schema_version = get_schema_version_for_filesystem(interaction_log.trace[0])

        log_dict = {
            "schema_version": schema_version,
            "trace_id": interaction_log.id,
            "spans": spans,
        }

        async with aiofiles.open(self.filepath, "a", encoding="utf-8") as f:
            await f.write(json.dumps(log_dict) + "\n")
