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

import logging
import re


class SimplifyFormatter(logging.Formatter):
    """A formatter to simplify the log messages for easy reading."""

    def format(self, record):
        text: str
        if isinstance(record, str):
            text = record
        else:
            text = super().format(record)

        # Replace all UUIDs
        pattern = re.compile(
            r"([0-9a-fA-F]{8}[-_][0-9a-fA-F]{4}[-_][0-9a-fA-F]{4}[-_][0-9a-fA-F]{4}[-_][0-9a-fA-F]{12})"
        )
        text = pattern.sub(lambda m: m.group(1)[:4] + "...", text)

        # Replace time stamps
        pattern = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}[+-]\d{2}:\d{2}")
        text = pattern.sub(lambda m: "...", text)

        # Hide certain event properties
        fields_to_hide = [
            "_created_at",
            "_finished_at",
            "_started_at",
            "source_uid",
            "action_info_modality",
            "action_info_modality_policy",
        ]

        pattern = re.compile(r"(, )?'[^']*(?:" + "|".join(fields_to_hide) + ")': '[^']*'")
        text = pattern.sub("", text)

        # Hide main loop id
        text = re.sub(r"\{'loop_id': '(main)[^']+'}", "", text)

        # Hide Object references
        text = re.sub(r"<[^>]* object at [^>]*>", "<>", text)

        # Emphasize certain bits
        pattern = re.compile(r"('final_transcript': '[^']*')")
        text = pattern.sub(lambda m: ":thumbs_up:" + m.group(0) + ":thumbs_up:", text)

        # Remove certain bits
        text = text.replace("Process internal event: ", "")

        # Hide completely the line with "Processing event"
        if text.startswith("Processing event"):
            text = ""

        # Hide lines with special formatting from the VerboseHandler
        if " :: " in text:
            text = ""

        return text
