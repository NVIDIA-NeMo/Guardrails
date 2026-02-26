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


from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

LLMMessage: TypeAlias = dict[str, str]  # e.g. {"role": "user", "content": "What can you do?"}
LLMMessages: TypeAlias = list[LLMMessage]


class RailDirection(Enum):
    """Direction of a rail check, used for logging."""

    INPUT = "Input"
    OUTPUT = "Output"


@dataclass(frozen=True, slots=True)
class RailResult:
    """Result of a rail safety check."""

    is_safe: bool
    reason: str | None = None
