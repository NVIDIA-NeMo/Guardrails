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

"""Native judgment types shared by decision adapters and their consumers."""

from dataclasses import dataclass
from typing import ClassVar, Literal


@dataclass(frozen=True)
class NoulQuestion:
    instructions: str
    type: ClassVar[Literal["noul"]] = "noul"


@dataclass(frozen=True)
class ChoiceQuestion:
    instructions: str
    choices: dict[str, str]
    type: ClassVar[Literal["choice"]] = "choice"


Question = NoulQuestion | ChoiceQuestion


@dataclass(frozen=True)
class NoulAnswer:
    probability: float


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    probabilities: dict[str, float] | None
