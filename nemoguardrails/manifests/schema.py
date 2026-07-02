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

"""Building blocks for declaring typed, per-rail configuration.

Rails define their configuration as a `RailConfigBaseModel` (a Pydantic base
that ignores unknown keys) with fields built via `rail_field`.
`RailConfigSpec` describes a single configuration field: its annotation, the
Pydantic `FieldInfo`, and any named values it exports to the runtime.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, cast

from pydantic import BaseModel as _PydanticBaseModel
from pydantic import ConfigDict as _ConfigDict
from pydantic import Discriminator, Field, PrivateAttr, SecretStr, model_validator
from pydantic.fields import FieldInfo


@dataclass(frozen=True)
class RailConfigSpec:
    annotation: Any
    field_info: FieldInfo
    exports: Dict[str, Any] = field(default_factory=dict)
    key: Optional[str] = field(default=None, kw_only=True)


setattr(RailConfigSpec, "__hash__", None)


def rail_field(*args: Any, **kwargs: Any) -> FieldInfo:
    return cast(FieldInfo, Field(*args, **kwargs))


class RailConfigBaseModel(_PydanticBaseModel):
    model_config = _ConfigDict(extra="ignore")


__all__ = [
    "Discriminator",
    "Field",
    "PrivateAttr",
    "RailConfigBaseModel",
    "RailConfigSpec",
    "SecretStr",
    "model_validator",
    "rail_field",
]
