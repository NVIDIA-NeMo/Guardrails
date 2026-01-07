# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""GLiNER server package for PII detection and entity extraction."""

from .models import (
    EntitySpan,
    GLiNERRequest,
    GLiNERResponse,
    ModelInfo,
    ModelsResponse,
)
from .pii_utils import (
    DEFAULT_CATEGORIES,
    DEFAULT_LABELS,
    adjust_entity_positions,
    create_tagged_text,
    create_text_chunks,
    deduplicate_entities_by_score,
    process_raw_entities,
    remove_subset_entities,
)

__all__ = [
    # Models
    "EntitySpan",
    "GLiNERRequest",
    "GLiNERResponse",
    "ModelInfo",
    "ModelsResponse",
    # Constants
    "DEFAULT_CATEGORIES",
    "DEFAULT_LABELS",
    # Functions
    "adjust_entity_positions",
    "create_tagged_text",
    "create_text_chunks",
    "deduplicate_entities_by_score",
    "process_raw_entities",
    "remove_subset_entities",
]
