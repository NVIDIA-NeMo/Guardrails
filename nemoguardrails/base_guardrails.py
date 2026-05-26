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

"""Abstract base class for guardrails engines.

`BaseGuardrails` defines the minimum public surface every guardrails engine
must implement. Concrete engines (`LLMRails`, `IORails`) inherit and provide
real implementations. The `Guardrails` facade also inherits, presenting the
same surface while delegating to a wrapped engine.

The contract is deliberately minimal: only what is truly shared across all
engines belongs here. Engine-specific features (e.g. `update_llm`, `check`,
`register_action`, `runtime`, `explain_info`) remain on the concrete classes
that actually provide them.
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from nemoguardrails.rails.llm.config import RailsConfig


class BaseGuardrails(ABC):
    """Minimum public surface shared by all guardrails engines.

    Subclasses must set ``self.config`` (a :class:`RailsConfig` instance) in
    their ``__init__``. The bare annotation below is informational only — it
    is not enforced by the ABC machinery, so a subclass that forgets to assign
    ``self.config`` will instantiate fine and only fail on first access.
    """

    config: RailsConfig

    @abstractmethod
    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Generate an LLM response synchronously with guardrails applied."""
        ...

    @abstractmethod
    async def generate_async(self, *args: Any, **kwargs: Any) -> Any:
        """Generate an LLM response asynchronously with guardrails applied."""
        ...

    @abstractmethod
    def stream_async(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        """Stream LLM response tokens with guardrails applied."""
        ...
