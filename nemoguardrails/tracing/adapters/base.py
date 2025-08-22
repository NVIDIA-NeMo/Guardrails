# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from abc import ABC, abstractmethod
from typing import Optional

from nemoguardrails.tracing.interaction_types import InteractionLog


class InteractionLogAdapter(ABC):
    name: Optional[str] = None

    @abstractmethod
    def transform(self, interaction_log: InteractionLog):
        """Transforms the InteractionLog into the backend-specific format."""
        pass

    @abstractmethod
    async def transform_async(self, interaction_log: InteractionLog):
        """Transforms the InteractionLog into the backend-specific format asynchronously."""
        raise NotImplementedError

    async def close(self):
        """Placeholder for any cleanup actions if needed."""
        pass

    async def __aenter__(self):
        """Enter the runtime context related to this object."""
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        """Exit the runtime context related to this object."""
        await self.close()
