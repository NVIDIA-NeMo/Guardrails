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

from typing import Optional, Type

from nemoguardrails.registry import Registry


class LogAdapterRegistry(Registry):
    def validate(self, name: str, item: Type) -> None:
        """Validate the item to be registered.
        Raises:
            TypeError: If an item is not an instance of InteractionLogAdapter.
        """
        # Deferred import to avoid circular imports
        from nemoguardrails.tracing.adapters.base import InteractionLogAdapter

        if not issubclass(item, InteractionLogAdapter):
            raise TypeError(f"{name} is not an instance of InteractionLogAdapter")


def register_log_adapter(model: Type, name: Optional[str] = None):
    """Register an embedding provider.

    Args:
        model (Type[EmbeddingModel]): The embedding model class.
        name (str): The name of the embedding engine.

    Raises:
        ValueError: If the engine name is not provided and the model does not have an engine name.
        TypeError: If the model is not an instance of `EmbeddingModel`.
        ValueError: If the model does not have 'encode' or 'encode_async' methods.
    """

    if not name:
        name = model.name

    if not name:
        raise ValueError(
            "The engine name must be provided either in the model or as an argument."
        )

    registry = LogAdapterRegistry()
    registry.add(name, model)
