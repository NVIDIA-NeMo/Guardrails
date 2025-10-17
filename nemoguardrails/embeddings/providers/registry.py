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


from typing import Any

from nemoguardrails.registry import Registry


class EmbeddingProviderRegistry(Registry):
    def validate(self, name: str, item: Any) -> None:
        """Validate the item to be registered.

        Raises:
            TypeError: If an item is not an instance of EmbeddingModel.
            ValueError: If an item does not have 'encode' or 'encode_async' methods.
        """
        if not callable(getattr(item, "encode", None)):
            raise ValueError(f"{name} does not have an 'encode' method")

        if not callable(getattr(item, "encode_async", None)):
            raise ValueError(f"{name} does not have an 'encode_async' method")
