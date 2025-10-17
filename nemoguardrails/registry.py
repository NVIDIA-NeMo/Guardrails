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

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Iterator, List

from .singleton import Singleton


class Registry(ABC, metaclass=Singleton):
    def __init__(self, enable_validation: bool = True) -> None:
        self.items = defaultdict(str)
        self.enable_validation = enable_validation

    @abstractmethod
    def validate(self, name: str, item: Any) -> None:
        pass

    def reset(self):
        self.items = defaultdict(str)

    def add(self, name: str, item: Any):
        """Add an item to the registry.

        Args:
            name (str): The name of the item.
            item (Any): The item to be added.

        Raises:
            ValueError: If the item name already exists in the registry.
        """
        if name in self.items:
            raise ValueError(f"{name} already exists in the registry")
        if self.enable_validation:
            self.validate(name, item)
        self.items[name] = item

    def get(self, name: str) -> Any:
        """Get an item by name.

        Args:
            name (str): The name of the item.

        Raises:
            KeyError: If the item name does not exist in the registry.
        """
        if name is None:
            raise ValueError("name cannot be None")
        if name not in self.items:
            raise KeyError(f"{name} does not exist in the registry")
        return self.items.get(name)

    def list(self) -> List[str]:
        """List all items in the registry.

        Returns:
            List[str]: A list of all item names.
        """
        return list(self.items.keys())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(items={self.items.keys()})"

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[str]:
        return iter(self.items)

    def __contains__(self, name: str) -> bool:
        return name in self.items

    def __getitem__(self, name: str) -> Any:
        return self.get(name)
