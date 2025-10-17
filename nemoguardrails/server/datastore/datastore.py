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

from typing import Optional


class DataStore:
    """A basic data store interface."""

    async def set(self, key: str, value: str):
        """Save data into the datastore.

        Args:
            key: The key to use.
            value: The value associated with the key.

        Returns:
            None
        """
        raise NotImplementedError()

    async def get(self, key: str) -> Optional[str]:
        """Return the value for the specified key.
        Args:
            key: The key to lookup.

        Returns:
            None if the key does not exist.
        """
        raise NotImplementedError()
