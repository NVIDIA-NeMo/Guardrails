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


class AttributeDict(dict):
    """
    A simple utility class that allows accessing dictionary members as attributes.

    This class inherits from the built-in `dict` class and overrides the `__getattr__`
    and `__setattr__` methods to provide attribute-style access to dictionary keys.

    Example:
    ```python
    # Using AttributeDict
    my_dict = AttributeDict({'key': 'value', 'nested': {'inner_key': 'inner_value'}})

    # Accessing dictionary keys as attributes
    print(my_dict.key)            # Output: 'value'
    print(my_dict.nested.inner_key)# Output: 'inner_value'
    ```

    Note:
    - If a dictionary value is itself a dictionary, it is converted to an `AttributeDict`.
    - If a dictionary value is a list of dictionaries, the list is converted to a list of `AttributeDict` objects.
    """

    def __getattr__(self, attr):
        val = self.get(attr, None)
        if isinstance(val, dict):
            return AttributeDict(val)
        elif isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
            return [AttributeDict(x) for x in val]
        else:
            return val

    def __setattr__(self, attr, value):
        self[attr] = value
