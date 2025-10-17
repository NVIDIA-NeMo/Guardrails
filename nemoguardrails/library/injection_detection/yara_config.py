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

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from enum import Enum, EnumMeta


class YaraEnumMeta(EnumMeta):
    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        else:
            return True

    def __repr__(cls):
        return ", ".join([member.value for member in list(cls)])

    def __le__(cls, other):
        if isinstance(other, list):
            other = set(other)
        if isinstance(other, set):
            values = {member.value for member in list(cls)}
            return values <= other
        else:
            raise TypeError(
                f"Comparison not supported between instances of '{type(other)}' and '{cls.__name__}'"
            )

    def __ge__(cls, other):
        if isinstance(other, list):
            other = set(other)
        if isinstance(other, set):
            values = {member.value for member in list(cls)}
            return values >= other
        else:
            raise TypeError(
                f"Comparison not supported between instances of '{type(other)}' and '{cls.__name__}'"
            )


class Rules(Enum, metaclass=YaraEnumMeta):
    SQLI = "sqli"
    TEMPLATE = "template"
    CODE = "code"
    XSS = "xss"


class ActionOptions(Enum, metaclass=YaraEnumMeta):
    REJECT = "reject"
    OMIT = "omit"
    SANITIZE = "sanitize"
