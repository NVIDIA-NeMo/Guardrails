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

"""Guarded HTTP route ownership declarations."""

import re
from dataclasses import dataclass

from starlette.convertors import CONVERTOR_TYPES, PathConvertor

HTTP_METHODS = ("DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT")


_PARAMETER = re.compile(r"{([a-zA-Z_][a-zA-Z0-9_]*)(:[a-zA-Z_][a-zA-Z0-9_]*)?}")


def _compile_guarded_path(path: str) -> re.Pattern[str]:
    pieces = ["^"]
    names = set()
    position = 0
    for parameter in _PARAMETER.finditer(path):
        name, converter_name = parameter.groups("str")
        converter = CONVERTOR_TYPES.get(converter_name.lstrip(":"))
        if converter is None or name in names:
            raise ValueError("A guarded HTTP path must be a valid route template.")
        if isinstance(converter, PathConvertor):
            raise ValueError("A guarded HTTP path must not contain a path-spanning parameter.")
        names.add(name)
        pieces.extend((re.escape(path[position : parameter.start()]), f"(?P<{name}>{converter.regex})"))
        position = parameter.end()
    pieces.extend((re.escape(path[position:]), "$"))
    return re.compile("".join(pieces))


@dataclass(frozen=True, slots=True)
class GuardedOperationPath:
    """Describe a path shape owned by one guarded provider operation."""

    route_path: str
    methods: frozenset[str] = frozenset({"POST"})

    def __post_init__(self) -> None:
        """Validate the route template and its allowed methods."""

        if not self.route_path.startswith("/") or self.route_path == "/" or self.route_path.endswith("/"):
            raise ValueError("A guarded HTTP path must be absolute, non-root, and have no trailing slash.")
        _compile_guarded_path(self.route_path)
        if not self.methods or any(method not in HTTP_METHODS for method in self.methods):
            raise ValueError("Guarded HTTP methods must be supported uppercase methods.")

    def matches(self, path: str) -> bool:
        """Return whether a concrete path belongs to this operation."""

        return _compile_guarded_path(self.route_path).fullmatch(path) is not None
