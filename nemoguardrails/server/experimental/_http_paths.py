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
from urllib.parse import unquote_to_bytes

from starlette.convertors import (
    CONVERTOR_TYPES,
    FloatConvertor,
    IntegerConvertor,
    PathConvertor,
    StringConvertor,
    UUIDConvertor,
)
from starlette.routing import compile_path

HTTP_METHODS = ("DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT")


_PARAMETER = re.compile(r"{([a-zA-Z_][a-zA-Z0-9_]*)(:[a-zA-Z_][a-zA-Z0-9_]*)?}")


def _strip_mount_prefix(path: str, raw_path: bytes, mount_path: str) -> bytes:
    """Match a decoded mount while preserving the encoded suffix.

    The mount and its following decoded slash are replaced by one literal
    separator. Percent escapes in the remaining suffix are untouched.
    """

    if path == mount_path:
        raw_mount = mount_path.encode("utf-8")
    elif path.startswith(mount_path + "/"):
        raw_mount = (mount_path + "/").encode("utf-8")
    else:
        raise ValueError("The outbound request path is outside the configured mount path.")
    offset = 0
    for character in raw_mount:
        token = raw_path[offset : offset + 1]
        width = 1
        if token == b"%":
            decoded = unquote_to_bytes(raw_path[offset : offset + 3])
            if len(decoded) == 1:
                token, width = decoded, 3
        if token != bytes((character,)):
            raise ValueError("The encoded outbound path does not match the configured mount path.")
        offset += width
    if path == mount_path and offset != len(raw_path):
        raise ValueError("The encoded outbound path does not match the configured mount path.")
    return b"/" + raw_path[offset:]


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
    passthrough_methods: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        """Validate the route template and its allowed methods."""

        if not self.route_path.startswith("/") or self.route_path == "/" or self.route_path.endswith("/"):
            raise ValueError("A guarded HTTP path must be absolute, non-root, and have no trailing slash.")
        _compile_guarded_path(self.route_path)
        if not self.methods or any(method not in HTTP_METHODS for method in self.methods):
            raise ValueError("Guarded HTTP methods must be supported uppercase methods.")
        if any(method not in HTTP_METHODS for method in self.passthrough_methods):
            raise ValueError("Pass-through HTTP methods must be supported uppercase methods.")
        if self.methods & self.passthrough_methods:
            raise ValueError("Guarded and pass-through HTTP methods must not overlap.")

    def matches(self, path: str) -> bool:
        """Return whether a concrete path belongs to this operation."""

        return _compile_guarded_path(self.route_path).fullmatch(path) is not None

    def intersects_mount(self, mount_path: str) -> bool:
        _, path_format, convertors = compile_path(self.route_path)
        segment_convertors = (StringConvertor, IntegerConvertor, FloatConvertor, UUIDConvertor)
        if any(type(convertor) not in segment_convertors for convertor in convertors.values()):
            prefix = path_format.split("{", 1)[0]
            return prefix.startswith(mount_path + "/") or mount_path.startswith(prefix) or prefix == mount_path
        route_segments = self.route_path.strip("/").split("/")
        mount_segments = mount_path.strip("/").split("/") if mount_path else []
        if len(mount_segments) > len(route_segments):
            return False
        return all(
            _compile_guarded_path("/" + route_segment).fullmatch("/" + mount_segment) is not None
            for route_segment, mount_segment in zip(route_segments, mount_segments)
        )
