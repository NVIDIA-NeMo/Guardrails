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
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from urllib.parse import unquote_to_bytes

from starlette.convertors import (
    CONVERTOR_TYPES,
    FloatConvertor,
    IntegerConvertor,
    PathConvertor,
    StringConvertor,
    UUIDConvertor,
)

HTTP_METHODS = ("DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT")


_PARAMETER = re.compile(r"{([a-zA-Z_][a-zA-Z0-9_]*)(:[a-zA-Z_][a-zA-Z0-9_]*)?}")


def _compile_route(path: str):
    from starlette.routing import compile_path

    return compile_path(path)


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
        _, path_format, convertors = _compile_route(self.route_path)
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


def _route_shape(path: str) -> str:
    """Normalize route parameter names for duplicate detection."""

    _, path_format, convertors = _compile_route(path)
    for name, convertor in convertors.items():
        path_format = path_format.replace(f"{{{name}}}", f"{{{type(convertor).__name__}}}")
    return path_format


def _route_boundaries_overlap(left: str, right: str) -> bool:
    """Compare the fixed prefix and suffix of two compiled route formats."""

    left_prefix, right_prefix = left.split("{", 1)[0], right.split("{", 1)[0]
    if not (left_prefix.startswith(right_prefix) or right_prefix.startswith(left_prefix)):
        return False
    left_suffix, right_suffix = left.rsplit("}", 1)[-1], right.rsplit("}", 1)[-1]
    return left_suffix.endswith(right_suffix) or right_suffix.endswith(left_suffix)


def _route_segments_overlap(left: str, right: str) -> bool:
    """Return whether two segments may overlap."""

    left_regex, left_format, left_convertors = _compile_route(f"/{left}")
    right_regex, right_format, right_convertors = _compile_route(f"/{right}")
    if not left_convertors:
        return right_regex.fullmatch(f"/{left}") is not None
    if not right_convertors:
        return left_regex.fullmatch(f"/{right}") is not None
    return _route_boundaries_overlap(left_format, right_format)


def _routes_overlap(left: str, right: str) -> bool:
    """Conservatively reject routes unless their segments prove disjoint.

    Custom converters may span segments, so only whole-route literal boundaries
    can prove two dynamic routes using them disjoint.
    """

    left_regex, left_format, left_convertors = _compile_route(left)
    right_regex, right_format, right_convertors = _compile_route(right)
    if not left_convertors:
        return right_regex.fullmatch(left) is not None
    if not right_convertors:
        return left_regex.fullmatch(right) is not None
    segment_convertors = (StringConvertor, IntegerConvertor, FloatConvertor, UUIDConvertor)
    if any(
        type(convertor) not in segment_convertors
        for convertor in (*left_convertors.values(), *right_convertors.values())
    ):
        return _route_boundaries_overlap(left_format, right_format)
    left_segments, right_segments = left.split("/"), right.split("/")
    return len(left_segments) == len(right_segments) and all(
        _route_segments_overlap(left_segment, right_segment)
        for left_segment, right_segment in zip(left_segments, right_segments)
    )


def _normalized_route_path(path: str) -> str:
    """Normalize ownership checks without changing the forwarded path."""

    segments = []
    for segment in path.split("/"):
        if not segment or segment == ".":
            continue
        if segment == "..":
            if segments:
                segments.pop()
        else:
            segments.append(segment)
    return "/" + "/".join(segments)


def _route_path_candidates(path: str) -> frozenset[str]:
    """Account for decoded delimiters being reparsed by an injected sender."""

    return frozenset(_normalized_route_path(candidate) for candidate in (path, re.split(r"[?#]", path, maxsplit=1)[0]))


class HttpRouteRejectionKind(str, Enum):
    """Classify proxy-owned route rejections before provider dispatch."""

    METHOD_NOT_ALLOWED = "method_not_allowed"
    NON_CANONICAL_PATH = "non_canonical_path"
    INVALID_REQUEST_PATH = "invalid_request_path"


@dataclass(frozen=True, slots=True)
class HttpRouteRejected:
    """Carry a route rejection and its allowed methods to the response mapping."""

    kind: HttpRouteRejectionKind
    allowed_methods: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class HttpRouteOwnership:
    local_paths: tuple[GuardedOperationPath, ...]
    upstream_paths: tuple[GuardedOperationPath, ...] = ()
    map_upstream_path: Callable[[str, bytes], str] | None = None

    def __post_init__(self) -> None:
        if bool(self.upstream_paths) != (self.map_upstream_path is not None):
            raise ValueError("Upstream ownership requires both path declarations and a path mapping.")

    def intersects_mount(self, mount_path: str) -> bool:
        return any(path.intersects_mount(mount_path) for path in self.local_paths)

    def reject_passthrough(self, method: str, path: str, raw_path: bytes) -> HttpRouteRejected | None:
        candidates = _route_path_candidates(path)
        normalized_path = _normalized_route_path(path)
        guarded_methods = frozenset(
            method
            for operation_path in self.local_paths
            if any(operation_path.matches(candidate) for candidate in candidates)
            for method in operation_path.methods
        )
        passthrough_methods = frozenset(
            method
            for operation_path in self.local_paths
            if operation_path.matches(normalized_path)
            for method in operation_path.passthrough_methods
        )
        if guarded_methods:
            if path == normalized_path and candidates == frozenset({path}) and method in passthrough_methods:
                return None
            if method not in guarded_methods:
                return HttpRouteRejected(
                    HttpRouteRejectionKind.METHOD_NOT_ALLOWED, guarded_methods | passthrough_methods
                )
            return HttpRouteRejected(HttpRouteRejectionKind.NON_CANONICAL_PATH)
        if self.map_upstream_path is None:
            return None
        upstream_candidates = _route_path_candidates(self.map_upstream_path(path, raw_path))
        upstream_methods = frozenset(
            method
            for operation_path in self.upstream_paths
            if any(operation_path.matches(candidate) for candidate in upstream_candidates)
            for method in operation_path.methods | operation_path.passthrough_methods
        )
        if not upstream_methods:
            return None
        if method not in upstream_methods:
            return HttpRouteRejected(HttpRouteRejectionKind.METHOD_NOT_ALLOWED, upstream_methods)
        return HttpRouteRejected(HttpRouteRejectionKind.NON_CANONICAL_PATH)
