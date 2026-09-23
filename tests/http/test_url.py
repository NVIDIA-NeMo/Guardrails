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

import pytest

from nemoguardrails.http._url import sanitize_url


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "https://user:password@example.com:8443/check?api_key=secret#fragment",
            "https://example.com:8443/check",
        ),
        (
            "https://user:password@[2001:db8::1]:8443/check?api_key=secret#fragment",
            "https://[2001:db8::1]:8443/check",
        ),
        (
            "https://user:password@/check?api_key=secret#fragment",
            "https:///check",
        ),
        (
            "//user:password@example.com/check?api_key=secret#fragment",
            "//example.com/check",
        ),
        (
            "https://example.com:not-a-port/check?api_key=secret#fragment",
            "https://example.com/check",
        ),
        (
            "https://example.com:99999/check?api_key=secret#fragment",
            "https://example.com/check",
        ),
        (
            "/check?api_key=secret#fragment",
            "/check",
        ),
        (
            "example.com/check?api_key=secret#fragment",
            "example.com/check",
        ),
        (
            "?api_key=secret#fragment",
            "",
        ),
        (
            "",
            "",
        ),
    ],
)
def test_sanitize_url(url: str, expected: str):
    assert sanitize_url(url) == expected
