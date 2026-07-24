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

"""Transport-neutral asynchronous HTTP clients for NeMo Guardrails integrations."""

from nemoguardrails.http.client import ClosableHTTPClient, HTTPClient
from nemoguardrails.http.errors import (
    HTTPClientError,
    HTTPConnectionError,
    HTTPResponseDecodeError,
    HTTPStatusError,
    HTTPTimeoutError,
)
from nemoguardrails.http.instrumented import InstrumentedHTTPClient, instrument_http_client
from nemoguardrails.http.request import http_call
from nemoguardrails.http.retry import RetryingHTTPClient, RetryPolicy
from nemoguardrails.http.runtime import create_http_client
from nemoguardrails.http.transport import HttpxHTTPClient
from nemoguardrails.http.types import HTTPRequest, HTTPResponse

__all__ = [
    "ClosableHTTPClient",
    "HTTPClient",
    "HTTPClientError",
    "HTTPConnectionError",
    "HTTPRequest",
    "HTTPResponse",
    "HTTPResponseDecodeError",
    "HTTPStatusError",
    "HTTPTimeoutError",
    "HttpxHTTPClient",
    "InstrumentedHTTPClient",
    "RetryPolicy",
    "RetryingHTTPClient",
    "create_http_client",
    "http_call",
    "instrument_http_client",
]
