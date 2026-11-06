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

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from nemoguardrails.server.experimental._guarded_proxy import create_guarded_http_operation
from nemoguardrails.server.experimental._http_kernel import create_http_proxy_router
from nemoguardrails.server.experimental._provider_transport import ProviderTransport
from nemoguardrails.server.experimental.provider.integration import UpstreamSettings
from nemoguardrails.server.experimental.providers.openai.errors import OPENAI_ERROR_MAPPING
from nemoguardrails.server.experimental.providers.openai.responses.endpoint import RESPONSES_ENDPOINT


@asynccontextmanager
async def responses_proxy(
    api_key,
    checker,
    dispatches,
    *,
    http_transport=None,
    endpoint=RESPONSES_ENDPOINT,
    base_url="https://api.openai.com/v1",
):
    async with httpx.AsyncClient(timeout=60, transport=http_transport) as upstream:
        transport = ProviderTransport(UpstreamSettings(base_url=base_url, mount_path="/v1"), upstream)

        async def buffered(request):
            dispatches.append(request)
            return await transport.dispatch(request, require_identity_response=True)

        async def streaming(request):
            dispatches.append(request)
            return await transport.stream_dispatch(request, require_identity_response=True)

        app = FastAPI()
        app.include_router(
            create_http_proxy_router(
                operations=(
                    create_guarded_http_operation(
                        endpoint,
                        OPENAI_ERROR_MAPPING,
                        stream_dispatch=streaming,
                        max_stream_event_bytes=1024 * 1024,
                        max_pending_stream_bytes=1024 * 1024,
                    ),
                ),
                checker=checker,
                dispatch=buffered,
                render_outcome=OPENAI_ERROR_MAPPING.renderer,
            )
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://recorded-responses.test",
            headers={"authorization": f"Bearer {api_key}"},
        ) as client:
            yield client
