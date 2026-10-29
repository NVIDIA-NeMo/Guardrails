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

"""Check the declared upstream inventory without treating open schemas as coverage."""

from nemoguardrails.server.experimental.provider.contract_export import export_guard_contract
from nemoguardrails.server.experimental.providers.openai.dialects.litellm.responses import LITELLM_RESPONSES_ENDPOINT
from tests.server.experimental.conformance.openai.litellm.support import OPENAPI, response_validator


def test_declared_response_fields_have_guard_policies():
    upstream = OPENAPI["components"]["schemas"]["ResponsesAPIResponse"]
    contract = export_guard_contract(LITELLM_RESPONSES_ENDPOINT)
    assert set(upstream["properties"]) <= set(contract["response"]["properties"])


def test_upstream_spec_does_not_enumerate_the_complete_wire_contract():
    assert OPENAPI["info"]["version"] == "1.106.6"
    chat = OPENAPI["paths"]["/v1/chat/completions"]["post"]["responses"]["200"]
    assert chat["content"]["application/json"]["schema"] == {}
    response = OPENAPI["components"]["schemas"]["ResponsesAPIResponse"]
    assert response["additionalProperties"] is True
    assert {"content_filters", "billing", "frequency_penalty", "presence_penalty", "tool_usage"}.isdisjoint(
        response["properties"]
    )
    stream = OPENAPI["paths"]["/v1/responses"]["post"]["responses"]["200"]["content"]["text/event-stream"]
    assert stream["schema"]["type"] == "string"
    # This deliberately invalid Chat payload passes upstream schema validation.
    response_validator("chat/completions").validate({"unreviewed_content": "text"})
