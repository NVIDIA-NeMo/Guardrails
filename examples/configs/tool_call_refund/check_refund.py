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

"""Check a refund tool's schema and policy without executing the tool."""

import asyncio
import json
from pathlib import Path

from nemoguardrails import Guardrails, RailsConfig
from nemoguardrails.rails.llm.options import RailStatus, RailType

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "parameters": {
                "type": "object",
                "properties": {"amount": {"type": "number", "minimum": 0}},
                "required": ["amount"],
                "additionalProperties": False,
            },
        },
    }
]


async def main() -> None:
    config = RailsConfig.from_path(str(Path(__file__).parent))
    async with Guardrails(config, require_iorails=True) as rails:
        for amount in (50, 5000):
            call = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "issue_refund",
                            "arguments": json.dumps({"amount": amount}),
                        },
                    }
                ],
            }
            result = await rails.check_async([call], rail_types=[RailType.TOOL_CALL], tools=TOOLS)
            print(f"${amount}: {result.status.value}")
            if result.status == RailStatus.PASSED:
                print("Ready for the application to execute")
            else:
                print(f"Rail: {result.rail}")
                print(f"Reason: {result.reason}")


if __name__ == "__main__":
    asyncio.run(main())
