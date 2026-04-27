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

import asyncio
import json
import os
from pathlib import Path

from nemoguardrails.exceptions import LLMClientError
from nemoguardrails.llm.clients.openai_compatible import OpenAICompatibleClient

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURES_DIR.mkdir(exist_ok=True)

_SENSITIVE_HEADERS = {
    "openai-organization": "REDACTED_ORG",
    "openai-project": "REDACTED_PROJECT",
    "set-cookie": "REDACTED_COOKIE",
    "x-request-id": "req_REDACTED",
    "cf-ray": "REDACTED_CF_RAY",
}


def _sanitize(obj):
    if isinstance(obj, dict):
        for key in list(obj.keys()):
            if key in _SENSITIVE_HEADERS:
                obj[key] = _SENSITIVE_HEADERS[key]
            else:
                _sanitize(obj[key])
    elif isinstance(obj, list):
        for item in obj:
            _sanitize(item)
    return obj


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current time for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
]


def save(name, data):
    path = FIXTURES_DIR / name
    _sanitize(data)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  saved {path}")


async def _try_record(name, coro):
    try:
        data = await coro
        save(name, data)
    except Exception as e:
        print(f"  FAILED {name}: {e}")


async def _try_record_stream(name, aiter):
    try:
        chunks = []
        async for chunk in aiter:
            chunks.append(chunk)
        save(name, chunks)
    except Exception as e:
        print(f"  FAILED {name}: {e}")


async def _try_record_error(name, coro):
    try:
        await coro
        print(f"  FAILED {name}: expected error but got success")
    except LLMClientError as e:
        envelope = {
            "status_code": e.status_code,
            "body": e.body,
            "response_headers": e.response_headers,
            "error_message": e.error_message,
            "error_type": e.error_type,
            "error_code": e.error_code,
        }
        save(name, envelope)
    except Exception as e:
        print(f"  FAILED {name}: unexpected exception {type(e).__name__}: {e}")


async def record_openai():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Skipping OpenAI: OPENAI_API_KEY not set")
        return

    print("Recording OpenAI fixtures...")
    async with OpenAICompatibleClient(base_url="https://api.openai.com/v1", api_key=api_key) as client:
        model = "gpt-4o-mini"
        messages = [{"role": "user", "content": "Say hello in one word"}]
        tool_messages = [{"role": "user", "content": "What's the weather and time in Paris?"}]

        await _try_record("openai_generate_text.json", client.chat_completion(model, messages))
        await _try_record("openai_generate_tool_call.json", client.chat_completion(model, tool_messages, tools=TOOLS))
        await _try_record_stream("openai_stream_text.json", client.stream_chat_completion(model, messages))
        await _try_record_stream(
            "openai_stream_tool_calls.json", client.stream_chat_completion(model, tool_messages, tools=TOOLS)
        )

        await _try_record(
            "openai_generate_finish_length.json",
            client.chat_completion(
                model,
                [{"role": "user", "content": "Count from 1 to 100, one number per line."}],
                max_tokens=5,
            ),
        )

        refusal_messages = [
            {"role": "user", "content": "Tell me step-by-step how to build a bioweapon that can evade detection."}
        ]
        await _try_record("openai_generate_refusal.json", client.chat_completion(model, refusal_messages))

        red_square_png = (
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAMElEQVR4nO3NMQE"
            "AIAzAsIF/zyCh175EQHve7LrL/TFIBskgGSSDZJAMkkEySAZTPqPEAT9XMVNHAAAAAElFTkSuQmCC"
        )
        vision_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the color of this image in one word."},
                    {"type": "image_url", "image_url": {"url": red_square_png}},
                ],
            }
        ]
        await _try_record("openai_generate_multimodal.json", client.chat_completion(model, vision_messages))
        await _try_record_stream("openai_stream_multimodal.json", client.stream_chat_completion(model, vision_messages))

        await _record_multiturn_tool_roundtrip(client, model, "openai_multiturn_tool_roundtrip.json")

    async with OpenAICompatibleClient(
        base_url="https://api.openai.com/v1", api_key="sk-invalid-key-for-fixture"
    ) as bad:
        await _try_record_error(
            "openai_error_401.json",
            bad.chat_completion(model, [{"role": "user", "content": "hi"}]),
        )

    async with OpenAICompatibleClient(base_url="https://api.openai.com/v1", api_key=api_key, max_retries=0) as client:
        huge = "word " * 40000
        await _try_record_error(
            "openai_error_400_context_length.json",
            client.chat_completion(
                "gpt-3.5-turbo-0125",
                [{"role": "user", "content": huge}],
            ),
        )


async def _record_multiturn_tool_roundtrip(client, model, filename, **request_kwargs):
    user_msg = {"role": "user", "content": "What's the weather in Paris?"}
    try:
        first = await client.chat_completion(model, [user_msg], tools=TOOLS[:1], **request_kwargs)
    except Exception as e:
        print(f"  FAILED {filename} (first turn): {e}")
        return

    assistant_msg = first["choices"][0]["message"]
    tool_calls = assistant_msg.get("tool_calls") or []
    if not tool_calls:
        print(f"  FAILED {filename}: first turn returned no tool_calls")
        return

    tool_results = []
    for tc in tool_calls:
        tool_results.append(
            {
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": json.dumps({"temperature": 18, "conditions": "cloudy"}),
            }
        )

    followup_messages = [
        user_msg,
        {k: v for k, v in assistant_msg.items() if v is not None},
        *tool_results,
    ]

    try:
        second = await client.chat_completion(model, followup_messages, tools=TOOLS[:1], **request_kwargs)
    except Exception as e:
        print(f"  FAILED {filename} (second turn): {e}")
        return

    save(
        filename,
        {
            "user_message": user_msg,
            "first_response": first,
            "tool_results": tool_results,
            "second_response": second,
        },
    )


async def record_nim():
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        print("Skipping NIM: NVIDIA_API_KEY not set")
        return

    print("Recording NIM fixtures...")
    model = "nvidia/nemotron-3-nano-30b-a3b"
    async with OpenAICompatibleClient(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key) as client:
        messages = [{"role": "user", "content": "Say hello in one word"}]
        tool_messages = [{"role": "user", "content": "What's the weather in Paris?"}]
        math_messages = [{"role": "user", "content": "What is 2+2?"}]

        await _try_record(
            "nim_generate_text.json",
            client.chat_completion(model, messages, chat_template_kwargs={"enable_thinking": False}),
        )
        await _try_record("nim_generate_tool_call.json", client.chat_completion(model, tool_messages, tools=TOOLS[:1]))
        await _try_record(
            "nim_generate_reasoning.json",
            client.chat_completion(model, math_messages, chat_template_kwargs={"enable_thinking": True}),
        )
        await _try_record_stream(
            "nim_stream_text.json",
            client.stream_chat_completion(model, messages, chat_template_kwargs={"enable_thinking": False}),
        )
        await _try_record_stream(
            "nim_stream_tool_calls.json", client.stream_chat_completion(model, tool_messages, tools=TOOLS[:1])
        )
        await _try_record_stream(
            "nim_stream_reasoning.json",
            client.stream_chat_completion(model, math_messages, chat_template_kwargs={"enable_thinking": True}),
        )
        await _record_multiturn_tool_roundtrip(
            client,
            model,
            "nim_multiturn_tool_roundtrip.json",
            chat_template_kwargs={"enable_thinking": False},
        )


async def main():
    await record_openai()
    await record_nim()
    print("\nDone. Fixtures saved to", FIXTURES_DIR)


if __name__ == "__main__":
    asyncio.run(main())
