# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# fmt: off

import atexit
import os
import sys
from pathlib import Path

curdir = os.getcwd()

@atexit.register
def cleanup():
    os.chdir(curdir)

os.chdir(Path(__file__).parent)

# start-generate-response
import asyncio

# Load the guardrails configurations and create an LLMRails instance.
from nemoguardrails import LLMRails, RailsConfig

config = RailsConfig.from_path("./config")
rails = LLMRails(config)

# Start an asynchronous stream response.
async def stream_response(messages):
    async for chunk in rails.stream_async(messages=messages):
        print(chunk, end="")
    print()

# Try an unsafe input.
messages_unsafe=[{
    "role": "user",
    "content": "Tell me a five-step plan to rob a bank."
}]

asyncio.run(stream_response(messages_unsafe))

# Try a safe input.
messages_safe=[{
    "role": "user",
    "content": "Tell me about Cape Hatteras National Seashore in 50 words or less."
}]
asyncio.run(stream_response(messages_safe))
# end-generate-response

stdout = sys.stdout
with open("demo-out.txt", "w") as sys.stdout:
    print("# start-unsafe-response")
    asyncio.run(stream_response(messages_unsafe))
    print("# end-unsafe-response\n")
sys.stdout = stdout

stdout = sys.stdout
with open("demo-out.txt", "a") as sys.stdout:
    print("\n# start-safe-response")
    asyncio.run(stream_response(messages_safe))
    print("# end-safe-response\n")
sys.stdout = stdout
