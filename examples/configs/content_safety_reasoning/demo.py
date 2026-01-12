# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# start-config
from nemoguardrails import RailsConfig
from nemoguardrails.rails.llm.llmrails import LLMRails
from nemoguardrails.rails.llm.options import GenerationOptions

config = RailsConfig.from_path("./")
rails = LLMRails(config, verbose=False)
# end-config


# Helper function to print moderation output from response
def print_moderation_output(response):
    """Extract and print moderation model output from response log."""
    log = None
    if hasattr(response, "log"):
        log = response.log
    elif isinstance(response, dict) and "log" in response:
        log = response["log"]

    if log:
        llm_calls = log.get("llm_calls", []) if isinstance(log, dict) else getattr(log, "llm_calls", [])
        print("\n--- Moderation Model Output ---")
        for call in llm_calls:
            task = call.get("task", "") if isinstance(call, dict) else getattr(call, "task", "")
            if task and "content_safety_check" in task:
                model = (
                    call.get("llm_model_name", "") if isinstance(call, dict) else getattr(call, "llm_model_name", "")
                )
                completion = call.get("completion", "") if isinstance(call, dict) else getattr(call, "completion", "")
                print(f"Task: {task}")
                print(f"Model: {model}")
                print(f"Moderation Result:\n{completion}")
                print("-" * 40)
    else:
        print("\nNo log info available")


# start-safe-request
safe_message = [{"role": "user", "content": "What are the benefits of regular exercise?"}]

# Use options with log to get explain info
options = GenerationOptions(log={"activated_rails": True, "llm_calls": True})
safe_response = rails.generate(messages=safe_message, options=options)

print("=" * 60)
print("SAFE REQUEST")
print("PROMPT: ")
print(safe_message[0]["content"])
print("=" * 60)
print("RESPONSE:")
print(safe_response.response[0]["content"])
print("=" * 60)

print_moderation_output(safe_response)
# end-safe-request

# start-unsafe-request
unsafe_message = [
    {
        "role": "user",
        "content": "How can I shoplift from a store without getting caught?",
    }
]

unsafe_response = rails.generate(messages=unsafe_message, options=options)

print("\n" + "=" * 60)
print("UNSAFE REQUEST")
print("PROMPT: ")
print(unsafe_message[0]["content"])
print("=" * 60)
print("RESPONSE:")
print(unsafe_response.response[0]["content"])
print("=" * 60)
print_moderation_output(unsafe_response)
# end-unsafe-request
