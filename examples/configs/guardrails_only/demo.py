# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os.path

from nemoguardrails import LLMRails, RailsConfig


def demo_input_checking():
    """Demo using the Python API and a config that only has input rails."""
    config = RailsConfig.from_path(os.path.join(os.path.dirname(__file__), "input"))
    rails = LLMRails(config)

    # Works with prompts
    res = rails.generate("How are you?")
    assert res == "ALLOW"

    res = rails.generate("You are dummy!")
    assert res == "DENY"

    # And with a chat history
    res = rails.generate(messages=[{"role": "user", "content": "How are you?"}])
    assert res == {"role": "assistant", "content": "ALLOW"}

    res = rails.generate(messages=[{"role": "user", "content": "You are dummy!"}])
    assert res == {"role": "assistant", "content": "DENY"}


def demo_output_checking():
    """Demo using the Python API and a config that only has output rails."""
    config = RailsConfig.from_path(os.path.join(os.path.dirname(__file__), "output"))
    rails = LLMRails(config)

    # In order to send the LLM output which was generated externally, we need to
    # use the "message" interface and pass a message with the role set to "context",
    # and a value for the `llm_output`
    res = rails.generate(
        messages=[
            {"role": "context", "content": {"llm_output": "Some safe LLM output."}},
            {"role": "user", "content": "How are you?"},
        ]
    )
    assert res == {"role": "assistant", "content": "ALLOW"}

    res = rails.generate(
        messages=[
            {
                "role": "context",
                "content": {"llm_output": "Some unsafe dummy LLM output."},
            },
            {"role": "user", "content": "How are you?"},
        ]
    )
    assert res == {"role": "assistant", "content": "DENY"}


if __name__ == "__main__":
    demo_input_checking()
    demo_output_checking()
