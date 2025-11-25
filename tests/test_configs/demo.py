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

"""Demo script."""

import logging

from nemoguardrails import LLMRails, RailsConfig

logging.basicConfig(level=logging.INFO)


def demo():
    """Quick demo using LLMRails with config from dict."""
    config = RailsConfig.parse_object(
        {
            "models": [{"type": "main", "engine": "openai", "model": "gpt-3.5-turbo-instruct"}],
            "instructions": [
                {
                    "type": "general",
                    "content": "Use a maximum of five words when answering any request.",
                }
            ],
        }
    )

    app = LLMRails(config)

    # Brainstorming for registering additional handlers
    # app.register_handler("before_action", {"action_name": "inform_get"}, handler)
    # app.register_handler("after_action", {"action_name": "inform_get"}, handler)
    # app.register_handler("before_bot_said", fact_checking)

    history = [{"role": "user", "content": "Hello! How are you?"}]
    result = app.generate(messages=history)
    print(result)


if __name__ == "__main__":
    demo()
