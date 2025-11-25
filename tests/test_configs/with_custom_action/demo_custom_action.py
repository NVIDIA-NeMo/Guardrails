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

COLANG_CONFIG = """
define user ask service status
  "what is the status of my service"
  "is the service up?"
  "is the service down?"

define flow
  user ask service status
  $status = execute check_service_status
  bot inform service status
"""

YAML_CONFIG = """
models:
  - type: main
    engine: openai
    model: gpt-3.5-turbo-instruct
"""


async def check_service_status():
    return "online"


def demo():
    """Quick demo using LLMRails with config from dict."""
    config = RailsConfig.from_content(COLANG_CONFIG, YAML_CONFIG)

    app = LLMRails(config)
    app.register_action(check_service_status)

    history = [{"role": "user", "content": "Tell me if the service is up"}]
    result = app.generate(messages=history)
    print(result)


if __name__ == "__main__":
    demo()
