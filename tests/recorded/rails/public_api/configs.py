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

from __future__ import annotations

from pathlib import Path

from tests.recorded.rails_config import RailsConfigSource

CONFIGS_DIR = Path(__file__).parent / "configs"
OPENAI_MODEL = "gpt-5.4-nano"
NIM_MODEL = "nvidia/nemotron-3-nano-30b-a3b"
OPENAI_INVALID_MODEL = "gpt-nonexistent-test-model"

OPENAI_BASELINE_CONFIG = RailsConfigSource.from_content(
    name="openai_baseline",
    yaml_content=f"""
    models:
      - type: main
        engine: openai
        model: {OPENAI_MODEL}
        parameters:
          max_retries: 0
    passthrough: true
    """,
)

OPENAI_INVALID_MODEL_CONFIG = RailsConfigSource.from_content(
    name="openai_invalid_model",
    yaml_content=f"""
    models:
      - type: main
        engine: openai
        model: {OPENAI_INVALID_MODEL}
        parameters:
          max_retries: 0
    passthrough: true
    """,
)

NIM_BASELINE_CONFIG = RailsConfigSource.from_path(CONFIGS_DIR, "nim_baseline")
NEMOGUARDS_FULL_CONFIG = RailsConfigSource.from_path(CONFIGS_DIR, "nemoguards_full")
DIALOG_CONFIG = RailsConfigSource.from_path(CONFIGS_DIR, "dialog")
SINGLE_CALL_CONFIG = RailsConfigSource.from_path(CONFIGS_DIR, "single_call")
TASK_MODELS_CONFIG = RailsConfigSource.from_path(CONFIGS_DIR, "task_models")
STREAMING_OUTPUT_RAILS_CONFIG = RailsConfigSource.from_path(CONFIGS_DIR, "streaming_output_rails")
PARALLEL_RAILS_CONFIG = RailsConfigSource.from_path(CONFIGS_DIR, "parallel_rails")

INPUT_RAILS_CONFIG = RailsConfigSource.from_content(
    name="input_rails",
    yaml_content="""
    rails:
      input:
        flows:
          - input rail
    """,
    colang_content="""
    define flow input rail
      if $user_message == "block input"
        bot refuse to respond
        stop
      else if $user_message == "modify input"
        $user_message = "modified input"
    """,
)

OUTPUT_RAILS_CONFIG = RailsConfigSource.from_content(
    name="output_rails",
    yaml_content=f"""
    models:
      - type: main
        engine: openai
        model: {OPENAI_MODEL}
        parameters:
          max_retries: 0
    passthrough: true
    rails:
      output:
        flows:
          - output rail
    """,
    colang_content="""
    define flow output rail
      if $bot_message == "block output"
        bot refuse to respond
        stop
      else if $bot_message == "modify output"
        $bot_message = "modified output"
    """,
)

INPUT_OUTPUT_RAILS_CONFIG = RailsConfigSource.from_content(
    name="input_output_rails",
    yaml_content="""
    rails:
      input:
        flows:
          - input rail
      output:
        flows:
          - output rail
    """,
    colang_content="""
    define flow input rail
      if $user_message == "block input"
        bot refuse to respond
        stop
      else if $user_message == "modify input"
        $user_message = "modified input"

    define flow output rail
      if $bot_message == "block output"
        bot refuse to respond
        stop
      else if $bot_message == "modify output"
        $bot_message = "modified output"
    """,
)

STREAMING_DISABLED_CONFIG = RailsConfigSource.from_content(
    name="streaming_disabled",
    yaml_content="""
    rails:
      output:
        flows:
          - output rail
    streaming: true
    """,
    colang_content="""
    define flow output rail
      if $bot_message == "BLOCK"
        bot refuse to respond
        stop
    """,
)
