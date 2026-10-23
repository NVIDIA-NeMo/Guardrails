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
DIALOG_GENERATION_CONFIG = RailsConfigSource.from_path(CONFIGS_DIR, "dialog_generation")
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

TOOL_ROUND_TRIP_RAILS_CONFIG = RailsConfigSource.from_content(
    name="tool_round_trip_rails",
    yaml_content=f"""
    models:
      - type: main
        engine: openai
        model: {OPENAI_MODEL}
        parameters:
          max_retries: 0
    passthrough: false
    rails:
      input:
        flows: [rewrite tool question]
      tool_result:
        flows: [sanitize weather]
      output:
        flows: [mark checked answer]
      dialog:
        single_call:
          enabled: true
    prompts:
      - task: general
        messages:
          - type: system
            content: "Call each requested tool once. After receiving tool results, quote both results exactly."
          - "{{{{ history | to_chat_messages }}}}"
    """,
    colang_content="""
    define flow rewrite tool question
      if $user_message == "PRIVATE_USER"
        $user_message = "Call get_paris and get_berlin, then quote both weather results exactly."

    define flow sanitize weather
      if $tool_name == "get_paris"
        $tool_message = "PARIS-4827: sunny"
      if $tool_name == "get_berlin"
        $tool_message = "BERLIN-6912: rainy"

    define flow mark checked answer
      $bot_message = "Checked: " + $bot_message
    """,
)

TOOL_CALL_BLOCK_CONFIG = RailsConfigSource.from_content(
    name="tool_call_block",
    yaml_content=f"""
    models:
      - type: main
        engine: openai
        model: {OPENAI_MODEL}
        parameters:
          max_retries: 0
    passthrough: false
    rails:
      tool_call:
        flows: [block tool call]
    """,
    colang_content="""
    define flow block tool call
      bot refuse tool call
      abort

    define bot refuse tool call
      "Tool call rejected."
    """,
)

PASSTHROUGH_SANITIZED_HISTORY_CONFIG = RailsConfigSource.from_content(
    name="passthrough_sanitized_history",
    yaml_content=f"""
    models:
      - type: main
        engine: openai
        model: {OPENAI_MODEL}
        parameters:
          max_retries: 0
    passthrough: true
    rails:
      input:
        flows: [sanitize history]
    """,
    colang_content="""
    define flow sanitize history
      if $user_message == "PRIVATE_USER"
        $user_message = "The fictional weather station is named MAPLE-4827. Reply exactly: noted"
    """,
)
