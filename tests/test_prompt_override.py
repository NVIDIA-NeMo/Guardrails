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

import os

from nemoguardrails import RailsConfig
from nemoguardrails.llm import prompts as prompts_module
from nemoguardrails.llm.prompts import get_prompt
from nemoguardrails.llm.types import Task

CONFIGS_FOLDER = os.path.join(os.path.dirname(__file__), ".", "test_configs")


def test_custom_llm_registration():
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "with_prompt_override"))

    prompt = get_prompt(config, Task.GENERATE_USER_INTENT)

    assert prompt.content == "<<This is a placeholder for a custom prompt for generating the user intent>>"


def test_load_prompts_sorts_files_for_deterministic_overrides(tmp_path, monkeypatch):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "z.yml").write_text("prompts:\n- task: generate_user_intent\n  content: z\n", encoding="utf-8")
    (prompts_dir / "a.yml").write_text("prompts:\n- task: generate_user_intent\n  content: a\n", encoding="utf-8")

    def walk(_path):
        yield str(prompts_dir), [], ["z.yml", "a.yml"]

    monkeypatch.setattr(prompts_module, "CURRENT_DIR", str(tmp_path))
    monkeypatch.setattr(prompts_module.os, "walk", walk)
    monkeypatch.delenv("PROMPTS_DIR", raising=False)

    assert [prompt.content for prompt in prompts_module._load_prompts()] == ["a", "z"]
