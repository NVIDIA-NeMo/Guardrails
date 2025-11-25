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

"""Prompts for the various steps in the interaction."""

import os
from typing import List, Optional, Union

import yaml

from nemoguardrails.llm.types import Task
from nemoguardrails.rails.llm.config import Model, RailsConfig, TaskPrompt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_prompts() -> List[TaskPrompt]:
    """Load the predefined prompts from the `prompts` directory."""

    # List of directory containing prompts
    prompts_dirs = [os.path.join(CURRENT_DIR, "prompts")]

    # Fetch prompt directory from env var this should be either abs path or relative to cwd
    prompts_dir = os.getenv("PROMPTS_DIR", None)
    if prompts_dir and os.path.exists(prompts_dir):
        prompts_dirs.append(prompts_dir)

    prompts = []

    for path in prompts_dirs:
        for root, dirs, files in os.walk(path):
            for filename in files:
                if filename.endswith(".yml") or filename.endswith(".yaml"):
                    with open(os.path.join(root, filename), encoding="utf-8") as prompts_file:
                        prompts.extend(yaml.safe_load(prompts_file.read())["prompts"])

    return [TaskPrompt(**prompt) for prompt in prompts]


_prompts = _load_prompts()


def _get_prompt(task_name: str, model: str, prompting_mode: str, prompts: List) -> TaskPrompt:
    """Return the prompt for the given task.

    We intentionally update the matching model at equal score, to take the last one,
    basically allowing to override a prompt for a specific model.
    """
    matching_prompt = None
    matching_score = 0

    model = model.lower()
    for prompt in prompts:
        if prompt.task != task_name:
            continue

        _score = 0

        # If no model is specified, we are dealing with a general prompt, and it has the
        # lowest score.
        if not prompt.models:
            _score = 0.2
        else:
            for _model in prompt.models:
                # If we have an exact match for the full task_model string (e.g., "engine/provider/model-variant")
                if _model == model:
                    _score = 1
                    break

                # is a provider/base_model pattern matching the model path component of `model` (task_model string).
                parts = model.split("/", 1)
                config_model_path = parts[1] if len(parts) > 1 else parts[0]

                if "/" in _model and config_model_path.startswith(_model):
                    if _model == config_model_path:
                        # _model exactly matches the model path component (e.g., "nvidia/llama-3.1-nemotron-ultra-253b-v1")
                        _score = 0.8
                    else:
                        # _model is a proper prefix (e.g., "nvidia/llama-3.1-nemotron" for "...-ultra-253b-v1")
                        _score = 0.9
                    break

                elif model.startswith(_model + "/"):
                    _score = 0.5
                    break

                elif model.endswith("/" + _model):
                    _score = 0.8
                    break

                elif _model in model:
                    _score = 0.4
                    break

        if prompt.mode != prompting_mode:
            # Penalize matching score for being in an incorrect mode.
            # This way, if a prompt with the correct mode (say "compact") is found, it will be preferred over a prompt with another mode (say "standard").
            if prompt.mode == "standard":
                # why 0.5? why not <0.2? To give preference to matching model or provider over matching mode.
                # This way, standard mode with matching provider at gets a score of 0.5 * 0.5 = 0.25
                # (> 0.2 for a matching mode but without a matching provider or model).
                _score *= 0.5
            else:
                continue  # if it's the mode doesn't match AND it's not standard too, discard this match

        if _score >= matching_score:
            matching_score = _score
            matching_prompt = prompt

    if matching_prompt:
        return matching_prompt

    raise ValueError(f"Could not find prompt for task {task_name} and model {model}")


def get_task_model(config: RailsConfig, task: Union[str, Task]) -> Optional[Model]:
    """Return the model for the given task in the current config."""
    # Fetch current task parameters like name, models to use, and the prompting mode
    task_name = str(task.value) if isinstance(task, Task) else task

    if config.models:
        _models = [model for model in config.models if model.type == task_name]
        if not _models:
            _models = [model for model in config.models if model.type == "main"]

        if _models:
            return _models[0]

    return None


def get_prompt(config: RailsConfig, task: Union[str, Task]) -> TaskPrompt:
    """Return the prompt for the given task."""

    # Fetch current task parameters like name, models to use, and the prompting mode
    task_name = str(task.value) if isinstance(task, Task) else task

    task_model = "unknown"
    _model = get_task_model(config, task)
    if _model:
        task_model = _model.engine
        if _model.model:
            task_model += "/" + _model.model

    task_prompting_mode = "standard"
    if config.prompting_mode:
        # if exists in config, overwrite, else, default to "standard"
        task_prompting_mode = config.prompting_mode

    prompts = _prompts + (config.prompts or [])
    prompt = _get_prompt(task_name, task_model, task_prompting_mode, prompts)

    if prompt:
        return prompt
    else:
        raise ValueError(f"No prompt found for task: {task}")
