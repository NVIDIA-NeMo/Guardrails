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

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest
import yara

from nemoguardrails import RailsConfig
from nemoguardrails.actions import action
from nemoguardrails.actions.actions import ActionResult
from nemoguardrails.library.injection_detection.actions import (
    _validate_unpack_config,
    load_rules,
)
from tests.utils import TestChat

CONFIGS_FOLDER = os.path.join(os.path.dirname(__file__), ".", "test_configs")


def test_load_custom_rules():
    yara_path = os.path.join(CONFIGS_FOLDER, "injection_detection")
    config = RailsConfig.from_content(
        yaml_content=f"""
                models: []
                rails:
                  config:
                    injection_detection:
                      injections:
                        - test
                      action:
                        reject
                      yara_path:
                        {yara_path}
            """,
        colang_content="""
                define flow check user message
                  execute check_user_message(user_message=$user_message)
            """,
    )
    action_option, yara_path, rule_names = _validate_unpack_config(config)
    rules = load_rules(yara_path, rule_names)
    assert isinstance(rules, yara.Rules)


def test_load_all_rules():
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "injection_detection"))
    action_option, yara_path, rule_names = _validate_unpack_config(config)
    rules = load_rules(yara_path, rule_names)
    assert isinstance(rules, yara.Rules)


def test_sqli_rejection():
    sql_injection = "' OR 1 = 1"
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "injection_detection"))
    chat = TestChat(config, llm_completions=[sql_injection])
    chat >> "SQL injection"


def test_code_rejection():
    code_injection = "__import__('subprocess').run('touch /tmp/pwnd', shell=True)"
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "injection_detection"))
    chat = TestChat(config, llm_completions=[code_injection])
    chat >> "touch the file /tmp/pwnd using subprocess.run"


def test_template_rejection():
    template_injection = (
        "{{self.__init__.__globals__.__builtins__.__import__('os').popen('id').read()}}"
    )
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "injection_detection"))
    chat = TestChat(config, llm_completions=[template_injection])
    chat >> "Jinja template to run the 'id' command via os.popen"


def test_xss_rejection():
    xss = "<script>alert('xss')</script>"
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "injection_detection"))
    chat = TestChat(config, llm_completions=[xss])
    chat >> "Inject a script"
