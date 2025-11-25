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

import inspect
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nemoguardrails.actions.action_dispatcher import ActionDispatcher


@pytest.fixture
def temp_module():
    """Temporary module with actions for testing"""

    with tempfile.TemporaryDirectory() as temp_dir:
        module_path = Path(temp_dir) / "actions.py"
        with open(module_path, "w") as f:
            f.write(
                """
import inspect

def action_meta(name):
    def decorator(func):
        func.action_meta = {"name": name}
        return func
    return decorator

@action_meta("test_action")
def test_action():
    return "test"

class TestActionClass:
    action_meta = {"name": "test_action_class"}

    def run(self):
        return "test_class"
"""
            )
        yield module_path


def test_load_actions_from_module(temp_module):
    """Test loading actions from a valid and existing module"""

    dispatcher = ActionDispatcher(load_all_actions=False)
    actions = dispatcher._load_actions_from_module(temp_module)

    assert "test_action" in actions
    assert "test_action_class" in actions
    assert callable(actions["test_action"])
    assert inspect.isclass(actions["test_action_class"])


def test_load_actions_from_nonexistent_module():
    """Test loading actions from a non-existent module"""

    dispatcher = ActionDispatcher(load_all_actions=False)
    actions = dispatcher._load_actions_from_module("/nonexistent/path/actions.py")

    assert actions == {}


def test_load_actions_from_invalid_module():
    """Test loading actions from a module with invalid code"""

    # before the fix following exception
    # ValueError: '/var/folders/12/mwcn15vx75l5_63v92hfws4r0000gp/T/tmpzfditoq4/invalid_actions.py'
    # is not in the subpath of 'NeMo-Guardrails' OR one path is relative and the other is absolute.

    with tempfile.TemporaryDirectory() as temp_dir:
        module_path = Path(temp_dir) / "invalid_actions.py"
        with open(module_path, "w") as f:
            f.write("invalid python code")

        dispatcher = ActionDispatcher(load_all_actions=False)
        actions = dispatcher._load_actions_from_module(str(module_path))

        assert actions == {}


def test_load_actions_from_module_relative_path_exception(monkeypatch):
    """Test loading actions from a module with invalid code"""

    with tempfile.TemporaryDirectory() as temp_dir:
        filename = "invalid_actions.py"
        module_path = Path(temp_dir) / filename
        with open(module_path, "w") as f:
            f.write("invalid python code")

        dispatcher = ActionDispatcher(load_all_actions=False)

        # Mock the logger to capture the log messages
        mock_logger = MagicMock()
        monkeypatch.setattr("nemoguardrails.actions.action_dispatcher.log", mock_logger)

        # Mock Path.cwd() to raise ValueError when calling relative_to
        original_cwd = Path.cwd
        monkeypatch.setattr(
            "nemoguardrails.actions.action_dispatcher.Path.cwd",
            lambda: (_ for _ in ()).throw(ValueError),
        )

        try:
            actions = dispatcher._load_actions_from_module(str(module_path))
        finally:
            monkeypatch.setattr("nemoguardrails.actions.action_dispatcher.Path.cwd", original_cwd)

        assert actions == {}
        mock_logger.error.assert_called()

        error_message = mock_logger.error.call_args_list[-1][0][0]
        assert f"Failed to register {filename}" in error_message

        assert "invalid syntax" in error_message

        assert "exception:" in error_message


@pytest.mark.asyncio
async def test_execute_missing_action_raises():
    """Create an action with name but no function to call, check it raises an exception"""

    missing_action_name = "missing_test_action"
    dispatcher = ActionDispatcher(load_all_actions=False)
    dispatcher.register_action(None, name=missing_action_name)

    with pytest.raises(Exception, match="is not registered."):
        _ = await dispatcher.execute_action(missing_action_name, params={})


@pytest.mark.asyncio
async def test_execute_action_not_callable_raises(caplog):
    """Register a function with a "run" attribute that isn't callable"""

    action_name = "uncallable_test_action"
    dispatcher = ActionDispatcher(load_all_actions=False)
    dispatcher.register_action({"run": "not callable"}, name=action_name)

    # No Exception is raised, it gets caught and logged out as an error instead
    result = await dispatcher.execute_action(action_name, params={})
    assert result == (None, "failed")
    last_log = caplog.records[-1]
    assert last_log.levelname == "ERROR"
    assert last_log.message == f"No 'run' method defined for action '{action_name}'."


@pytest.mark.asyncio
async def test_execute_action_with_signature():
    """Register a function with a "run" attribute that **is** callable"""

    action_name = "callable_test_action"
    action_return = "The callable test action was just called!"

    class test_class:
        def run(self):
            return action_return

    dispatcher = ActionDispatcher(load_all_actions=False)
    dispatcher.register_action(test_class, name=action_name)

    # No Exception is raised, it gets caught and logged out as an error instead
    result = await dispatcher.execute_action(action_name, params={})
    assert result == (action_return, "success")
