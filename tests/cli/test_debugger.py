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

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from nemoguardrails.cli import debugger
from nemoguardrails.cli.chat import ChatState
from nemoguardrails.colang.v2_x.runtime.flows import FlowConfig, FlowState, State

runner = CliRunner()


class TestDebuggerSetup:
    def test_set_chat_state(self):
        chat_state = MagicMock()
        debugger.set_chat_state(chat_state)
        assert debugger.chat_state == chat_state

    def test_set_runtime(self):
        runtime = MagicMock()
        debugger.set_runtime(runtime)
        assert debugger.runtime == runtime

    def test_set_output_state(self):
        state = MagicMock()
        debugger.set_output_state(state)
        assert debugger.state == state


class TestDebuggerCommands:
    def setup_method(self):
        self.chat_state = ChatState()
        self.state = MagicMock(spec=State)
        self.runtime = MagicMock()
        debugger.chat_state = self.chat_state
        debugger.state = self.state
        debugger.runtime = self.runtime

    def test_restart_command(self):
        self.chat_state.state = MagicMock()
        # FIXME:   input_events: List[dict] = field(default_factory=list)
        self.chat_state.input_events = ["event1", "event2"]
        self.chat_state.first_time = False

        result = runner.invoke(debugger.app, ["restart"])

        assert result.exit_code == 0
        assert self.chat_state.state is None
        assert self.chat_state.input_events == []
        assert self.chat_state.first_time is True

    def test_pause_command(self):
        self.chat_state.paused = False

        result = runner.invoke(debugger.app, ["pause"])

        assert result.exit_code == 0
        assert self.chat_state.paused is True

    def test_resume_command(self):
        self.chat_state.paused = True

        result = runner.invoke(debugger.app, ["resume"])

        assert result.exit_code == 0
        assert self.chat_state.paused is False


class TestFlowCommand:
    def setup_method(self):
        self.state = MagicMock(spec=State)
        self.chat_state = ChatState()
        debugger.state = self.state
        debugger.chat_state = self.chat_state

    @patch("nemoguardrails.cli.debugger.console")
    def test_flow_command_with_flow_config(self, mock_console):
        flow_config = MagicMock(spec=FlowConfig)
        self.state.flow_configs = {"test_flow": flow_config}
        self.state.flow_states = {}

        result = runner.invoke(debugger.app, ["flow", "test_flow"])

        assert result.exit_code == 0
        mock_console.print.assert_called_once_with(flow_config)

    @patch("nemoguardrails.cli.debugger.console")
    def test_flow_command_with_flow_instance(self, mock_console):
        flow_instance = MagicMock(spec=FlowState)
        flow_instance.__dict__ = {"uid": "flow_uid_123", "status": "active"}
        self.state.flow_configs = {}
        self.state.flow_states = {"flow_uid_123": flow_instance}

        result = runner.invoke(debugger.app, ["flow", "uid_123"])

        assert result.exit_code == 0
        mock_console.print.assert_called_once_with(flow_instance.__dict__)

    @patch("nemoguardrails.cli.debugger.console")
    def test_flow_command_not_found(self, mock_console):
        self.state.flow_configs = {}
        self.state.flow_states = {}

        result = runner.invoke(debugger.app, ["flow", "nonexistent"])

        assert result.exit_code == 0
        mock_console.print.assert_called_once_with("Flow 'nonexistent' does not exist.")


class TestFlowsCommand:
    def setup_method(self):
        self.state = MagicMock(spec=State)
        self.chat_state = ChatState()
        debugger.state = self.state
        debugger.chat_state = self.chat_state

    @patch("nemoguardrails.cli.debugger.console")
    @patch("nemoguardrails.cli.debugger.is_active_flow")
    def test_flows_command_active_only(self, mock_is_active, mock_console):
        flow_config = MagicMock(spec=FlowConfig)
        flow_config.loop_priority = 1
        flow_config.loop_type = MagicMock(value="default")
        flow_config.source_file = "/path/to/nemoguardrails/flow.py"

        flow_instance = MagicMock(spec=FlowState)
        flow_instance.uid = "(flow)12345"

        self.state.flow_configs = {"test_flow": flow_config}
        self.state.flow_id_states = {"test_flow": [flow_instance]}

        mock_is_active.return_value = True

        result = runner.invoke(debugger.app, ["flows"])

        assert result.exit_code == 0
        mock_console.print.assert_called()

    @patch("nemoguardrails.cli.debugger.console")
    def test_flows_command_all_flows(self, mock_console):
        flow_config = MagicMock(spec=FlowConfig)
        flow_config.loop_priority = 1
        flow_config.loop_type = MagicMock(value="default")
        flow_config.source_file = "/path/to/flow.py"

        self.state.flow_configs = {"test_flow": flow_config}
        self.state.flow_id_states = {}

        result = runner.invoke(debugger.app, ["flows", "--all"])

        assert result.exit_code == 0
        mock_console.print.assert_called()

    @patch("nemoguardrails.cli.debugger.console")
    def test_flows_command_order_by_name(self, mock_console):
        flow_config_a = MagicMock(spec=FlowConfig)
        flow_config_a.loop_priority = 2
        flow_config_a.loop_type = MagicMock(value="default")
        flow_config_a.source_file = None

        flow_config_b = MagicMock(spec=FlowConfig)
        flow_config_b.loop_priority = 1
        flow_config_b.loop_type = MagicMock(value="default")
        flow_config_b.source_file = None

        self.state.flow_configs = {"flow_b": flow_config_b, "flow_a": flow_config_a}
        self.state.flow_id_states = {}

        result = runner.invoke(debugger.app, ["flows", "--all", "--order-by-name"])

        assert result.exit_code == 0
        mock_console.print.assert_called()


class TestTreeCommand:
    def setup_method(self):
        self.state = MagicMock(spec=State)
        self.runtime = MagicMock()
        debugger.state = self.state
        debugger.runtime = self.runtime

    @patch("nemoguardrails.cli.debugger.Tree")
    @patch("nemoguardrails.cli.debugger.console")
    @patch("nemoguardrails.cli.debugger.is_active_flow")
    def test_tree_command(self, mock_is_active, mock_console, mock_tree):
        # create a main flow (required by tree command)
        main_flow = MagicMock(spec=FlowState)
        main_flow.uid = "main_flow"
        main_flow.flow_id = "main"
        main_flow.head = MagicMock()
        main_flow.head.elements = []
        main_flow.child_flow_uids = []

        # create mock flow configs with proper elements
        main_config = MagicMock()
        main_config.elements = []

        self.state.flow_states = {"main_flow": main_flow}
        self.state.flow_id_states = {"main": [main_flow]}
        self.state.flow_configs = {"main": main_config}
        mock_is_active.return_value = True

        # create a mock tree object
        mock_tree_instance = MagicMock()
        mock_tree.return_value = mock_tree_instance

        result = runner.invoke(debugger.app, ["tree"])

        assert result.exit_code == 0
        mock_tree.assert_called_with("main")
