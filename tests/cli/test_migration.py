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

import os
import tempfile
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from nemoguardrails.cli.migration import (
    _confirm_and_tag_replace,
    _get_co_files_to_process,
    _get_config_files_to_process,
    _globalize_variable_assignment,
    _is_anonymous_flow,
    _is_flow,
    _process_co_files,
    _process_config_files,
    _revise_anonymous_flow,
    _write_to_file,
    _write_transformed_content_and_rename_original,
    convert_colang_1_syntax,
    convert_colang_2alpha_syntax,
    migrate,
)


class TestColang2AlphaSyntaxConversion:
    def test_orwhen_replacement(self):
        input_lines = ["orwhen condition met"]
        expected_output = ["or when condition met"]
        assert convert_colang_2alpha_syntax(input_lines) == expected_output

    def test_flow_start_uid_replacement(self):
        input_lines = ["flow_start_uid: 12345"]
        expected_output = ["flow_instance_uid: 12345"]
        assert convert_colang_2alpha_syntax(input_lines) == expected_output

    def test_regex_replacement_1(self):
        input_lines = ['r"(?i).*({{$text}})((\s*\w+\s*){0,2})\W*$"']
        expected_output = ['regex("((?i).*({{$text}})((\\s*\\w+\\s*){0,2})\\W*$)")']
        assert convert_colang_2alpha_syntax(input_lines) == expected_output

    def test_regex_replacement_2(self):
        input_lines = ["r'(?i).*({{$text}})((\s*\w+\s*){0,2})\W*$'"]
        expected_output = ["regex('((?i).*({{$text}})((\\s*\\w+\\s*){0,2})\\W*$)')"]
        assert convert_colang_2alpha_syntax(input_lines) == expected_output

    def test_curly_braces_replacement(self):
        input_lines = ['"{{variable}}"']
        expected_output = ['"{variable}"']
        assert convert_colang_2alpha_syntax(input_lines) == expected_output

    def test_findall_replacement(self):
        input_lines = ["findall matches"]
        expected_output = ["find_all matches"]
        assert convert_colang_2alpha_syntax(input_lines) == expected_output

    def test_triple_quotes_replacement_1(self):
        input_lines = ['$ """some text"""']
        expected_output = ['$ ..."some text"']
        assert convert_colang_2alpha_syntax(input_lines) == expected_output

    def test_triple_quotes_replacement_2(self):
        input_lines = ["$ '''some text'''"]
        expected_output = ["$ ...'some text'"]
        assert convert_colang_2alpha_syntax(input_lines) == expected_output

    def test_specific_phrases_replacement(self):
        input_lines = [
            "catch colang errors",
            "catch undefined flows",
            "catch unexpected user utterance",
            "track bot talking state",
            "track user talking state",
            "track user utterance state",
            "poll llm request response",
            "trigger user intent for unhandled user utterance",
            "generate then continue interaction",
            "track unhandled user intent state",
            "respond to unhandled user intent",
        ]
        expected_output = [
            "catch colang errors",
            "notification of undefined flow start",
            "notification of unexpected user utterance",
            "tracking bot talking state",
            "tracking user talking state",
            "tracking user talking state",
            "polling llm request response",
            "generating user intent for unhandled user utterance",
            "llm continue interaction",
            "tracking unhandled user intent state",
            "continuation on unhandled user intent",
        ]
        assert convert_colang_2alpha_syntax(input_lines) == expected_output

    def test_meta_decorator_replacement(self):
        input_lines = [
            "flow some_flow:",
            "# meta: loop_id=123",
            "# meta: example meta",
            "some action",
        ]
        expected_output = [
            '@loop("123")',
            "@meta(example_meta=True)",
            "flow some_flow:",
            "some action",
        ]
        assert convert_colang_2alpha_syntax(input_lines) == expected_output

    def test_convert_flow_examples(self):
        input_1 = """
        flow bot inform something like issue
            # meta: bot intent
            (bot inform "ABC"
                or bot inform "DEFG"
                or bot inform "HJKL")
                and (bot gesture "abc def" or bot gesture "hij kl")
        """
        input_lines = textwrap.dedent(input_1).strip().split("\n")

        output_1 = """
        @meta(bot_intent=True)
        flow bot inform something like issue
            (bot inform "ABC"
                or bot inform "DEFG"
                or bot inform "HJKL")
                and (bot gesture "abc def" or bot gesture "hij kl")
        """
        output_lines = textwrap.dedent(output_1).strip().split("\n")

        assert convert_colang_2alpha_syntax(input_lines) == output_lines


class TestColang1SyntaxConversion:
    def test_define_flow_conversion(self):
        input_lines = ["define flow express greeting"]
        expected_output = ["flow express greeting"]
        assert convert_colang_1_syntax(input_lines) == expected_output

    def test_define_subflow_conversion(self):
        input_lines = ["define subflow my_subflow"]
        expected_output = ["flow my_subflow"]
        assert convert_colang_1_syntax(input_lines) == expected_output

    def test_execute_to_await_and_pascal_case_action(self):
        input_lines = ["execute some_action"]
        expected_output = ["await SomeAction"]
        assert convert_colang_1_syntax(input_lines) == expected_output

    def test_stop_to_abort(self):
        input_lines = ["stop"]
        expected_output = ["abort"]
        assert convert_colang_1_syntax(input_lines) == expected_output

    def test_anonymous_flow_revised(self):
        input_lines = ["flow", "user said hello"]
        # because the flow is anonymous and only 'flow' is given, it will be converted to 'flow said hello' based on the first message
        expected_output = ["flow said hello", "user said hello"]
        output = convert_colang_1_syntax(input_lines)
        # strip newline characters from the strings in the output list
        output = [line.rstrip("\n") for line in output]
        assert output == expected_output

    def test_global_variable_assignment(self):
        input_lines = ["$variable = value"]
        expected_output = ["global $variable\n$variable = value"]
        assert convert_colang_1_syntax(input_lines) == expected_output

    def test_variable_assignment_in_await(self):
        input_lines = ["$result = await some_action"]
        expected_output = ["$result = await SomeAction"]
        assert convert_colang_1_syntax(input_lines) == expected_output

    def test_bot_say_conversion(self):
        input_lines = ["define bot", '"Hello!"', '"How can I help you?"']
        expected_output = [
            "flow bot",
            'bot say "Hello!"',
            'or bot say "How can I help you?"',
        ]
        assert convert_colang_1_syntax(input_lines) == expected_output

    def test_user_said_conversion(self):
        input_lines = ["define user", '"I need assistance."', '"Can you help me?"']
        expected_output = [
            "flow user",
            'user said "I need assistance."',
            'or user said "Can you help me?"',
        ]
        assert convert_colang_1_syntax(input_lines) == expected_output

    def test_create_event_to_send(self):
        input_lines = ["    create event user_asked_question"]
        expected_output = ["    send user_asked_question"]
        assert convert_colang_1_syntax(input_lines) == expected_output

    def test_config_variable_replacement(self):
        input_lines = ["$config.setting = true"]
        expected_output = ["global $system.config.setting\n$system.config.setting = true"]
        assert convert_colang_1_syntax(input_lines) == expected_output

    def test_flow_with_special_characters(self):
        input_lines = ["define flow my-flow's_test"]
        expected_output = ["flow my flow s_test"]
        assert convert_colang_1_syntax(input_lines) == expected_output

    def test_ellipsis_variable_assignment(self):
        input_lines = ["# User's name", "$name = ...", "await greet_user"]
        expected_output = [
            "# User's name",
            "global $name\n$name = ...",
            "await GreetUserAction",
        ]

        expected_output = [
            "# User's name",
            'global $name\n$name = ... "User\'s name"',
            "await GreetUserAction",
        ]
        assert convert_colang_1_syntax(input_lines) == expected_output

    @pytest.mark.skip("not implemented conversion")
    def test_complex_conversion(self):
        # TODO: add bot $response to bot say $response conversion
        input_script = """
        define flow greeting_flow
            when user express greeting
                $response = execute generate_greeting
                bot $response
        """
        expected_output_script = """
        flow greeting_flow
            when user express greeting
                $response = await GenerateGreetingAction
                bot say $response
        """
        input_lines = textwrap.dedent(input_script).strip().split("\n")
        expected_output = textwrap.dedent(expected_output_script).strip().split("\n")

        print(convert_colang_1_syntax(input_lines))
        assert convert_colang_1_syntax(input_lines) == expected_output

    def test_flow_with_execute_and_stop(self):
        input_lines = [
            "define flow sample_flow",
            '    when user "Cancel"',
            "        execute cancel_operation",
            "        stop",
        ]
        expected_output = [
            "flow sample_flow",
            '    when user "Cancel"',
            "        await CancelOperationAction",
            "        abort",
        ]
        assert convert_colang_1_syntax(input_lines) == expected_output

    def test_await_camelcase_conversion(self):
        input_lines = ["await sample_action"]
        expected_output = ["await SampleAction"]
        assert convert_colang_1_syntax(input_lines) == expected_output

    def test_nested_flow_conversion(self):
        input_script = """
        define flow outer_flow
            when condition_met
                define subflow inner_flow
                    execute inner_action
        """
        expected_output_script = """
        flow outer_flow
            when condition_met
                flow inner_flow
                    await InnerAction
        """
        input_lines = textwrap.dedent(input_script).strip().split("\n")
        expected_output = textwrap.dedent(expected_output_script).strip().split("\n")
        assert convert_colang_1_syntax(input_lines) == expected_output


class TestMigrateFunction:
    @patch("nemoguardrails.cli.migration._process_config_files")
    @patch("nemoguardrails.cli.migration._process_co_files")
    @patch("nemoguardrails.cli.migration._get_config_files_to_process")
    @patch("nemoguardrails.cli.migration._get_co_files_to_process")
    @patch("nemoguardrails.cli.migration.console")
    def test_migrate_with_defaults(
        self,
        mock_console,
        mock_get_co_files,
        mock_get_config_files,
        mock_process_co,
        mock_process_config,
    ):
        mock_get_co_files.return_value = ["file1.co", "file2.co"]
        mock_get_config_files.return_value = ["config.yml"]
        mock_process_co.return_value = 2
        mock_process_config.return_value = 1

        migrate("/test/path")

        mock_console.print.assert_any_call(
            "Starting migration for path: /test/path from version 1.0 to latest version."
        )
        mock_process_co.assert_called_once_with(["file1.co", "file2.co"], "1.0", False, True, True)
        mock_process_config.assert_called_once_with(["config.yml"])

    @patch("nemoguardrails.cli.migration._process_config_files")
    @patch("nemoguardrails.cli.migration._process_co_files")
    @patch("nemoguardrails.cli.migration._get_config_files_to_process")
    @patch("nemoguardrails.cli.migration._get_co_files_to_process")
    @patch("nemoguardrails.cli.migration.console")
    def test_migrate_from_2_0_alpha(
        self,
        mock_console,
        mock_get_co_files,
        mock_get_config_files,
        mock_process_co,
        mock_process_config,
    ):
        mock_get_co_files.return_value = []
        mock_get_config_files.return_value = []
        mock_process_co.return_value = 0
        mock_process_config.return_value = 0

        migrate("/test/path", from_version="2.0-alpha", include_main_flow=True)

        mock_process_co.assert_called_once_with([], "2.0-alpha", True, True, True)


class TestGetFilesToProcess:
    def test_get_co_files_from_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "file1.co").touch()
            Path(tmpdir, "file2.co").touch()
            Path(tmpdir, "file3.txt").touch()
            Path(tmpdir, "subdir").mkdir()
            Path(tmpdir, "subdir", "file4.co").touch()

            files = _get_co_files_to_process(tmpdir)

            assert len(files) == 3
            assert all(str(f).endswith(".co") for f in files)

    def test_get_co_files_single_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir, "test.co")
            test_file.touch()

            files = _get_co_files_to_process(str(test_file))

            assert len(files) == 1
            assert str(files[0]) == str(test_file)

    def test_get_config_files_from_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "config.yml").touch()
            Path(tmpdir, "config.yaml").touch()
            Path(tmpdir, "other.txt").touch()

            files = _get_config_files_to_process(tmpdir)

            assert len(files) == 2
            assert all(str(f).endswith((".yml", ".yaml")) for f in files)

    def test_get_config_files_single_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir, "config.yml")
            test_file.touch()

            files = _get_config_files_to_process(tmpdir)

            assert len(files) == 1
            assert str(files[0]) == str(test_file)


class TestWriteFunctions:
    def test_write_to_file_success(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            filename = f.name

        try:
            lines = ["line 1\n", "line 2\n", "line 3\n"]
            result = _write_to_file(filename, lines)

            assert result is True

            with open(filename, "r") as f:
                content = f.readlines()
            assert content == lines
        finally:
            os.unlink(filename)

    @patch("builtins.open", side_effect=IOError("Permission denied"))
    @patch("logging.error")
    def test_write_to_file_failure(self, mock_log_error, mock_open_func):
        result = _write_to_file("/invalid/path/file.txt", ["test"])

        assert result is False
        mock_log_error.assert_called_once()

    def test_write_transformed_content_and_rename_original(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            original_file = Path(tmpdir, "test.co")
            original_file.write_text("original content")

            new_lines = ["new line 1\n", "new line 2\n"]
            result = _write_transformed_content_and_rename_original(str(original_file), new_lines)

            assert result is True
            assert original_file.exists()
            assert original_file.read_text() == "new line 1\nnew line 2\n"

            renamed_file = Path(tmpdir, "test.v1.co")
            assert renamed_file.exists()
            assert renamed_file.read_text() == "original content"


class TestHelperFunctions:
    def test_is_anonymous_flow(self):
        assert _is_anonymous_flow("flow  ") is True
        assert _is_anonymous_flow("flow") is True
        assert _is_anonymous_flow("flow my_flow") is False
        assert _is_anonymous_flow("  flow  ") is False
        assert _is_anonymous_flow("not_flow") is False

    def test_is_flow(self):
        assert _is_flow("flow my_flow") is True
        assert _is_flow("flow test_flow_123") is True
        assert _is_flow("flow") is False
        assert _is_flow("  flow my_flow") is False
        assert _is_flow("not_flow test") is False

    def test_globalize_variable_assignment(self):
        result = _globalize_variable_assignment("$var = value")
        assert "global $var" in result

        result = _globalize_variable_assignment("$result = await action")
        assert "global" not in result

        result = _globalize_variable_assignment("$result = execute action")
        assert "global" not in result

        result = _globalize_variable_assignment("regular = value")
        assert "global regular" in result
        assert "regular = value" in result

    def test_revise_anonymous_flow(self):
        result = _revise_anonymous_flow("flow", "user said hello")
        assert result == "flow said hello"

        result = _revise_anonymous_flow("flow", 'bot say "hi"')
        assert "flow say" in result

        result = _revise_anonymous_flow("flow", "other content")
        assert "flow other content" in result

    def test_confirm_and_tag_replace(self):
        from nemoguardrails.cli import migration

        migration._LIBS_USED.clear()

        _confirm_and_tag_replace("new line", "old line", "core")
        assert "core" in migration._LIBS_USED

        _confirm_and_tag_replace("same line", "same line", "llm")
        assert "llm" not in migration._LIBS_USED


class TestProcessFiles:
    @patch("nemoguardrails.cli.migration._write_transformed_content_and_rename_original")
    @patch("builtins.open", new_callable=mock_open, read_data="flow test\n")
    @patch("nemoguardrails.cli.migration.console")
    def test_process_co_files_v1_to_v2(self, mock_console, mock_file, mock_write):
        mock_write.return_value = True

        files = [Path("test.co")]
        result = _process_co_files(files, "1.0", False, True, False)

        assert result == 1
        mock_write.assert_called_once()

    @patch("nemoguardrails.cli.migration._write_transformed_content_and_rename_original")
    @patch("builtins.open", new_callable=mock_open, read_data="orwhen test\n")
    @patch("nemoguardrails.cli.migration.console")
    def test_process_co_files_v2_alpha_to_v2_beta(self, mock_console, mock_file, mock_write):
        mock_write.return_value = True

        files = [Path("test.co")]
        result = _process_co_files(files, "2.0-alpha", False, True, False)

        assert result >= 0

    @patch("nemoguardrails.cli.migration.parse_colang_file")
    @patch("nemoguardrails.cli.migration._write_transformed_content_and_rename_original")
    @patch("builtins.open", new_callable=mock_open, read_data="flow test")
    @patch("nemoguardrails.cli.migration.console")
    def test_process_co_files_with_validation(self, mock_console, mock_file, mock_write, mock_parse):
        mock_write.return_value = True
        mock_parse.return_value = {"flows": []}

        files = [Path("test.co")]
        result = _process_co_files(files, "1.0", False, True, True)

        assert result == 1
        mock_parse.assert_called()

    @patch("builtins.open", new_callable=MagicMock)
    @patch("nemoguardrails.cli.migration._process_sample_conversation_in_config")
    @patch("nemoguardrails.cli.migration._comment_rails_flows_in_config")
    @patch("nemoguardrails.cli.migration._get_rails_flows")
    @patch("nemoguardrails.cli.migration._get_raw_config")
    @patch("nemoguardrails.cli.migration.console")
    def test_process_config_files(
        self,
        mock_console,
        mock_get_config,
        mock_get_rails,
        mock_comment_rails,
        mock_process_sample,
        mock_open,
    ):
        mock_get_config.return_value = {"colang_version": "1.0"}
        mock_get_rails.return_value = {}  # No rails flows to process

        # Mock the file operations for _set_colang_version
        mock_file = MagicMock()
        mock_file.readlines.return_value = ["some_key: value\n"]
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=None)
        mock_open.return_value = mock_file

        files = ["config.yml"]
        result = _process_config_files(files)

        # The function returns 0 because there are no rails flows to write
        assert result == 0
        # Verify the colang version setting was attempted
        assert mock_open.called


class TestColang2AlphaSyntaxAdvanced:
    def test_regex_conversion_complex(self):
        input_lines = [
            'r"\\d{3}-\\d{3}-\\d{4}"',
            "r'[a-zA-Z0-9]+@[a-zA-Z0-9]+\\.[a-zA-Z]{2,}'",
        ]
        expected_output = [
            'regex("(\\d{3}-\\d{3}-\\d{4})")',
            "regex('([a-zA-Z0-9]+@[a-zA-Z0-9]+\\.[a-zA-Z]{2,})')",
        ]
        assert convert_colang_2alpha_syntax(input_lines) == expected_output

    def test_multiple_meta_decorators(self):
        input_lines = [
            "flow test_flow:",
            "# meta: loop_id=test_loop",
            "# meta: bot intent",
            "# meta: user visible",
            "    some content",
        ]
        expected_output = [
            '@loop("test_loop")',
            "@meta(bot_intent=True)",
            "@meta(user_visible=True)",
            "flow test_flow:",
            "    some content",
        ]
        assert convert_colang_2alpha_syntax(input_lines) == expected_output

    def test_deprecated_flows_conversion(self):
        input_lines = [
            "manage attentive posture",
            "track user presence state",
            "user became no longer present",
        ]
        expected_output = [
            "!!!! DEPRECATED FLOW (please remove): manage attentive posture",
            "!!!! DEPRECATED FLOW (please remove): track user presence state",
            "!!!! DEPRECATED FLOW (please remove): user became no longer present",
        ]
        assert convert_colang_2alpha_syntax(input_lines) == expected_output


class TestColang1SyntaxAdvanced:
    def test_complex_flow_conversion(self):
        input_lines = [
            "define flow user-request's_flow",
            '    when user "help"',
            "        $response = execute get_help",
            "        stop",
        ]
        expected_output = [
            "flow user request s_flow",
            '    when user "help"',
            "        $response = await GetHelpAction",
            "        abort",
        ]
        assert convert_colang_1_syntax(input_lines) == expected_output

    def test_variable_globalization_with_config(self):
        input_lines = [
            "$config.timeout = 30",
            "$config.retry_count = 3",
        ]
        result = convert_colang_1_syntax(input_lines)
        assert "global $system.config.timeout" in result[0]
        assert "$system.config.timeout = 30" in result[0]
        assert "global $system.config.retry_count" in result[1]
        assert "$system.config.retry_count = 3" in result[1]

    def test_bot_and_user_ellipsis_conversion(self):
        input_lines = [
            "bot ...",
            "user ...",
        ]
        expected_output = [
            "bot said something",
            "user said something",
        ]
        assert convert_colang_1_syntax(input_lines) == expected_output

    def test_mixed_define_statements(self):
        input_lines = [
            "define bot greeting",
            '"Hello there!"',
            '"Hi!"',
            "define user request",
            '"I need help"',
            '"Can you assist me?"',
        ]
        expected_output = [
            "flow bot greeting",
            'bot say "Hello there!"',
            'or bot say "Hi!"',
            "flow user request",
            'user said "I need help"',
            'or user said "Can you assist me?"',
        ]
        assert convert_colang_1_syntax(input_lines) == expected_output

    def test_await_action_camelcase_conversion(self):
        input_lines = [
            "await fetch_user_data",
            "await process_payment_info",
            "await send_email_notification",
        ]
        expected_output = [
            "await FetchUserDataAction",
            "await ProcessPaymentInfoAction",
            "await SendEmailNotificationAction",
        ]
        assert convert_colang_1_syntax(input_lines) == expected_output
