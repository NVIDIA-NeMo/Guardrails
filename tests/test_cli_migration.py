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

import textwrap

import pytest

from nemoguardrails.cli.migration import (
    convert_colang_1_syntax,
    convert_colang_2alpha_syntax,
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
        # TODO(Rdinu): Need to see if this conversion is correct
        input_lines = ["$config.setting = true"]
        expected_output = [
            "global $system.config.setting\n$system.config.setting = true"
        ]
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
