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

"""Tests for multi-line bot say quoted string handling in get_first_bot_action."""

from nemoguardrails.actions.llm.utils import (
    _MAX_QUOTE_CONTINUATION_LINES,
    _has_unclosed_quote,
    get_first_bot_action,
)


class TestHasUnclosedQuote:
    """Tests for the _has_unclosed_quote helper."""

    def test_no_quotes(self):
        assert _has_unclosed_quote("hello world") is False

    def test_closed_quote(self):
        assert _has_unclosed_quote('bot say "hello"') is False

    def test_unclosed_quote(self):
        assert _has_unclosed_quote('bot say "hello') is True

    def test_escaped_quote_inside(self):
        assert _has_unclosed_quote('bot say "he said \\"hi\\"') is True

    def test_escaped_quote_at_end(self):
        assert _has_unclosed_quote('bot say "he said \\"hi\\""') is False

    def test_empty_string(self):
        assert _has_unclosed_quote("") is False

    def test_empty_quoted_string(self):
        assert _has_unclosed_quote('""') is False


class TestGetFirstBotActionMultiline:
    """Tests for multi-line quoted string handling in get_first_bot_action."""

    def test_single_line_bot_action(self):
        """Single-line bot say continues to work correctly."""
        lines = [
            "bot intent: bot respond about GPU comparison",
            'bot action: bot say "The RTX 4090 is faster than the RTX 3090."',
        ]
        result = get_first_bot_action(lines)
        assert result == 'bot say "The RTX 4090 is faster than the RTX 3090."'

    def test_multiline_bot_action_collected_and_escaped(self):
        """Multi-line bot say is fully collected with newlines escaped."""
        llm_completion = (
            "bot intent: bot respond provide information about NVIDIA GPU comparison\n"
            "bot action: bot say \"Let's break down the differences.\n"
            "\n"
            "**Performance:** The RTX 4090 is a beast.\n"
            "\n"
            '**Price:** The RTX 4090 has a higher price tag."'
        )
        lines = llm_completion.splitlines()
        result = get_first_bot_action(lines)

        assert result is not None
        expected = (
            "bot say \"Let's break down the differences."
            "\\n"
            "\\n"
            "**Performance:** The RTX 4090 is a beast."
            "\\n"
            "\\n"
            '**Price:** The RTX 4090 has a higher price tag."'
        )
        assert result == expected
        assert "Performance" in result
        assert "Price" in result
        assert result.endswith('"')

    def test_multiline_response_produces_valid_colang(self):
        """The fixed bot action produces valid Colang that parses successfully."""
        llm_completion = (
            "bot intent: bot respond provide information about NVIDIA GPU comparison\n"
            "bot action: bot say \"Let's break down the differences.\n"
            "\n"
            '**Performance:** The RTX 4090 is faster."'
        )
        lines = llm_completion.splitlines()
        bot_action = get_first_bot_action(lines)

        bot_intent = "bot respond provide information about NVIDIA GPU comparison"
        flow_name = f"_dynamic_test {bot_intent}"
        flow_body = f'@meta(bot_intent="{bot_intent}")\n' + f"flow {flow_name}\n" + f"  {bot_action}"

        assert "\\n" in flow_body
        assert flow_body.endswith('"')

        from nemoguardrails.colang.v2_x.lang.parser import parse_colang_file

        parsed = parse_colang_file(
            filename="",
            content=flow_body,
            include_source_mapping=True,
        )
        assert parsed is not None
        assert "flows" in parsed
        assert len(parsed["flows"]) == 1

    def test_full_multiline_reproduction(self):
        """Full reproduction using a multi-paragraph LLM response."""
        llm_completion = (
            "user intent: user asked about NVIDIA GPU comparison\n"
            "bot intent: bot respond provide information about NVIDIA GPU comparison\n"
            "bot action: bot say \"Let's break down the differences between the RTX 3090 and RTX 4090.\n"
            "\n"
            "**Price per Performance:** The RTX 4090 is a more recent release, and as such, "
            "it comes with a higher price tag. However, in terms of price per performance, "
            "the RTX 4090 offers significant improvements over its predecessor.\n"
            "\n"
            "**Performance:** In terms of raw performance, the RTX 4090 is a beast. It features "
            "24 GB of GDDR6X memory and a boost clock speed of up to 1.71 GHz.\n"
            "\n"
            "**Price:** As mentioned earlier, the RTX 4090 comes with a higher price tag "
            "compared to the RTX 3090. The RTX 3090 starts at around $1,499, while the "
            'RTX 4090 starts at around $1,599."'
        )

        lines = llm_completion.splitlines()
        result = get_first_bot_action(lines)

        assert "Price per Performance" in result
        assert "Performance" in result
        assert "Price" in result
        assert "$1,599" in result
        assert result.endswith('"')
        assert "\\n" in result
        assert "\n" not in result

    def test_single_paragraph_response_unchanged(self):
        """Single paragraph responses continue to work correctly."""
        llm_completion = (
            "user intent: user asked about NVIDIA GPU comparison\n"
            "bot intent: bot respond provide information about NVIDIA GPU comparison\n"
            'bot action: bot say "The main difference between the RTX 3090 and RTX 4090 '
            "lies in their performance capabilities. In terms of price per performance, "
            "the RTX 3090 is generally considered a better value, offering around 80-90% "
            'of the performance of the RTX 4090 at a significantly lower cost."'
        )

        lines = llm_completion.splitlines()
        result = get_first_bot_action(lines)

        assert result.startswith('bot say "The main difference')
        assert result.endswith('lower cost."')
        assert "80-90%" in result
        assert "\\n" not in result

    def test_bot_action_with_trailing_text_after_quote(self):
        """Content after the closing quote is not included in the action."""
        llm_completion = 'bot action: bot say "Line one.\nLine two."\nNow, let\'s continue the conversation!'
        lines = llm_completion.splitlines()
        result = get_first_bot_action(lines)

        assert result == 'bot say "Line one.\\nLine two."'
        assert "continue" not in result

    def test_bot_action_with_escaped_quotes_inside(self):
        """Escaped quotes inside the string don't break collection."""
        llm_completion = 'bot action: bot say "She said \\"hello\\" to everyone.\nThen she left."'
        lines = llm_completion.splitlines()
        result = get_first_bot_action(lines)

        assert result == 'bot say "She said \\"hello\\" to everyone.\\nThen she left."'

    def test_multiple_bot_actions_concatenated(self):
        """Multiple bot actions are concatenated (existing behavior preserved)."""
        lines = [
            'bot action: bot say "First action."',
            'bot action: bot say "Second action."',
        ]
        result = get_first_bot_action(lines)
        assert 'bot say "First action."' in result
        assert 'bot say "Second action."' in result

    def test_empty_input(self):
        """Empty input returns empty string."""
        result = get_first_bot_action([])
        assert result == ""

    def test_no_bot_action(self):
        """Input without bot action returns empty string."""
        lines = [
            "user intent: user asked something",
            "bot intent: bot respond",
        ]
        result = get_first_bot_action(lines)
        assert result == ""

    def test_permanently_unclosed_quote_has_safety_bound(self):
        """Permanently unclosed quote stops collecting after max continuation lines."""
        lines = ['bot action: bot say "This never closes'] + [
            f"line {i}" for i in range(_MAX_QUOTE_CONTINUATION_LINES + 10)
        ]
        result = get_first_bot_action(lines)
        # Should not absorb all lines — safety bound kicks in
        assert result is not None
        assert f"line {_MAX_QUOTE_CONTINUATION_LINES + 5}" not in result

    def test_unclosed_quote_does_not_absorb_subsequent_actions(self):
        """Unclosed quote with safety bound doesn't consume later bot actions."""
        lines = (
            ['bot action: bot say "Never closed']
            + [f"line {i}" for i in range(_MAX_QUOTE_CONTINUATION_LINES + 5)]
            + ['bot action: bot say "Next action"']
        )
        result = get_first_bot_action(lines)
        assert "Next action" not in result

    def test_whitespace_continuation_lines(self):
        """Whitespace-only lines inside a quoted string are preserved."""
        lines = [
            'bot action: bot say "Hello',
            "   ",
            'world"',
        ]
        result = get_first_bot_action(lines)
        assert result == 'bot say "Hello\\n   \\nworld"'

    def test_closing_quote_on_own_line(self):
        """A closing quote on its own line properly closes the string."""
        lines = [
            'bot action: bot say "Hello',
            '"',
        ]
        result = get_first_bot_action(lines)
        assert result == 'bot say "Hello\\n"'
        assert not _has_unclosed_quote(result)

    def test_multiline_with_leading_and_in_content(self):
        """Multi-line content starting with '  and' should not be double-appended."""
        lines = [
            'bot action: bot say "You can choose option A',
            '  and option B"',
        ]
        result = get_first_bot_action(lines)
        assert result == 'bot say "You can choose option A\\n  and option B"'
        assert result.count("and option B") == 1, "Content should not be duplicated"

    def test_multiline_with_leading_or_in_content(self):
        """Multi-line content starting with '  or' should not be double-appended."""
        lines = [
            'bot action: bot say "Choose option A',
            '  or option B"',
        ]
        result = get_first_bot_action(lines)
        assert result == 'bot say "Choose option A\\n  or option B"'
        assert result.count("or option B") == 1, "Content should not be duplicated"

    def test_escaped_backslash_before_closing_quote(self):
        """Escaped backslash before closing quote is correctly treated as closed."""
        # \\\\" means escaped backslash + real closing quote
        assert _has_unclosed_quote('bot say "path C:\\\\"') is False
        assert _has_unclosed_quote('bot say "test\\\\"') is False
        # But \\\" is still an escaped quote (unclosed)
        assert _has_unclosed_quote('bot say "test\\"') is True
