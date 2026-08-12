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

"""Unit tests for guardrails_types module."""

import pytest

from nemoguardrails.guardrails.guardrails_types import (
    LLMMessage,
    LLMMessages,
    RailCallRecord,
    RailResult,
    client_reason,
    display_reason,
    truncate,
)


class TestRailResult:
    """Tests for the RailResult frozen dataclass."""

    def test_safe_result_defaults(self):
        """Test creating a safe result with default reason=None."""
        result = RailResult.allow()
        assert result.is_safe is True
        assert result.reason is None

    def test_safe_result_explicit_none(self):
        """Test creating a safe result with explicit reason=None."""
        result = RailResult.allow(reason=None)
        assert result.is_safe is True
        assert result.reason is None

    def test_unsafe_result_with_reason(self):
        """Test creating an unsafe result with a reason string."""
        result = RailResult.block(reason="Content safety violation")
        assert result.is_safe is False
        assert result.reason == "Content safety violation"

    def test_unsafe_result_without_reason(self):
        """Test creating an unsafe result without a reason."""
        result = RailResult.block()
        assert result.is_safe is False
        assert result.reason is None

    def test_equality_same_values(self):
        """Test that two RailResults with the same values are equal."""
        a = RailResult.allow()
        b = RailResult.allow()
        assert a == b

    def test_equality_with_reason(self):
        """Test equality when both have the same reason."""
        a = RailResult.block(reason="blocked")
        b = RailResult.block(reason="blocked")
        assert a == b

    def test_inequality_different_is_safe(self):
        """Test inequality when is_safe differs."""
        a = RailResult.allow()
        b = RailResult.block()
        assert a != b

    def test_inequality_different_reason(self):
        """Test inequality when reason differs."""
        a = RailResult.block(reason="reason1")
        b = RailResult.block(reason="reason2")
        assert a != b

    def test_repr(self):
        """Test that the representation shows the wrapped outcome, which carries the verdict."""
        result = RailResult.block(reason="jailbreak")
        assert "RailDecision.BLOCK" in repr(result)
        assert "reason='jailbreak'" in repr(result)

    def test_reason_with_empty_string(self):
        """Test that empty string reason is distinct from None."""
        result = RailResult.block(reason="")
        assert result.reason == ""
        assert result != RailResult.block(reason=None)

    def test_return_value_derives_from_the_outcome(self):
        """The structured verdict is the decision plus the outcome's metadata, not a stored copy."""
        result = RailResult.block(metadata={"policy_violations": ["S1: Violence"]})

        assert result.return_value == {"allowed": False, "policy_violations": ["S1: Violence"]}

    def test_return_value_of_a_bare_allow_states_only_the_decision(self):
        """A rail with no metadata still yields a verdict the GenerationLog can record."""
        assert RailResult.allow().return_value == {"allowed": True}

    def test_metadata_participates_in_equality(self):
        """Two blocks differing only in evidence are no longer equal.

        The verdict now lives in a single ``RailOutcome``, so metadata is compared with the
        rest of it. Previously ``return_value`` was excluded from equality and these compared
        equal; this is the one behavioural change in the wrapper.
        """
        a = RailResult.block(reason="blocked", metadata={"score": 0.9})
        b = RailResult.block(reason="blocked", metadata={"score": 0.1})

        assert a != b

    def test_records_stay_out_of_equality(self):
        """Captured log data is not part of the verdict, so it does not affect comparison."""
        record = RailCallRecord(flow="content safety check input", rail_type="input", is_safe=False)

        assert RailResult.block(reason="blocked", records=(record,)) == RailResult.block(reason="blocked")

    def test_is_unhashable_and_says_so(self):
        """The wrapped outcome is unhashable, and the error names this type rather than leaking from inside."""
        with pytest.raises(TypeError, match="unhashable type: 'RailResult'"):
            hash(RailResult.allow())


class TestTruncate:
    """Tests for the truncate helper."""

    def test_short_string_unchanged(self):
        assert truncate("hello", 10) == "hello"

    def test_exact_length_unchanged(self):
        assert truncate("hello", 5) == "hello"

    def test_long_string_truncated(self):
        assert truncate("hello world", 5) == "hello..."

    def test_max_len_zero_truncates_everything(self):
        assert truncate("hello", 0) == "..."

    def test_none_max_len_uses_default(self):
        short = "x" * 200
        assert truncate(short, None) == short
        long = "x" * 201
        assert truncate(long, None) == "x" * 200 + "..."

    def test_non_string_input_converted(self):
        assert truncate(12345, 3) == "123..."


class TestDisplayReason:
    """Tests for the display_reason helper."""

    @pytest.mark.parametrize(
        "result, expected",
        [
            (RailResult.block(reason="Safety categories: S1: Violence"), "Safety categories: S1: Violence"),
            (
                RailResult.block(
                    reason="Safety categories: S1: Violence",
                    metadata={"policy_violations": ["S2: Sexual"]},
                    triggered_rail="content safety check input",
                ),
                "Safety categories: S1: Violence",
            ),
            (
                RailResult.block(
                    metadata={"policy_violations": ["S1: Violence", "S2: Sexual"]},
                    triggered_rail="content safety check input",
                ),
                "policy_violations: S1: Violence, S2: Sexual",
            ),
            (
                RailResult.block(
                    metadata={"policy_violations": [], "score": 0}, triggered_rail="content safety check input"
                ),
                "score: 0",
            ),
            (
                RailResult.block(
                    metadata={"policy_violations": ["S1: Violence"], "score": 0.97},
                    triggered_rail="content safety check input",
                ),
                "policy_violations: S1: Violence; score: 0.97",
            ),
            (
                RailResult.block(metadata={"policy_violations": None}, triggered_rail="content safety check input"),
                "content safety check input",
            ),
            (
                RailResult.block(triggered_rail="jailbreak detection model"),
                "jailbreak detection model",
            ),
            (RailResult.block(), "unspecified"),
        ],
        ids=[
            "reason",
            "reason-beats-evidence",
            "evidence",
            "empty-evidence-skipped",
            "multiple-fields-joined",
            "null-evidence-skipped",
            "rail-name",
            "nothing-at-all",
        ],
    )
    def test_prefers_the_most_specific_source(self, result, expected):
        """A stated reason wins, then the verdict's non-empty evidence, then the rail's name."""
        assert display_reason(result) == expected


class TestClientReason:
    """What reaches the caller in a block payload, as opposed to what reaches the log.

    The caller is told the rail's own ``reason`` and nothing else. Metadata is neutral
    evidence for diagnosis, not a client contract: crowdstrike_aidr puts the user and bot
    messages in it, regex puts the matched text in it, and f5 forwards the provider
    response. A rail that wants the caller to know why it blocked says so in ``reason``,
    which it authors for exactly that purpose.
    """

    CROWDSTRIKE_VERDICT = {
        "blocked": True,
        "guard_output": "policy 7",
        "user_message": "my private question",
        "bot_message": "the draft reply",
    }

    # ``_regex_outcome`` builds metadata from its whole ``RegexDetectionResult``, so the
    # text being checked and the substrings that matched are both in there.
    REGEX_VERDICT = {
        "is_match": True,
        "text": "my card is 4111 1111 1111 1111",
        "detections": ["4111 1111 1111 1111"],
        "source": "input",
    }

    def test_a_rail_authored_reason_is_what_the_caller_sees(self):
        """A rail that explains itself has that explanation forwarded verbatim."""
        result = RailResult.block(reason="Safety categories: S1: Violence")

        assert client_reason(result) == "Safety categories: S1: Violence"

    def test_the_reason_wins_over_the_rail_name(self):
        """Naming the rail is the fallback, not a prefix or a suffix."""
        result = RailResult.block(reason="Cisco AI Defense flagged the content as unsafe.", triggered_rail="a rail")

        assert client_reason(result) == "Cisco AI Defense flagged the content as unsafe."

    @pytest.mark.parametrize(
        ("metadata", "withheld"),
        [
            (CROWDSTRIKE_VERDICT, "my private question"),
            (REGEX_VERDICT, "4111 1111 1111 1111"),
            ({"policy_violations": ["S1: Violence"]}, "S1: Violence"),
            ({"trustworthiness_score": 0.12}, "0.12"),
            ({"score": 0.4, "threshold": 0.75}, "0.4"),
        ],
        ids=["crowdstrike", "regex", "content-safety", "cleanlab", "autoalign"],
    )
    def test_no_verdict_metadata_reaches_the_caller(self, metadata, withheld):
        """Metadata stays out of the payload whether or not it happens to look safe to disclose.

        Pinned across payloads that differ in how sensitive they look, because the previous
        design turned on that judgement per field and this one deliberately does not.
        """
        result = RailResult.block(metadata=metadata, triggered_rail="a rail")

        assert withheld not in client_reason(result)
        assert client_reason(result) == "a rail"

    def test_the_log_still_carries_what_the_caller_does_not(self):
        """The restriction is on the payload only; diagnosis keeps the full verdict."""
        result = RailResult.block(metadata=self.CROWDSTRIKE_VERDICT, triggered_rail="crowdstrike aidr guard input")

        rendered = display_reason(result)

        assert "my private question" in rendered
        assert "the draft reply" in rendered
        assert "my private question" not in client_reason(result)

    def test_a_block_with_nothing_to_say_is_unspecified(self):
        """A rail that supplies neither a reason nor a name still yields a printable string."""
        assert client_reason(RailResult.block()) == "unspecified"


class TestTypeAliases:
    """Tests for the LLMMessage and LLMMessages type aliases."""

    def test_llm_message_is_dict(self):
        """Test that LLMMessage is a dict type alias."""
        msg: LLMMessage = {"role": "user", "content": "hello"}
        assert isinstance(msg, dict)

    def test_llm_messages_is_list(self):
        """Test that LLMMessages is a list of dicts."""
        msgs: LLMMessages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        assert isinstance(msgs, list)
        assert all(isinstance(m, dict) for m in msgs)
