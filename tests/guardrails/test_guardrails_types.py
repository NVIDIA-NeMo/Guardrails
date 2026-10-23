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

import copy
from typing import Any, ClassVar

import pytest

from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.exceptions import InvalidCheckRequestError, UnreadableContentPartError
from nemoguardrails.guardrails.guardrails_types import (
    LLMMessage,
    LLMMessages,
    RailCallRecord,
    RailResult,
    client_reason,
    current_user_turn_index,
    display_reason,
    flatten_text_parts,
    last_user_content,
    rewrite_user_message,
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

        assert result.return_value == {"allowed": False, "failed": False, "policy_violations": ["S1: Violence"]}

    def test_return_value_of_a_bare_allow_states_only_the_decision(self):
        """A rail with no metadata still yields a verdict the GenerationLog can record."""
        assert RailResult.allow().return_value == {"allowed": True, "failed": False}

    def test_return_value_reports_a_rail_that_broke(self):
        """The log separates a rail that failed from one that decided, as the message does."""
        failed = RailResult(RailOutcome.failure(reason="content safety check input error: parser blew up"))

        assert failed.return_value == {"allowed": False, "failed": True}

    def test_metadata_cannot_overwrite_the_decision(self):
        """A custom action putting ``allowed`` in metadata cannot log a block as an allow."""
        result = RailResult.block(metadata={"allowed": True, "policy_violations": ["S1: Violence"]})

        assert result.return_value == {"allowed": False, "failed": False, "policy_violations": ["S1: Violence"]}

    def test_metadata_cannot_forge_a_rail_failure(self):
        """A rail that decided to block cannot log itself as having broken."""
        result = RailResult.block(metadata={"failed": True})

        assert result.return_value == {"allowed": False, "failed": False}

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

    def test_a_rewrite_before_a_block_stays_out_of_equality(self):
        """What the rails rewrote ahead of a block is capture data, so the block alone is the verdict compared."""
        kept = RailResult(RailOutcome.block(reason="blocked"), rewrite_before_block="my ssn is <SSN>")

        assert kept == RailResult.block(reason="blocked")

    def test_a_pending_rewrite_stays_out_of_equality(self):
        """Whether a rewrite never landed is capture data, so the verdict alone is compared."""
        pending = RailResult(RailOutcome.allow(), rewrite_pending=True)

        assert pending == RailResult.allow()

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

    CROWDSTRIKE_VERDICT: ClassVar[dict[str, Any]] = {
        "blocked": True,
        "guard_output": "policy 7",
        "user_message": "my private question",
        "bot_message": "the draft reply",
    }

    # ``_regex_outcome`` builds metadata from its whole ``RegexDetectionResult``, so the
    # text being checked and the substrings that matched are both in there. The account
    # number is a deliberately non-card-shaped sentinel: the test needs a string that
    # *looks* sensitive, and a real card format trips PII scanners on every run.
    REGEX_VERDICT: ClassVar[dict[str, Any]] = {
        "is_match": True,
        "text": "my account is ACCT-0000-1111-2222",
        "detections": ["ACCT-0000-1111-2222"],
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
            (REGEX_VERDICT, "ACCT-0000-1111-2222"),
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

    def test_the_log_carries_the_evidence_the_caller_does_not(self):
        """The restriction is on the payload; diagnosis keeps every field that describes the verdict."""
        result = RailResult.block(metadata=self.CROWDSTRIKE_VERDICT, triggered_rail="crowdstrike aidr guard input")

        rendered = display_reason(result)

        assert "guard_output: policy 7" in rendered
        assert "guard_output" not in client_reason(result)

    @pytest.mark.parametrize(
        ("metadata", "withheld"),
        [
            (CROWDSTRIKE_VERDICT, "my private question"),
            (CROWDSTRIKE_VERDICT, "the draft reply"),
            (REGEX_VERDICT, "my account is ACCT-0000-1111-2222"),
            ({"text": "the whole bot response", "detections": ["markdown_xss"]}, "the whole bot response"),
        ],
        ids=["user-message", "bot-message", "regex-text", "injection-text"],
    )
    def test_a_rail_echoing_the_request_does_not_put_it_in_the_log(self, metadata, withheld):
        """A block writes a log line unconditionally, so content in it would be an opt-out-free leak."""
        result = RailResult.block(metadata=metadata, triggered_rail="a rail")

        assert withheld not in display_reason(result)

    def test_dropping_the_content_leaves_the_rest_of_the_verdict(self):
        """Withholding the text must not cost the fields that say why the rail objected to it."""
        result = RailResult.block(metadata=self.REGEX_VERDICT, triggered_rail="regex check input")

        rendered = display_reason(result)

        assert "detections: ACCT-0000-1111-2222" in rendered
        assert "is_match: True" in rendered
        assert "source: input" in rendered

    def test_a_verdict_of_nothing_but_content_falls_back_to_the_rail_name(self):
        """With every field withheld there is no evidence left, and the rail name is what remains."""
        result = RailResult.block(metadata={"text": "the whole bot response"}, triggered_rail="injection detection")

        assert display_reason(result) == "injection detection"

    def test_a_block_with_nothing_to_say_is_unspecified(self):
        """A rail that supplies neither a reason nor a name still yields a printable string."""
        assert client_reason(RailResult.block()) == "unspecified"


class TestMessageRewriting:
    """Tests for the helpers that locate and rewrite the turn a rail checks."""

    MESSAGES: ClassVar[LLMMessages] = [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "my ssn is 123-45-6789"},
        {"role": "assistant", "content": "I cannot help with that"},
    ]

    def test_the_checked_turn_is_the_last_user_message_carrying_content(self):
        """The one turn a rail reads, so also the one a rewrite must replace."""
        assert current_user_turn_index(self.MESSAGES) == 1
        assert last_user_content(self.MESSAGES) == "my ssn is 123-45-6789"

    def test_an_empty_trailing_user_turn_is_not_the_checked_turn(self):
        """A blank turn is not what the rail was given, so it must not attract the rewrite."""
        messages: LLMMessages = [{"role": "user", "content": "my ssn is 123-45-6789"}, {"role": "user", "content": ""}]

        assert current_user_turn_index(messages) == 0
        assert rewrite_user_message(messages, "my ssn is <SSN>")[0]["content"] == "my ssn is <SSN>"

    def test_a_conversation_with_no_user_turn_has_nothing_to_check(self):
        """Rails read an empty string rather than raising, matching what the library actions expect."""
        assert current_user_turn_index([{"role": "assistant", "content": "hello"}]) is None
        assert last_user_content([{"role": "assistant", "content": "hello"}]) == ""
        assert last_user_content([]) == ""

    def test_rewriting_replaces_only_the_checked_turn(self):
        """Surrounding turns are the conversation's history and are left as they were."""
        rewritten = rewrite_user_message(self.MESSAGES, "my ssn is <SSN>")

        assert rewritten[1]["content"] == "my ssn is <SSN>"
        assert rewritten[0] == self.MESSAGES[0]
        assert rewritten[2] == self.MESSAGES[2]

    def test_rewriting_leaves_the_callers_list_and_turns_untouched(self):
        """The caller's list reaches the engine by identity, so neither it nor its dicts may be mutated."""
        messages: LLMMessages = [{"role": "user", "content": "my ssn is 123-45-6789"}]
        original_turn = messages[0]

        rewritten = rewrite_user_message(messages, "my ssn is <SSN>")

        assert messages == [{"role": "user", "content": "my ssn is 123-45-6789"}]
        assert messages[0] is original_turn
        assert rewritten[0] is not original_turn

    def test_rewriting_with_no_user_turn_raises(self):
        """A rewrite with nowhere to land fails loudly rather than being dropped."""
        with pytest.raises(ValueError, match="nothing to rewrite"):
            rewrite_user_message([{"role": "assistant", "content": "hello"}], "masked")


PART_SECRET = "SECRET-CONTENT"


def _text_part(text: str) -> dict:
    """A text content part."""
    return {"type": "text", "text": text}


class TestFlattenTextParts:
    """Tests for flatten_text_parts, which turns text content parts into the string a rail reads."""

    @pytest.mark.parametrize("role", ["user", "assistant"])
    def test_text_parts_are_joined_with_newlines(self, role):
        """A list of text parts on a flattened role becomes one string joined with newlines."""
        messages = [{"role": role, "content": [_text_part("Hello"), _text_part("world")]}]

        assert flatten_text_parts(messages) == [{"role": role, "content": "Hello\nworld"}]

    def test_a_single_text_part_becomes_its_text_part(self):
        """A one-part list becomes that part's text, with no separator added."""
        messages = [{"role": "user", "content": [_text_part("Hello")]}]

        assert flatten_text_parts(messages) == [{"role": "user", "content": "Hello"}]

    def test_an_assistant_refusal_part_contributes_its_text_part(self):
        """An assistant refusal part is model text the user sees, so it is joined in alongside the text parts."""
        messages = [
            {"role": "assistant", "content": [_text_part("Sorry."), {"type": "refusal", "refusal": "I can't."}]}
        ]

        assert flatten_text_parts(messages) == [{"role": "assistant", "content": "Sorry.\nI can't."}]

    def test_other_message_fields_are_kept(self):
        """Flattening replaces only content, keeping the message's other fields."""
        messages = [{"role": "user", "content": [_text_part("Hello")], "name": "alice"}]

        assert flatten_text_parts(messages) == [{"role": "user", "content": "Hello", "name": "alice"}]

    def test_an_empty_list_becomes_an_empty_string(self):
        """An empty part list reads as empty content, like an empty string."""
        messages = [{"role": "user", "content": []}]

        assert flatten_text_parts(messages) == [{"role": "user", "content": ""}]

    @pytest.mark.parametrize(
        ("role", "part"),
        [
            ("user", {"type": "image_url", "image_url": {"url": f"https://example.com/{PART_SECRET}.png"}}),
            ("user", {"type": "file", "file": {"file_data": PART_SECRET, "filename": "doc.pdf"}}),
            ("user", {"type": "input_audio", "input_audio": {"data": PART_SECRET, "format": "wav"}}),
            ("user", {"type": "refusal", "refusal": PART_SECRET}),
            ("user", {"type": "video", "video": PART_SECRET}),
            ("user", {"type": [PART_SECRET], "text": "hi"}),
            ("user", PART_SECRET),
            ("user", {"type": "text", "text": [PART_SECRET]}),
            ("assistant", {"type": "image_url", "image_url": {"url": f"https://example.com/{PART_SECRET}.png"}}),
        ],
        ids=[
            "image_url",
            "file",
            "input_audio",
            "refusal_on_user",
            "unknown_type",
            "non_string_type",
            "not_a_dict",
            "non_string_text",
            "image_url_on_assistant",
        ],
    )
    def test_a_non_text_part_raises_without_echoing_content(self, role, part):
        """A part that is not text raises InvalidCheckRequestError, whose message echoes none of the content."""
        messages = [{"role": role, "content": [_text_part(PART_SECRET), part]}]

        with pytest.raises(InvalidCheckRequestError) as exc_info:
            flatten_text_parts(messages)

        assert PART_SECRET not in str(exc_info.value)

    def test_the_error_names_the_role_index_and_part_type(self):
        """The error locates the offending part by role, message index and part type."""
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}]},
        ]

        with pytest.raises(InvalidCheckRequestError) as exc_info:
            flatten_text_parts(messages)

        error = str(exc_info.value)
        assert "user" in error
        assert "messages[1]" in error
        assert "image_url" in error

    def test_a_non_text_part_raises_a_structured_error(self):
        """The error is an UnreadableContentPartError carrying the part's location and problem as fields."""
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": [_text_part("Hi"), {"type": "image_url", "image_url": {"url": "x"}}]},
        ]

        with pytest.raises(UnreadableContentPartError) as exc_info:
            flatten_text_parts(messages)

        error = exc_info.value
        assert isinstance(error, InvalidCheckRequestError)
        assert (error.message_index, error.role, error.part_index, error.problem) == (
            1,
            "user",
            1,
            "has type 'image_url'",
        )
        assert str(error) == (
            "cannot check messages[1] (user): content part 1 has type 'image_url'; "
            "a check reads only text parts, and refusal parts on assistant messages"
        )


class TestUnreadableContentPartError:
    """Tests for the structured error flatten_text_parts raises."""

    def test_at_message_index_moves_only_the_position(self):
        """at_message_index returns the same error at a new message position, with the message text to match."""
        error = UnreadableContentPartError(message_index=1, role="user", part_index=0, problem="has type 'file'")

        moved = error.at_message_index(0)

        assert (moved.message_index, moved.role, moved.part_index, moved.problem) == (0, "user", 0, "has type 'file'")
        assert str(moved).startswith("cannot check messages[0] (user): content part 0 has type 'file';")
        assert error.message_index == 1

    @pytest.mark.parametrize(
        "message",
        [
            {"role": "system", "content": [_text_part("Be helpful.")]},
            {"role": "developer", "content": [_text_part("Be helpful.")]},
            {"role": "tool", "tool_call_id": "call_1", "content": [_text_part("18C")]},
            {"role": "tool", "tool_call_id": "call_1", "content": [{"type": "image_url"}]},
        ],
        ids=["system", "developer", "tool", "tool_with_non_text_part"],
    )
    def test_roles_outside_roles_pass_through_unchanged(self, message):
        """A message whose role is not flattened is passed through as the same object, list content and all."""
        messages = [message]

        flattened = flatten_text_parts(messages)

        assert flattened[0] is message

    @pytest.mark.parametrize(
        "message",
        [
            {"role": "user", "content": "hello"},
            {"role": "user", "content": None},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1"}]},
            {"role": "context", "content": {"user_id": "u1"}},
        ],
        ids=["string", "user_none", "assistant_tool_call_none", "context_dict"],
    )
    def test_non_list_content_passes_through_unchanged(self, message):
        """String, None and dict content is passed through as the same object."""
        messages = [message]

        flattened = flatten_text_parts(messages)

        assert flattened[0] is message

    def test_roles_selects_which_messages_are_flattened(self):
        """Only the roles passed in are flattened; the defaults no longer apply."""
        messages = [
            {"role": "user", "content": [_text_part("Hi")]},
            {"role": "tool", "tool_call_id": "call_1", "content": [_text_part("18C")]},
        ]

        flattened = flatten_text_parts(messages, roles=("tool",))

        assert flattened[0] is messages[0]
        assert flattened[1] == {"role": "tool", "tool_call_id": "call_1", "content": "18C"}

    def test_the_input_is_not_mutated(self):
        """Flattening returns a new list and leaves the caller's list and message dicts untouched."""
        messages = [
            {"role": "user", "content": [_text_part("Hello"), _text_part("world")]},
            {"role": "assistant", "content": [_text_part("Hi")]},
        ]
        original = copy.deepcopy(messages)

        flattened = flatten_text_parts(messages)

        assert flattened is not messages
        assert messages == original


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
