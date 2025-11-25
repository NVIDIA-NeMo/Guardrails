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

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from nemoguardrails.actions.actions import action
from nemoguardrails.colang.v2_x.runtime.flows import ActionEvent

UNKNOWN_ATTENTION_STATE = "unknown"

logger = logging.getLogger("nemoguardrails")


def log_p(what: str):
    """Log compatible with the nemoguardrails log output to show output as part of logging output"""
    logger.info("Colang Log %s :: %s", "(actions.py)0000", what)


def read_isoformat(timestamp: str) -> datetime:
    """
    ISO 8601 has multiple legal ways to indicate UTC timezone. 'Z' or '+00:00'. However the Python
    datetime.fromisoformat only accepts the latter.
    This function provides a more flexible wrapper to accept more valid IOS 8601 formats
    """
    normalized = timestamp.replace("Z", "+00:00")

    ms_digits = normalized.find("+") - normalized.find(".") - 1
    if ms_digits < 6:
        missing_zeros = "0" * (6 - ms_digits)
        normalized = normalized.replace("+", f"{missing_zeros}+")
    return datetime.fromisoformat(normalized)


def _get_action_timestamp(action_event_name: str, event_args) -> Optional[datetime]:
    """Extract the correct timestamp from the action event."""
    _mapping = {
        "UtteranceUserActionStarted": "action_started_at",
        "UtteranceUserActionFinished": "action_finished_at",
        "UtteranceUserActionTranscriptUpdated": "action_updated_at",
        "AttentionUserActionStarted": "action_started_at",
        "AttentionUserActionUpdated": "action_updated_at",
        "AttentionUserActionFinished": "action_finished_at",
    }
    if action_event_name not in _mapping:
        return None
    try:
        return read_isoformat(event_args[_mapping[action_event_name]])
    except Exception:
        log_p(f"Could not parse timestamp {event_args[_mapping[action_event_name]]}")
        return None


@dataclass
class StateChange:
    """Hold information about a state change"""

    state: str
    time: datetime


def compute_time_spent_in_states(changes: list[StateChange]) -> dict[str, timedelta]:
    """Returns the total number of seconds spent for each state in the list of state changes."""
    result: dict[str, timedelta] = {}
    for i in range(len(changes) - 1):
        result[changes[i].state] = result.get(changes[i].state, timedelta(seconds=0.0)) + (
            changes[i + 1].time - changes[i].time
        )

    return result


class UserAttentionMaterializedView:
    """
    Materialized view of the attention state distribution of the user while the user is talking.

    Note: This materialized view provides a very basic attention statistics,
    computed over the temporal distribution of attention levels during the duration of a user utterance,
    meaning what percentage of time the user was at particular attention level during the duration of
    the last utterance.
    """

    def __init__(self) -> None:
        self.user_is_talking = False
        self.sentence_distribution = {UNKNOWN_ATTENTION_STATE: 0.0}
        self.attention_events: list[ActionEvent] = []
        self.utterance_started_event = None
        self.utterance_last_event = None

    def reset_view(self) -> None:
        """Reset the view. Removing all attention events except for the most recent one"""
        self.attention_events = self.attention_events[-1:]
        self.utterance_last_event = None

    def update(self, event: ActionEvent, offsets: dict[str, float]) -> None:
        """Update the view based on the event to keep relevant attention events for the last user utterance.

        Args:
            event (ActionEvent): Action event to use for updating the view
            offsets (dict[str, float]): You can provide static offsets in seconds for every event type to correct for known latencies of these events.
        """
        # print(f"attention_events: {self.attention_events}")
        timestamp = _get_action_timestamp(event.name, event.arguments)
        if not timestamp:
            return

        event.corrected_datetime = timestamp + timedelta(seconds=offsets.get(event.name, 0.0))

        if event.name == "UtteranceUserActionStarted":
            self.reset_view()
            self.utterance_started_event = event
        elif event.name == "UtteranceUserActionFinished" or event.name == "UtteranceUserActionTranscriptUpdated":
            self.utterance_last_event = event
        elif event.name == "AttentionUserActionFinished":
            event.arguments["attention_level"] = UNKNOWN_ATTENTION_STATE
            self.attention_events.append(event)
        elif "Attention" in event.name:
            self.attention_events.append(event)

    def get_time_spent_percentage(self, attention_levels: list[str]) -> float:
        """Compute the time spent in the attention levels provided in `attention_levels` over the duration
        of the last user utterance.

        Args:
            attention_levels (list[str]): List of attention level names to consider `attentive`

        Returns:
            float: The percentage the user was in the attention levels provided. Returns 1.0 if no attention events have been registered.
        """
        log_p(f"attention_events={self.attention_events}")

        if not attention_levels:
            log_p("Attention: no attention_levels provided. Attention percentage set to 0.0")
            return 0.0

        # If one of the utterance boundaries are not available we return the attention percentage based on the most
        # recent attention level observed.
        if not self.utterance_started_event or not self.utterance_last_event:
            level = attention_levels[0]
            if self.attention_events:
                level = self.attention_events[-1].arguments["attention_level"]
            log_p(f"Attention: Utterance boundaries unclear. Deciding based on most recent attention_level={level}")
            return 1.0 if level in attention_levels else 0.0

        events = [
            e for e in self.attention_events if e.corrected_datetime < self.utterance_last_event.corrected_datetime
        ]
        log_p(f"filtered attention_events={events}")

        if len(events) == 0:
            return 1.0

        start_of_sentence_state = StateChange(
            events[0].arguments["attention_level"],
            self.utterance_started_event.corrected_datetime,
        )
        end_of_sentence_state = StateChange("no_state", self.utterance_last_event.corrected_datetime)
        state_changes_during_sentence = [
            StateChange(e.arguments["attention_level"], e.corrected_datetime) for e in events[1:]
        ]

        state_changes = [start_of_sentence_state] + state_changes_during_sentence + [end_of_sentence_state]
        durations = compute_time_spent_in_states(state_changes)

        # If the only state we observed during the duration of the utterance is UNKNOWN_ATTENTION_STATE we treat it as 1.0
        if len(durations) == 1 and UNKNOWN_ATTENTION_STATE in durations:
            return 1.0

        total = sum(durations.values(), timedelta())
        states_time = timedelta()
        for s in attention_levels:
            states_time += durations.get(s, timedelta())

        if total.total_seconds() == 0:
            log_p("No attention states observed. Assuming attentive.")
            return 1.0
        else:
            return abs(states_time.total_seconds() / total.total_seconds())


_attention_view = UserAttentionMaterializedView()


@action(name="UpdateAttentionMaterializedViewAction")
async def update_attention_materialized_view_action(
    event: ActionEvent, timestamp_offsets: Optional[dict] = None
) -> None:
    """
    Update the attention view. The attention view stores events relevant to computing
    user attention during the last user utterance.

    Args:
        event (ActionEvent): Supported actions events: AttentionUserAction and UtteranceUserAction
        timestamp_offsets (Optional[dict]): timestamp offset (in seconds) for certain event types.
            Example: timestamp_offsets = {"UtteranceUserActionFinished": -0.8} will adjust the
            timestamp of `UtteranceUserActionFinished` by -0.8seconds
    """
    _attention_view.update(event, offsets=timestamp_offsets or {})


@action(name="GetAttentionPercentageAction")
async def get_attention_percentage_action(attention_levels: list[str]) -> float:
    """Compute the attention level in percent during the last user utterance.

    Args:
        attention_levels : Name of attention levels for which the user is considered to be `attentive`

    Returns:
        float: The percentage the user was in the attention levels provided. Returns 1.0 if no attention events have been registered.
    """
    return _attention_view.get_time_spent_percentage(attention_levels)
