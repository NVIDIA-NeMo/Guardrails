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
from datetime import datetime, timedelta

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.utils import new_event_dict, new_uuid
from tests.utils import TestChat


@pytest.fixture
def config_1():
    return RailsConfig.from_content(
        colang_content="""
        import attention
        import core


        flow handle inattentive utterances
          user said something inattentively
          bot say "got inattentive"

        flow handle attentive utterances
          user said something
          bot say "got attentive"

        flow switching to attentive
          user said "up"
          $id = str(uid())
          send AttentionUserActionStarted(attention_level="engaged", action_uid=$id)
          bot say "up"

        flow switching to inattentive
          user said "down"
          $id = str(uid())
          send AttentionUserActionStarted(attention_level="disengaged", action_uid=$id)
          bot say "down"

        flow main
          activate tracking user attention
          activate handle inattentive utterances
          activate handle attentive utterances
          activate switching to attentive
          activate switching to inattentive

        """,
        yaml_content="""
        colang_version: "2.x"
        """,
    )


def test_1_1(config_1):
    chat = TestChat(
        config_1,
        llm_completions=[],
    )

    chat >> "hi"
    chat << "got attentive"


def test_1_2(config_1):
    chat = TestChat(
        config_1,
        llm_completions=[],
    )

    chat >> "up"
    chat << "up"
    chat >> "hello there"
    chat << "got attentive"


def test_1_3(config_1):
    chat = TestChat(
        config_1,
        llm_completions=[],
    )

    chat >> "down"
    chat << "down"
    chat >> "hello there"
    chat << "got inattentive"


def test_1_4(config_1):
    chat = TestChat(
        config_1,
        llm_completions=[],
    )
    attention_action_uid = new_uuid()
    utterance_action_uid = new_uuid()

    now = datetime.now()

    events = [
        new_event_dict(
            "AttentionUserActionStarted",
            action_uid=attention_action_uid,
            attention_level="engaged",
            action_started_at=now.isoformat(),
        ),
        new_event_dict(
            "UtteranceUserActionStarted",
            action_uid=utterance_action_uid,
            action_started_at=(now + timedelta(seconds=1)).isoformat(),
        ),
        new_event_dict(
            "AttentionUserActionUpdated",
            action_uid=attention_action_uid,
            attention_level="disengaged",
            action_updated_at=(now + timedelta(seconds=3)).isoformat(),
        ),
        new_event_dict(
            "UtteranceUserActionFinished",
            action_uid=utterance_action_uid,
            final_transcript="hello there",
            is_success=True,
            action_finished_at=(now + timedelta(seconds=4)).isoformat(),
        ),
    ]

    for event in events:
        chat >> event

    chat << "got attentive"


def test_1_5(config_1):
    chat = TestChat(
        config_1,
        llm_completions=[],
    )
    attention_action_uid = new_uuid()
    utterance_action_uid = new_uuid()

    now = datetime.now()

    events = [
        new_event_dict(
            "AttentionUserActionStarted",
            action_uid=attention_action_uid,
            attention_level="disengaged",
            action_started_at=now.isoformat(),
        ),
        new_event_dict(
            "UtteranceUserActionStarted",
            action_uid=utterance_action_uid,
            action_started_at=(now + timedelta(seconds=1)).isoformat(),
        ),
        new_event_dict(
            "AttentionUserActionUpdated",
            action_uid=attention_action_uid,
            attention_level="engaged",
            action_updated_at=(now + timedelta(seconds=3)).isoformat(),
        ),
        new_event_dict(
            "UtteranceUserActionFinished",
            action_uid=utterance_action_uid,
            final_transcript="hello there",
            is_success=True,
            action_finished_at=(now + timedelta(seconds=4)).isoformat(),
        ),
    ]

    for event in events:
        chat >> event

    chat << "got inattentive"


@pytest.fixture
def config_2():
    return RailsConfig.from_content(
        colang_content="""
        import core

        flow counting events
          global $counting
          match UtteranceUserActionStarted() as $event
            or UtteranceUserActionFinished() as $event
            or UtteranceUserActionUpdated() as $event
            or AttentionUserActionUpdated() as $event
            or AttentionUserActionStarted() as $event
            or AttentionUserActionFinished() as $event
          await UpdateAttentionMaterializedViewAction(event=$event)
          $counting = $counting + 1

        flow responding with count
          priority 0.1
          global $counting
          $counting = 0
          user said "hi"
          bot say "count is {$counting}"

        flow main
          activate counting events
          activate responding with count



        """,
        yaml_content="""
        colang_version: "2.x"
        """,
    )


def test_2_1(config_2):
    chat = TestChat(
        config_2,
        llm_completions=[],
    )
    uid = new_uuid()
    now = datetime.now()
    event = new_event_dict(
        "AttentionUserActionStarted",
        action_uid=uid,
        attention_level="engaged",
        action_started_at=now,
    )
    chat >> event
    chat >> "hello"
    chat >> "hi"
    chat << "count is 4"
