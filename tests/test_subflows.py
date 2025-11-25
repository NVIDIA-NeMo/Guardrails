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

from nemoguardrails import RailsConfig
from tests.utils import TestChat


def test_simple_subflow_call():
    config = RailsConfig.from_content(
        """
        define user express greeting
          "hey"
          "hello"

        define flow greeting
          user express greeting
          do express welcome

        define subflow express welcome
          bot express greeting
          bot offer to help

        define bot express greeting
          "Hello there!"

        define bot offer to help
          "How can I help you today?"
        """
    )

    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
        ],
    )

    chat >> "Hello!"
    chat << "Hello there!\nHow can I help you today?"


def test_two_consecutive_calls():
    config = RailsConfig.from_content(
        """
        define user express greeting
          "hey"
          "hello"

        define flow greeting
          user express greeting
          do express welcome
          do offer to help

        define subflow express welcome
          bot express greeting
          bot offer to help

        define subflow offer to help
          bot offer to help
          bot ask if ok

        define bot express greeting
          "Hello there!"

        define bot offer to help
          "How can I help you today?"

        define bot ask if ok
          "Is this ok?"
        """
    )

    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
        ],
    )

    chat >> "Hello!"
    (chat << "Hello there!\nHow can I help you today?\nHow can I help you today?\nIs this ok?")


def test_subflow_that_exists_immediately():
    config = RailsConfig.from_content(
        """
        define user express greeting
          "hey"
          "hello"

        define flow greeting
          user express greeting
          do check auth
          bot express greeting

        define subflow check auth
          if $auth
            bot you are authenticated

        define bot express greeting
          "Hello there!"

        define bot offer to help
          "How can I help you today?"
        """
    )

    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
        ],
    )

    chat >> "Hello!"
    chat << "Hello there!"


def test_subflow_edge_case_multiple_subflows_exit_immediately():
    config = RailsConfig.from_content(
        """
        define user express greeting
          "hey"
          "hello"

        define flow greeting
          user express greeting
          do check auth
          do check auth_2
          do check auth_3
          bot express greeting

        define subflow check auth
          if $auth
            bot you are authenticated

        define subflow check auth_2
          if $auth
            bot you are authenticated

        define subflow check auth_3
          if $auth
            bot you are authenticated

        define bot express greeting
          "Hello there!"

        define bot offer to help
          "How can I help you today?"
        """
    )

    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
        ],
    )

    chat >> "Hello!"
    chat << "Hello there!"


def test_subflow_that_takes_over():
    config = RailsConfig.from_content(
        """
        define user express greeting
          "hey"
          "hello"

        define user inform name
          "John"

        define flow greeting
          user express greeting
          do check auth
          bot express greeting

        define subflow check auth
          if not $auth
            bot ask name
            user inform name

        define bot express greeting
          "Hello there!"

        define bot ask name
          "What is your name?"
        """
    )

    chat = TestChat(
        config,
        llm_completions=["  express greeting", "  inform name"],
    )

    chat >> "Hello!"
    chat << "What is your name?"
    chat >> "John"
    chat << "Hello there!"
