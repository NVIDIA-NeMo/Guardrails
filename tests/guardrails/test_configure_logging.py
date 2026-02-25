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

"""Tests for the configure_logging helper in nemoguardrails.guardrails."""

import logging

import pytest

from nemoguardrails.guardrails import configure_logging

@pytest.fixture(autouse=True)
def _clean_logger():
    """Runs before and after each test to revert changes to logging"""
    logger = logging.getLogger("nemoguardrails.guardrails")
    logger.handlers.clear()
    yield
    logger.handlers.clear()


class TestConfigureLogging:

    def test_adds_handler_on_first_call(self):
        logger = configure_logging(logging.INFO)
        assert len(logger.handlers) == 1

    def test_does_not_stack_handlers_on_repeated_calls(self):
        configure_logging(logging.INFO)
        configure_logging(logging.DEBUG)
        configure_logging(logging.WARNING)

        logger = logging.getLogger("nemoguardrails.guardrails")
        assert logger.level == logging.WARNING
        assert len(logger.handlers) == 1

    def test_sets_log_level(self):
        logger = configure_logging(logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_propagate_is_false(self):
        logger = configure_logging(logging.INFO)
        assert logger.propagate is False

    def test_returns_package_logger(self):
        logger = configure_logging()
        assert logger.name == "nemoguardrails.guardrails"
