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

import pytest

from nemoguardrails import LLMRails, RailsConfig


def test_input_rail_exists_check():
    with pytest.raises(ValueError) as exc_info:
        config = RailsConfig.from_content(
            yaml_content="""
            rails:
                input:
                    flows:
                        - example input rail
            """,
        )
        LLMRails(config=config)

    assert "`example input rail` does not exist" in str(exc_info.value)


def test_output_rail_exists_check():
    with pytest.raises(ValueError) as exc_info:
        config = RailsConfig.from_content(
            yaml_content="""
            rails:
                output:
                    flows:
                        - example output rail
            """,
        )
        LLMRails(config=config)

    assert "`example output rail` does not exist" in str(exc_info.value)


def test_retrieval_rail_exists_check():
    with pytest.raises(ValueError) as exc_info:
        config = RailsConfig.from_content(
            yaml_content="""
            rails:
                retrieval:
                    flows:
                        - example retrieval rail
            """,
        )
        LLMRails(config=config)

    assert "`example retrieval rail` does not exist" in str(exc_info.value)


def test_self_check_input_prompt_exception():
    with pytest.raises(ValueError) as exc_info:
        config = RailsConfig.from_content(
            yaml_content="""
            rails:
                input:
                    flows:
                        - self check input
            """,
        )
        LLMRails(config=config)

    assert "You must provide a `self_check_input` prompt" in str(exc_info.value)


def test_self_check_output_prompt_exception():
    with pytest.raises(ValueError) as exc_info:
        config = RailsConfig.from_content(
            yaml_content="""
            rails:
                output:
                    flows:
                        - self check output
            """,
        )
        LLMRails(config=config)

    assert "You must provide a `self_check_output` prompt" in str(exc_info.value)


def test_passthrough_and_single_call_incompatibility():
    with pytest.raises(ValueError) as exc_info:
        config = RailsConfig.from_content(
            yaml_content="""
            rails:
                dialog:
                    single_call:
                        enabled: True
            passthrough: True
            """,
        )
        LLMRails(config=config)

    assert "The passthrough mode and the single call dialog" in str(exc_info.value)


# def test_self_check_facts_prompt_exception():
#     with pytest.raises(ValueError) as exc_info:
#         config = RailsConfig.from_content(
#             yaml_content="""
#             rails:
#                 output:
#                     flows:
#                         - self check facts
#             """,
#         )
#         LLMRails(config=config)
#
#     assert "You must provide a `self_check_facts` prompt" in str(exc_info.value)
