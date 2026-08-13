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

    assert "Missing a `self_check_input` prompt template" in str(exc_info.value)


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

    assert "Missing a `self_check_output` prompt template" in str(exc_info.value)


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


MASKING_OUTPUT_RAIL = """
models: []
rails:
  config:
    privateai:
      server_endpoint: http://privateai.example/process
      output:
        entities: [NAME]
  output:
    flows:
      - mask pii on output
    streaming:
      enabled: true
      chunk_size: 5
      context_size: {context_size}
      stream_first: {stream_first}
"""

JUDGING_OUTPUT_RAIL = """
models: []
rails:
  config:
    regex_detection:
      output:
        patterns: ['\\d{3}-\\d{2}-\\d{4}']
  output:
    flows:
      - regex check output
    streaming:
      enabled: true
      chunk_size: 5
      context_size: 50
      stream_first: true
"""


@pytest.mark.parametrize(
    "stream_first, context_size, offending",
    [("true", 50, "stream_first"), ("false", 50, "context_size"), ("true", 0, "stream_first")],
    ids=["stream-first", "context-window", "stream-first-with-zero-context"],
)
def test_streaming_refuses_a_rewrite_it_could_not_apply(stream_first, context_size, offending):
    """A masking output rail is refused where streaming would compute the mask and ship the original."""
    with pytest.raises(ValueError) as exc_info:
        RailsConfig.from_content(
            yaml_content=MASKING_OUTPUT_RAIL.format(stream_first=stream_first, context_size=context_size)
        )

    assert offending in str(exc_info.value)
    assert "mask pii on output" in str(exc_info.value)


def test_streaming_accepts_a_rewrite_it_can_apply():
    """With the judged window equal to the batch being sent, the mask lands on what is still to come."""
    config = RailsConfig.from_content(yaml_content=MASKING_OUTPUT_RAIL.format(stream_first="false", context_size=0))

    assert config.rails.output.streaming.enabled is True
    assert config.rails.output.streaming.context_size == 0


def test_a_masking_rail_without_streaming_is_untouched():
    """The refusal is about streaming, not about masking."""
    config = RailsConfig.from_content(
        yaml_content="""
        models: []
        rails:
          config:
            privateai:
              server_endpoint: http://privateai.example/process
              output:
                entities: [NAME]
          output:
            flows:
              - mask pii on output
        """
    )

    assert config.rails.output.flows == ["mask pii on output"]


def test_a_judging_rail_keeps_stream_first():
    """Nothing changes for the streaming configs that shipped before rewrites existed."""
    config = RailsConfig.from_content(yaml_content=JUDGING_OUTPUT_RAIL)

    assert config.rails.output.streaming.stream_first is True
    assert config.rails.output.streaming.context_size == 50


def test_streaming_ignores_a_flow_the_surface_parser_rejects():
    """A flow this validator cannot parse is left to the validators that own flow names.

    Reached directly rather than through a config, because a config carrying such a flow is
    refused by an earlier check and would never arrive here.
    """
    values = {
        "rails": {
            "output": {
                "flows": ["$model=orphaned"],
                "streaming": {"enabled": True, "stream_first": True},
            }
        }
    }

    assert RailsConfig.check_streaming_can_apply_output_rewrites(values) is values


def test_streaming_ignores_a_flow_the_catalog_does_not_describe():
    """A custom or Colang-defined output rail declares no transform target, so nothing is refused."""
    values = {
        "rails": {
            "output": {
                "flows": ["my custom output rail"],
                "streaming": {"enabled": True, "stream_first": True},
            }
        }
    }

    assert RailsConfig.check_streaming_can_apply_output_rewrites(values) is values
