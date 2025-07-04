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

"""Tests for the parallel output rails streaming functionality."""

import asyncio
import json
import time
from json.decoder import JSONDecodeError

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.actions import action
from tests.utils import TestChat


@pytest.fixture
def parallel_output_rails_streaming_config():
    """Config for testing parallel output rails with streaming enabled and multiple flows"""

    return RailsConfig.from_content(
        config={
            "models": [],
            "rails": {
                "output": {
                    "parallel": True,
                    "flows": [
                        "self check output safety",
                        "self check output compliance",
                        "self check output quality",
                    ],
                    "streaming": {
                        "enabled": True,
                        "chunk_size": 4,
                        "context_size": 2,
                        "stream_first": False,
                    },
                }
            },
            "streaming": False,
            "prompts": [
                {"task": "self_check_output", "content": "Check: {{ bot_response }}"},
            ],
        },
        colang_content="""
        define user express greeting
          "hi"

        define flow
          user express greeting
          bot tell joke

        define subflow self check output safety
          $allowed = execute self_check_output_safety
          if not $allowed
            bot refuse to respond
            stop

        define subflow self check output compliance
          $allowed = execute self_check_output_compliance
          if not $allowed
            bot refuse to respond
            stop

        define subflow self check output quality
          $allowed = execute self_check_output_quality
          if not $allowed
            bot refuse to respond
            stop
        """,
    )


@pytest.fixture
def parallel_output_rails_streaming_single_flow_config():
    """Config for testing parallel output rails with single flow"""

    return RailsConfig.from_content(
        config={
            "models": [],
            "rails": {
                "output": {
                    "parallel": True,
                    "flows": ["self check output"],
                    "streaming": {
                        "enabled": True,
                        "chunk_size": 4,
                        "context_size": 2,
                        "stream_first": False,
                    },
                }
            },
            "streaming": False,
            "prompts": [
                {"task": "self_check_output", "content": "Check: {{ bot_response }}"},
            ],
        },
        colang_content="""
        define user express greeting
          "hi"

        define flow
          user express greeting
          bot tell joke

        define subflow self check output
          execute self_check_output
        """,
    )


@pytest.fixture
def parallel_output_rails_default_config():
    """Config for testing parallel output rails with default streaming settings"""

    return RailsConfig.from_content(
        config={
            "models": [],
            "rails": {
                "output": {
                    "parallel": True,
                    "flows": [
                        "self check output safety",
                        "self check output compliance",
                    ],
                }
            },
            "streaming": True,
            "prompts": [
                {"task": "self_check_output", "content": "Check: {{ bot_response }}"},
            ],
        },
        colang_content="""
        define user express greeting
          "hi"

        define flow
          user express greeting
          bot tell joke

        define subflow self check output safety
          execute self_check_output_safety

        define subflow self check output compliance
          execute self_check_output_compliance
        """,
    )


@action(is_system_action=True)
def self_check_output_safety(context=None, **params):
    """Safety check that blocks content containing UNSAFE keyword."""
    if context and context.get("bot_message"):
        bot_message_chunk = context.get("bot_message")
        if "UNSAFE" in bot_message_chunk:
            return False
    return True


@action(is_system_action=True)
def self_check_output_compliance(context=None, **params):
    """Compliance check that blocks content containing VIOLATION keyword."""
    if context and context.get("bot_message"):
        bot_message_chunk = context.get("bot_message")
        if "VIOLATION" in bot_message_chunk:
            return False
    return True


@action(is_system_action=True)
def self_check_output_quality(context=None, **params):
    """Quality check that blocks content containing LOWQUALITY keyword."""
    if context and context.get("bot_message"):
        bot_message_chunk = context.get("bot_message")
        if "LOWQUALITY" in bot_message_chunk:
            return False
    return True


@action(is_system_action=True)
def self_check_output(context=None, **params):
    """Generic check that blocks content containing BLOCK keyword."""
    if context and context.get("bot_message"):
        bot_message_chunk = context.get("bot_message")
        if "BLOCK" in bot_message_chunk:
            return False
    return True


@action(is_system_action=True, output_mapping=lambda result: not result)
async def slow_self_check_output_safety(**params):
    """Slow safety check for timing tests."""
    await asyncio.sleep(0.1)
    return self_check_output_safety(**params)


@action(is_system_action=True, output_mapping=lambda result: not result)
async def slow_self_check_output_compliance(**params):
    """Slow compliance check for timing tests."""
    await asyncio.sleep(0.1)
    return self_check_output_compliance(**params)


@action(is_system_action=True, output_mapping=lambda result: not result)
async def slow_self_check_output_quality(**params):
    """Slow quality check for timing tests."""
    await asyncio.sleep(0.1)
    return self_check_output_quality(**params)


async def run_parallel_self_check_test(config, llm_completions, register_actions=True):
    """Helper function to run parallel self check test with the given config and llm completions"""

    chat = TestChat(
        config,
        llm_completions=llm_completions,
        streaming=True,
    )

    if register_actions:
        chat.app.register_action(self_check_output_safety)
        chat.app.register_action(self_check_output_compliance)
        chat.app.register_action(self_check_output_quality)
        chat.app.register_action(self_check_output)

    chunks = []
    async for chunk in chat.app.stream_async(
        messages=[{"role": "user", "content": "Hi!"}]
    ):
        chunks.append(chunk)

    return chunks


@pytest.mark.asyncio
async def test_parallel_streaming_output_rails_allowed(
    parallel_output_rails_streaming_config,
):
    """Tests that parallel output rails allow content when no blocking keywords are present"""

    llm_completions = [
        " bot express insult",
        '  "Hi, how are you doing?"',
        '  "This is a safe and compliant high quality joke that should pass all checks."',
    ]

    chunks = await run_parallel_self_check_test(
        parallel_output_rails_streaming_config, llm_completions
    )

    # should receive all chunks without blocking
    response = "".join(chunks)
    assert len(response) > 0
    assert len(chunks) > 1
    assert "This is a safe" in response
    assert "compliant high quality" in response

    error_chunks = [chunk for chunk in chunks if chunk.startswith('{"error":')]
    assert len(error_chunks) == 0

    await asyncio.gather(*asyncio.all_tasks() - {asyncio.current_task()})


@pytest.mark.asyncio
async def test_parallel_streaming_output_rails_blocked_by_safety(
    parallel_output_rails_streaming_config,
):
    """Tests that parallel output rails block content when safety rail detects UNSAFE keyword"""

    llm_completions = [
        '  express greeting\nbot express greeting\n  "Hi, how are you doing?"',
        '  "This is an UNSAFE joke that should be blocked by safety check."',
    ]

    chunks = await run_parallel_self_check_test(
        parallel_output_rails_streaming_config, llm_completions
    )

    expected_error = {
        "error": {
            "message": "Blocked by self check output safety rails.",
            "type": "guardrails_violation",
            "param": "self check output safety",
            "code": "content_blocked",
        }
    }

    error_found = False
    for chunk in chunks:
        try:
            parsed = json.loads(chunk)
            if "error" in parsed and parsed == expected_error:
                error_found = True
                break
        except JSONDecodeError:
            continue

    assert error_found, f"Expected error not found in chunks: {chunks}"

    await asyncio.gather(*asyncio.all_tasks() - {asyncio.current_task()})


@pytest.mark.asyncio
async def test_parallel_streaming_output_rails_blocked_by_compliance(
    parallel_output_rails_streaming_config,
):
    """Tests that parallel output rails block content when compliance rail detects VIOLATION keyword"""

    llm_completions = [
        '  express greeting\nbot express greeting\n  "Hi, how are you doing?"',
        '  "This joke contains a policy VIOLATION and should be blocked."',
    ]

    chunks = await run_parallel_self_check_test(
        parallel_output_rails_streaming_config, llm_completions
    )

    expected_error = {
        "error": {
            "message": "Blocked by self check output compliance rails.",
            "type": "guardrails_violation",
            "param": "self check output compliance",
            "code": "content_blocked",
        }
    }

    error_found = False
    for chunk in chunks:
        try:
            parsed = json.loads(chunk)
            if "error" in parsed and parsed == expected_error:
                error_found = True
                break
        except JSONDecodeError:
            continue

    assert error_found, f"Expected error not found in chunks: {chunks}"

    await asyncio.gather(*asyncio.all_tasks() - {asyncio.current_task()})


@pytest.mark.asyncio
async def test_parallel_streaming_output_rails_blocked_by_quality(
    parallel_output_rails_streaming_config,
):
    """Tests that parallel output rails block content when quality rail detects LOWQUALITY keyword"""

    llm_completions = [
        '  express greeting\nbot express greeting\n  "Hi, how are you doing?"',
        '  "This is a LOWQUALITY joke that should be blocked by quality check."',
    ]

    chunks = await run_parallel_self_check_test(
        parallel_output_rails_streaming_config, llm_completions
    )

    expected_error = {
        "error": {
            "message": "Blocked by self check output quality rails.",
            "type": "guardrails_violation",
            "param": "self check output quality",
            "code": "content_blocked",
        }
    }

    error_found = False
    for chunk in chunks:
        try:
            parsed = json.loads(chunk)
            if "error" in parsed and parsed == expected_error:
                error_found = True
                break
        except JSONDecodeError:
            continue

    assert error_found, f"Expected error not found in chunks: {chunks}"

    await asyncio.gather(*asyncio.all_tasks() - {asyncio.current_task()})


@pytest.mark.asyncio
async def test_parallel_streaming_output_rails_blocked_at_start(
    parallel_output_rails_streaming_single_flow_config,
):
    """Tests parallel blocking when BLOCK keyword appears at the very beginning"""

    llm_completions = [
        '  express greeting\nbot express greeting\n  "Hi, how are you doing?"',
        '  "[BLOCK] This should be blocked immediately at the start."',
    ]

    chunks = await run_parallel_self_check_test(
        parallel_output_rails_streaming_single_flow_config, llm_completions
    )

    expected_error = {
        "error": {
            "message": "Blocked by self check output rails.",
            "type": "guardrails_violation",
            "param": "self check output",
            "code": "content_blocked",
        }
    }

    # should be blocked immediately with only one error chunk
    assert len(chunks) == 1
    assert json.loads(chunks[0]) == expected_error

    await asyncio.gather(*asyncio.all_tasks() - {asyncio.current_task()})


@pytest.mark.asyncio
async def test_parallel_streaming_output_rails_multiple_blocking_keywords(
    parallel_output_rails_streaming_config,
):
    """Tests parallel rails when multiple blocking keywords are present - should block on first detected"""

    llm_completions = [
        '  express greeting\nbot express greeting\n  "Hi, how are you doing?"',
        '  "This contains both UNSAFE content and a VIOLATION which is also LOWQUALITY."',
    ]

    chunks = await run_parallel_self_check_test(
        parallel_output_rails_streaming_config, llm_completions
    )

    # should be blocked by one of the rails (whichever detects first in parallel execution)
    error_chunks = []
    for chunk in chunks:
        try:
            parsed = json.loads(chunk)
            if "error" in parsed:
                error_chunks.append(parsed)
        except JSONDecodeError:
            continue

    assert (
        len(error_chunks) == 1
    ), f"Expected exactly one error chunk, got {len(error_chunks)}"

    error = error_chunks[0]
    assert error["error"]["type"] == "guardrails_violation"
    assert error["error"]["code"] == "content_blocked"
    assert "Blocked by" in error["error"]["message"]

    # should be blocked by one of the three rails
    blocked_by_options = [
        "self check output safety",
        "self check output compliance",
        "self check output quality",
    ]
    assert error["error"]["param"] in blocked_by_options

    await asyncio.gather(*asyncio.all_tasks() - {asyncio.current_task()})


@pytest.mark.asyncio
async def test_parallel_streaming_output_rails_performance_benefits():
    """Tests that parallel rails execution provides performance benefits over sequential"""

    parallel_config = RailsConfig.from_content(
        config={
            "models": [],
            "rails": {
                "output": {
                    "parallel": True,
                    "flows": [
                        "slow self check output safety",
                        "slow self check output compliance",
                        "slow self check output quality",
                    ],
                    "streaming": {
                        "enabled": True,
                        "chunk_size": 4,
                        "context_size": 2,
                    },
                }
            },
            "streaming": False,
        },
        colang_content="""
        define user express greeting
          "hi"
        define flow
          user express greeting
          bot tell joke

        define subflow slow self check output safety
          execute slow_self_check_output_safety

        define subflow slow self check output compliance
          execute slow_self_check_output_compliance

        define subflow slow self check output quality
          execute slow_self_check_output_quality
        """,
    )

    sequential_config = RailsConfig.from_content(
        config={
            "models": [],
            "rails": {
                "output": {
                    "parallel": False,
                    "flows": [
                        "slow self check output safety",
                        "slow self check output compliance",
                        "slow self check output quality",
                    ],
                    "streaming": {
                        "enabled": True,
                        "chunk_size": 4,
                        "context_size": 2,
                    },
                }
            },
            "streaming": False,
        },
        colang_content="""
        define user express greeting
          "hi"
        define flow
          user express greeting
          bot tell joke

        define subflow slow self check output safety
          execute slow_self_check_output_safety

        define subflow slow self check output compliance
          execute slow_self_check_output_compliance

        define subflow slow self check output quality
          execute slow_self_check_output_quality
        """,
    )

    llm_completions = [
        '  express greeting\nbot express greeting\n  "Hi, how are you doing?"',
        '  "This is a safe and compliant high quality response for timing tests."',
    ]

    parallel_chat = TestChat(
        parallel_config, llm_completions=llm_completions, streaming=True
    )
    parallel_chat.app.register_action(slow_self_check_output_safety)
    parallel_chat.app.register_action(slow_self_check_output_compliance)
    parallel_chat.app.register_action(slow_self_check_output_quality)

    start_time = time.time()
    parallel_chunks = []
    async for chunk in parallel_chat.app.stream_async(
        messages=[{"role": "user", "content": "Hi!"}]
    ):
        parallel_chunks.append(chunk)
    parallel_time = time.time() - start_time

    sequential_chat = TestChat(
        sequential_config, llm_completions=llm_completions, streaming=True
    )
    sequential_chat.app.register_action(slow_self_check_output_safety)
    sequential_chat.app.register_action(slow_self_check_output_compliance)
    sequential_chat.app.register_action(slow_self_check_output_quality)

    start_time = time.time()
    sequential_chunks = []
    async for chunk in sequential_chat.app.stream_async(
        messages=[{"role": "user", "content": "Hi!"}]
    ):
        sequential_chunks.append(chunk)
    sequential_time = time.time() - start_time

    # Parallel should be faster than sequential (allowing some margin for test variability)
    print(
        f"Parallel time: {parallel_time:.2f}s, Sequential time: {sequential_time:.2f}s"
    )

    # with 3 rails each taking ~0.1 s sequential should take ~0.3 s per chunk, parallel should be closer to 0.1s
    # we allow some margin for test execution overhead
    assert parallel_time < sequential_time * 0.8, (
        f"Parallel execution ({parallel_time:.2f}s) should be significantly faster than "
        f"sequential execution ({sequential_time:.2f}s)"
    )

    parallel_response = "".join(parallel_chunks)
    sequential_response = "".join(sequential_chunks)
    assert parallel_response == sequential_response

    await asyncio.gather(*asyncio.all_tasks() - {asyncio.current_task()})


@pytest.mark.asyncio
async def test_parallel_streaming_output_rails_default_config_behavior(
    parallel_output_rails_default_config,
):
    """Tests parallel output rails with default streaming configuration"""

    llm_completions = [
        '  express greeting\nbot express greeting\n  "Hi, how are you doing?"',
        '  "This is a test message with default streaming config."',
    ]

    chunks = await run_parallel_self_check_test(
        parallel_output_rails_default_config, llm_completions
    )

    response = "".join(chunks)
    assert len(response) > 0
    assert len(chunks) > 0
    assert "test message" in response

    await asyncio.gather(*asyncio.all_tasks() - {asyncio.current_task()})


@pytest.mark.asyncio
async def test_parallel_streaming_output_rails_error_handling():
    """Tests error handling in parallel streaming when rails fail"""

    @action(is_system_action=True, output_mapping=lambda result: not result)
    def failing_rail(**params):
        raise Exception("Simulated rail failure")

    @action(is_system_action=True, output_mapping=lambda result: not result)
    def working_rail(**params):
        return True

    config = RailsConfig.from_content(
        config={
            "models": [],
            "rails": {
                "output": {
                    "parallel": True,
                    "flows": ["failing rail", "working rail"],
                    "streaming": {
                        "enabled": True,
                        "chunk_size": 4,
                        "context_size": 2,
                    },
                }
            },
            "streaming": False,
        },
        colang_content="""
        define user express greeting
          "hi"
        define flow
          user express greeting
          bot tell joke

        define subflow failing rail
          execute failing_rail

        define subflow working rail
          execute working_rail
        """,
    )

    llm_completions = [
        '  express greeting\nbot express greeting\n  "Hi, how are you doing?"',
        '  "This message should still be processed despite one rail failing."',
    ]

    chat = TestChat(config, llm_completions=llm_completions, streaming=True)
    chat.app.register_action(failing_rail)
    chat.app.register_action(working_rail)

    chunks = []
    async for chunk in chat.app.stream_async(
        messages=[{"role": "user", "content": "Hi!"}]
    ):
        chunks.append(chunk)

    # should continue processing despite one rail failing
    response = "".join(chunks)
    assert len(response) > 0
    assert "should still be processed" in response

    await asyncio.gather(*asyncio.all_tasks() - {asyncio.current_task()})


@pytest.mark.asyncio
async def test_parallel_streaming_output_rails_stream_first_enabled():
    """Tests parallel streaming with stream_first option enabled"""

    config = RailsConfig.from_content(
        config={
            "models": [],
            "rails": {
                "output": {
                    "parallel": True,
                    "flows": ["self check output"],
                    "streaming": {
                        "enabled": True,
                        "chunk_size": 4,
                        "context_size": 2,
                        "stream_first": True,
                    },
                }
            },
            "streaming": False,
            "prompts": [
                {"task": "self_check_output", "content": "Check: {{ bot_response }}"},
            ],
        },
        colang_content="""
        define user express greeting
          "hi"
        define flow
          user express greeting
          bot tell joke

        define subflow self check output
          execute self_check_output
        """,
    )

    llm_completions = [
        '  express greeting\nbot express greeting\n  "Hi, how are you doing?"',
        '  "This is a test message for stream first functionality."',
    ]

    chunks = await run_parallel_self_check_test(config, llm_completions)

    assert len(chunks) > 1
    response = "".join(chunks)
    assert "test message" in response

    await asyncio.gather(*asyncio.all_tasks() - {asyncio.current_task()})


@pytest.mark.asyncio
async def test_parallel_streaming_output_rails_large_chunk_processing():
    """Tests parallel streaming with larger chunks to ensure proper processing"""

    config = RailsConfig.from_content(
        config={
            "models": [],
            "rails": {
                "output": {
                    "parallel": True,
                    "flows": [
                        "self check output safety",
                        "self check output compliance",
                    ],
                    "streaming": {
                        "enabled": True,
                        "chunk_size": 10,
                        "context_size": 3,
                    },
                }
            },
            "streaming": False,
        },
        colang_content="""
        define user express greeting
          "hi"
        define flow
          user express greeting
          bot tell joke

        define subflow self check output safety
          execute self_check_output_safety

        define subflow self check output compliance
          execute self_check_output_compliance
        """,
    )

    llm_completions = [
        '  express greeting\nbot express greeting\n  "Hi, how are you doing?"',
        '  "This is a much longer response that will be processed in larger chunks to test the parallel rail processing functionality with bigger chunk sizes and ensure that everything works correctly."',
    ]

    chunks = await run_parallel_self_check_test(config, llm_completions)

    response = "".join(chunks)
    assert len(response) > 50
    assert "much longer response" in response
    assert "parallel rail processing" in response

    await asyncio.gather(*asyncio.all_tasks() - {asyncio.current_task()})


@pytest.mark.asyncio
async def test_sequential_vs_parallel_streaming_output_rails_comparison():
    """Direct comparison test between sequential and parallel streaming output rails.

    This test demonstrates the differences between sequential and parallel execution
    using identical content and configurations, except for the parallel flag.
    """

    @action(is_system_action=True, output_mapping=lambda result: not result)
    def test_self_check_output(context=None, **params):
        """Test check that blocks content containing BLOCK keyword."""

        if context and context.get("bot_message"):
            bot_message_chunk = context.get("bot_message")
            if "BLOCK" in bot_message_chunk:
                return False
        return True

    base_config = {
        "models": [],
        "rails": {
            "output": {
                "flows": ["test self check output"],
                "streaming": {
                    "enabled": True,
                    "chunk_size": 4,
                    "context_size": 2,
                    "stream_first": False,
                },
            }
        },
        "streaming": False,
    }

    colang_content = """
    define user express greeting
      "hi"

    define flow
      user express greeting
      bot tell joke

    define subflow test self check output
      execute test_self_check_output
    """

    sequential_config = RailsConfig.from_content(
        config=base_config,
        colang_content=colang_content,
    )

    parallel_config_dict = base_config.copy()
    parallel_config_dict["rails"]["output"]["parallel"] = True

    parallel_config = RailsConfig.from_content(
        config=parallel_config_dict,
        colang_content=colang_content,
    )

    llm_completions = [
        '  express greeting\nbot express greeting\n  "Hi, how are you doing?"',
        '  "This is a safe and compliant high quality joke that should pass all checks."',
    ]

    sequential_chat = TestChat(
        sequential_config,
        llm_completions=llm_completions,
        streaming=True,
    )
    sequential_chat.app.register_action(test_self_check_output)

    parallel_chat = TestChat(
        parallel_config,
        llm_completions=llm_completions,
        streaming=True,
    )
    parallel_chat.app.register_action(test_self_check_output)

    import time

    start_time = time.time()
    sequential_chunks = []
    async for chunk in sequential_chat.app.stream_async(
        messages=[{"role": "user", "content": "Hi!"}]
    ):
        sequential_chunks.append(chunk)
    sequential_time = time.time() - start_time

    start_time = time.time()
    parallel_chunks = []
    async for chunk in parallel_chat.app.stream_async(
        messages=[{"role": "user", "content": "Hi!"}]
    ):
        parallel_chunks.append(chunk)
    parallel_time = time.time() - start_time

    # both should produce the same successful output
    sequential_response = "".join(sequential_chunks)
    parallel_response = "".join(parallel_chunks)

    assert len(sequential_response) > 0
    assert len(parallel_response) > 0
    assert "This is a safe" in sequential_response
    assert "This is a safe" in parallel_response
    assert "compliant high quality" in sequential_response
    assert "compliant high quality" in parallel_response

    # neither should have error chunks
    sequential_error_chunks = [
        chunk for chunk in sequential_chunks if chunk.startswith('{"error":')
    ]
    parallel_error_chunks = [
        chunk for chunk in parallel_chunks if chunk.startswith('{"error":')
    ]

    assert (
        len(sequential_error_chunks) == 0
    ), f"Sequential had errors: {sequential_error_chunks}"
    assert (
        len(parallel_error_chunks) == 0
    ), f"Parallel had errors: {parallel_error_chunks}"

    assert sequential_response == parallel_response, (
        f"Sequential and parallel should produce identical content:\n"
        f"Sequential: {sequential_response}\n"
        f"Parallel: {parallel_response}"
    )

    # log timing comparison (parallel should be faster or similar for single rail)
    print(f"\nTiming Comparison:")
    print(f"Sequential: {sequential_time:.4f}s")
    print(f"Parallel: {parallel_time:.4f}s")
    print(f"Speedup: {sequential_time / parallel_time:.2f}x")

    await asyncio.gather(*asyncio.all_tasks() - {asyncio.current_task()})


@pytest.mark.asyncio
async def test_sequential_vs_parallel_streaming_blocking_comparison():
    """Test that both sequential and parallel handle blocking scenarios identically"""

    @action(is_system_action=True, output_mapping=lambda result: not result)
    def test_self_check_output_blocking(context=None, **params):
        """Test check that blocks content containing BLOCK keyword."""
        if context and context.get("bot_message"):
            bot_message_chunk = context.get("bot_message")
            if "BLOCK" in bot_message_chunk:
                return False
        return True

    base_config = {
        "models": [],
        "rails": {
            "output": {
                "flows": ["test self check output blocking"],
                "streaming": {
                    "enabled": True,
                    "chunk_size": 4,
                    "context_size": 2,
                    "stream_first": False,
                },
            }
        },
        "streaming": False,
    }

    colang_content = """
    define user express greeting
      "hi"

    define flow
      user express greeting
      bot tell joke

    define subflow test self check output blocking
      execute test_self_check_output_blocking
    """

    sequential_config = RailsConfig.from_content(
        config=base_config, colang_content=colang_content
    )

    parallel_config_dict = base_config.copy()
    parallel_config_dict["rails"]["output"]["parallel"] = True
    parallel_config = RailsConfig.from_content(
        config=parallel_config_dict, colang_content=colang_content
    )

    llm_completions = [
        '  express greeting\nbot express greeting\n  "Hi, how are you doing?"',
        '  "This contains a BLOCK keyword that should be blocked."',
    ]

    sequential_chat = TestChat(
        sequential_config,
        llm_completions=llm_completions,
        streaming=True,
    )
    sequential_chat.app.register_action(test_self_check_output_blocking)

    parallel_chat = TestChat(
        parallel_config,
        llm_completions=llm_completions,
        streaming=True,
    )
    parallel_chat.app.register_action(test_self_check_output_blocking)

    sequential_chunks = []
    async for chunk in sequential_chat.app.stream_async(
        messages=[{"role": "user", "content": "Hi!"}]
    ):
        sequential_chunks.append(chunk)

    parallel_chunks = []
    async for chunk in parallel_chat.app.stream_async(
        messages=[{"role": "user", "content": "Hi!"}]
    ):
        parallel_chunks.append(chunk)

    sequential_errors = []
    parallel_errors = []

    for chunk in sequential_chunks:
        try:
            parsed = json.loads(chunk)
            if "error" in parsed:
                sequential_errors.append(parsed)
        except JSONDecodeError:
            continue

    for chunk in parallel_chunks:
        try:
            parsed = json.loads(chunk)
            if "error" in parsed:
                parallel_errors.append(parsed)
        except JSONDecodeError:
            continue

    assert (
        len(sequential_errors) == 1
    ), f"Sequential should have 1 error, got {len(sequential_errors)}"
    assert (
        len(parallel_errors) == 1
    ), f"Parallel should have 1 error, got {len(parallel_errors)}"

    seq_error = sequential_errors[0]
    par_error = parallel_errors[0]

    assert seq_error["error"]["type"] == "guardrails_violation"
    assert par_error["error"]["type"] == "guardrails_violation"
    assert seq_error["error"]["code"] == "content_blocked"
    assert par_error["error"]["code"] == "content_blocked"
    assert "Blocked by" in seq_error["error"]["message"]
    assert "Blocked by" in par_error["error"]["message"]

    await asyncio.gather(*asyncio.all_tasks() - {asyncio.current_task()})


@pytest.mark.asyncio
async def test_parallel_vs_sequential_with_slow_actions():
    """Test that demonstrates real parallel speedup with slow actions"""

    import time

    @action(is_system_action=True, output_mapping=lambda result: not result)
    async def slow_safety_check(context=None, **params):
        """Slow safety check that simulates real processing time."""
        # simulate 100ms of processing
        await asyncio.sleep(0.1)
        if context and context.get("bot_message"):
            bot_message_chunk = context.get("bot_message")
            if "UNSAFE" in bot_message_chunk:
                return False
        return True

    @action(is_system_action=True, output_mapping=lambda result: not result)
    async def slow_compliance_check(context=None, **params):
        """Slow compliance check that simulates real processing time."""
        await asyncio.sleep(0.1)
        if context and context.get("bot_message"):
            bot_message_chunk = context.get("bot_message")
            if "VIOLATION" in bot_message_chunk:
                return False
        return True

    @action(is_system_action=True, output_mapping=lambda result: not result)
    async def slow_quality_check(context=None, **params):
        """Slow quality check that simulates real processing time."""
        await asyncio.sleep(0.1)
        if context and context.get("bot_message"):
            bot_message_chunk = context.get("bot_message")
            if "LOWQUALITY" in bot_message_chunk:
                return False
        return True

    base_config = {
        "models": [],
        "rails": {
            "output": {
                "flows": [
                    "slow safety check",
                    "slow compliance check",
                    "slow quality check",
                ],
                "streaming": {
                    "enabled": True,
                    "chunk_size": 4,
                    "context_size": 2,
                    "stream_first": False,
                },
            }
        },
        "streaming": False,
    }

    colang_content = """
    define user express greeting
      "hi"

    define flow
      user express greeting
      bot tell joke

    define subflow slow safety check
      execute slow_safety_check

    define subflow slow compliance check
      execute slow_compliance_check

    define subflow slow quality check
      execute slow_quality_check
    """

    sequential_config = RailsConfig.from_content(
        config=base_config,
        colang_content=colang_content,
    )

    parallel_config_dict = base_config.copy()
    parallel_config_dict["rails"]["output"]["parallel"] = True

    parallel_config = RailsConfig.from_content(
        config=parallel_config_dict,
        colang_content=colang_content,
    )

    llm_completions = [
        '  express greeting\nbot express greeting\n  "Hi, how are you doing?"',
        '  "This is a safe and compliant high quality joke that should pass all checks."',
    ]

    sequential_chat = TestChat(
        sequential_config,
        llm_completions=llm_completions,
        streaming=True,
    )
    sequential_chat.app.register_action(slow_safety_check)
    sequential_chat.app.register_action(slow_compliance_check)
    sequential_chat.app.register_action(slow_quality_check)

    parallel_chat = TestChat(
        parallel_config,
        llm_completions=llm_completions,
        streaming=True,
    )
    parallel_chat.app.register_action(slow_safety_check)
    parallel_chat.app.register_action(slow_compliance_check)
    parallel_chat.app.register_action(slow_quality_check)

    print(f"\n=== SLOW ACTIONS PERFORMANCE TEST ===")
    print(f"Each action takes 100ms, 3 actions total")
    print(f"Expected: Sequential ~300ms per chunk, Parallel ~100ms per chunk")

    start_time = time.time()
    sequential_chunks = []
    async for chunk in sequential_chat.app.stream_async(
        messages=[{"role": "user", "content": "Hi!"}]
    ):
        sequential_chunks.append(chunk)
    sequential_time = time.time() - start_time

    start_time = time.time()
    parallel_chunks = []
    async for chunk in parallel_chat.app.stream_async(
        messages=[{"role": "user", "content": "Hi!"}]
    ):
        parallel_chunks.append(chunk)
    parallel_time = time.time() - start_time

    sequential_response = "".join(sequential_chunks)
    parallel_response = "".join(parallel_chunks)

    assert len(sequential_response) > 0
    assert len(parallel_response) > 0
    assert "This is a safe" in sequential_response
    assert "This is a safe" in parallel_response

    sequential_error_chunks = [
        chunk for chunk in sequential_chunks if chunk.startswith('{"error":')
    ]
    parallel_error_chunks = [
        chunk for chunk in parallel_chunks if chunk.startswith('{"error":')
    ]

    assert len(sequential_error_chunks) == 0
    assert len(parallel_error_chunks) == 0

    assert sequential_response == parallel_response

    speedup = sequential_time / parallel_time

    print(f"\nSlow Actions Timing Results:")
    print(f"Sequential: {sequential_time:.4f}s")
    print(f"Parallel: {parallel_time:.4f}s")
    print(f"Speedup: {speedup:.2f}x")

    # with slow actions, parallel should be significantly faster
    # we expect at least 1.5x speedup (theoretical max ~3x, but overhead reduces it)
    assert speedup >= 1.5, (
        f"With slow actions, parallel should be at least 1.5x faster than sequential. "
        f"Got speedup of {speedup:.2f}x. Sequential: {sequential_time:.4f}s, Parallel: {parallel_time:.4f}s"
    )

    print(f" Parallel execution achieved {speedup:.2f}x speedup as expected!")

    await asyncio.gather(*asyncio.all_tasks() - {asyncio.current_task()})
