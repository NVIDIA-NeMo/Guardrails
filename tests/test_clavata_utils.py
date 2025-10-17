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

from nemoguardrails.library.clavata.utils import (
    AttemptsExceededError,
    calculate_exp_delay,
    exponential_backoff,
)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_exponential_backoff_successful_call():
    """Test that the exponential backoff decorator works with a successful call."""

    call_count = 0

    @exponential_backoff(max_attempts=3, initial_delay=0.01)
    async def successful_function():
        nonlocal call_count
        call_count += 1
        return "success"

    result = await successful_function()
    assert result == "success"
    assert call_count == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_exponential_backoff_retry_success():
    """Test that the exponential backoff retries and eventually succeeds."""

    call_count = 0

    @exponential_backoff(max_attempts=3, initial_delay=0.01)
    async def eventually_successful():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("Temporary failure")
        return "success after retries"

    result = await eventually_successful()
    assert result == "success after retries"
    assert call_count == 3


@pytest.mark.unit
@pytest.mark.asyncio
async def test_exponential_backoff_max_retries_exceeded():
    """Test that RetriesExceededError is raised when max_retries is exceeded."""

    call_count = 0

    @exponential_backoff(max_attempts=2, initial_delay=0.01)
    async def always_fails():
        nonlocal call_count
        call_count += 1
        raise ValueError("Failure")

    with pytest.raises(AttemptsExceededError) as excinfo:
        await always_fails()

    assert call_count == 2  # initial call + 1 retry
    assert excinfo.value.attempts == 2
    assert excinfo.value.max_attempts == 2
    assert isinstance(excinfo.value.last_exception, ValueError)


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "retry_exceptions",
    [
        ValueError,
        (ValueError,),
        (ValueError, TypeError),
    ],
    ids=["single_exception", "single_exception_tuple", "multiple_exceptions"],
)
async def test_exponential_backoff_specific_exceptions(retry_exceptions):
    """Test that only specified exceptions trigger retry logic."""

    call_count = 0

    @exponential_backoff(
        max_attempts=3,
        initial_delay=0.01,
        retry_exceptions=retry_exceptions,
    )
    async def raises_different_exception():
        nonlocal call_count
        call_count += 1
        raise KeyError("This exception shouldn't trigger retries")

    with pytest.raises(KeyError):
        await raises_different_exception()

    assert call_count == 1  # No retries since KeyError is not in retry_exceptions


@pytest.mark.unit
@pytest.mark.asyncio
async def test_exponential_backoff_permanent_failure_handler():
    """Test that on_permanent_failure handler is called when retries are exhausted."""

    call_count = 0
    handler_called = False

    async def failure_handler(attempts, exception):
        nonlocal handler_called
        handler_called = True
        assert attempts == 2
        assert isinstance(exception, AttemptsExceededError)
        return "handled failure"

    @exponential_backoff(
        max_attempts=2,
        initial_delay=0.01,
        on_permanent_failure=failure_handler,
    )
    async def always_fails():
        nonlocal call_count
        call_count += 1
        raise ValueError("Failure")

    result = await always_fails()
    assert result == "handled failure"
    assert call_count == 2  # initial call + 1 retry
    assert handler_called is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_exponential_backoff_permanent_failure_custom_exception():
    """Test that on_permanent_failure can return a custom exception to be raised."""

    class CustomError(Exception):
        pass

    async def failure_handler(attempts, exception):
        return CustomError("Custom failure")

    @exponential_backoff(
        max_attempts=2,
        initial_delay=0.01,
        on_permanent_failure=failure_handler,
    )
    async def always_fails():
        raise ValueError("Failure")

    with pytest.raises(CustomError) as excinfo:
        await always_fails()

    assert str(excinfo.value) == "Custom failure"


@pytest.mark.unit
@pytest.mark.parametrize(
    ["retries", "expected_delay", "initial_delay", "max_delay", "jitter"],
    [
        (0, 1, 1, 10, False),  # 1 * 1^0 = 1
        (1, 2, 1, 10, False),  # 1 * 1^1 = 2
        (2, 4, 1, 10, False),  # 1 * 1^2 = 4
        (3, 8, 1, 10, False),  # 1 * 1^3 = 8
        (4, 10, 1, 10, False),  # 1 * 1^4 = 10
    ],
    ids=[
        "first_attempt",
        "second_attempt",
        "third_attempt",
        "fourth_attempt",
        "fifth_attempt",
    ],
)
def test_calculate_exp_delay(retries, expected_delay, initial_delay, max_delay, jitter):
    """Test that the calculate_exp_delay function works correctly."""

    assert (
        calculate_exp_delay(retries, initial_delay, max_delay, jitter) == expected_delay
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ["retries", "expected_delay", "initial_delay", "max_delay"],
    [
        (0, 1, 1, 10),
        (1, 2, 1, 10),
        (2, 4, 1, 10),
        (3, 8, 1, 10),
        (4, 10, 1, 10),
    ],
    ids=[
        "first_attempt",
        "second_attempt",
        "third_attempt",
        "fourth_attempt",
        "fifth_attempt",
    ],
)
def test_calculate_exp_delay_jitter(retries, expected_delay, initial_delay, max_delay):
    """Test that the calculate_exp_delay function works correctly with jitter."""

    assert (
        0.0
        <= calculate_exp_delay(retries, initial_delay, max_delay, True)
        <= expected_delay
    )


# TESTS FOR ADDITIONAL PYDANTIC MODELS USED TO PARSE RESPONSES FROM THE CLAVATA API
