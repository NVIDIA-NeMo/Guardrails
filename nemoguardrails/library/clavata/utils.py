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

import asyncio
import random
from collections.abc import Awaitable, Callable, Iterable
from functools import wraps
from typing import Any, Optional, TypeVar, Union

# Import ParamSpec from typing_extensions for Python 3.9 compatibility
from typing_extensions import ParamSpec

from .errs import ClavataPluginTypeError


class AttemptsExceededError(Exception):
    """Exception raised when the maximum number of retries is exceeded."""

    attempts: int
    max_attempts: int
    last_exception: Optional[Exception]

    def __init__(self, attempts: int, max_attempts: int, last_exception: Optional[Exception]):
        self.attempts = attempts
        self.max_attempts = max_attempts
        self.last_exception = last_exception
        super().__init__(
            f"Maximum number of attempts ({max_attempts}) exceeded after {attempts} attempts."
            f"Last exception: {last_exception}"
        )


def calculate_exp_delay(
    retries: int,
    initial_delay: float,
    max_delay: float,
    jitter: bool,
) -> float:
    """
    Handles calculation of the delay for a specific attempt. Note that we specifically
    ask for the number of retries, not the number of attempts, because the first attempt
    is the initial call and we want the first delay to be raised to the power of 0.

    Using "retries" instead of "attempts" makes it clearer what the input is, even
    though a value called "attempts" is being passed in below.

    Because this is an exponential backoff, the factor of increase is always 2.

    Args:
        retries: The number of retries made so far.
        initial_delay: The initial delay.
        max_delay: The maximum delay.
        jitter: Whether to apply jitter to the delay. We use a full-jitter approach.
    """
    delay = min(
        initial_delay * (2**retries),
        max_delay,
    )
    if jitter:
        delay = random.uniform(0, delay)
    return delay


ReturnT = TypeVar("ReturnT")
P = ParamSpec("P")


def exponential_backoff(
    *,
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 10.0,
    jitter: bool = True,  # Set to False to disable jitter
    retry_exceptions: Union[type[Exception], Iterable[type[Exception]]] = Exception,
    on_permanent_failure: Optional[Callable[[int, Exception], Awaitable[Any]]] = None,
):
    """Exponential backoff retry mechanism."""

    # Ensure retry_exceptions is a tuple of exceptions
    retry_exceptions = (retry_exceptions,) if isinstance(retry_exceptions, type) else tuple(retry_exceptions)

    # Sanity check, make sure the types in the retry_exceptions are all exceptions
    if not all(isinstance(e, type) and issubclass(e, Exception) for e in retry_exceptions):
        raise ClavataPluginTypeError("retry_exceptions must be a tuple of exception types")

    def decorator(
        func: Callable[P, Awaitable[ReturnT]],
    ) -> Callable[P, Awaitable[ReturnT]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> ReturnT:
            attempts = 0
            last_exception: Optional[Exception] = None
            while attempts < max_attempts:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:  # noqa: BLE001
                    # If the exception is not in the list of retry exceptions, raise it rather than retrying
                    last_exception = e
                    if not isinstance(e, retry_exceptions):
                        if on_permanent_failure is None:
                            raise

                        perm_rv = await on_permanent_failure(attempts, e)
                        if isinstance(perm_rv, Exception):
                            raise perm_rv from e
                        return perm_rv

                    # We want to calculate the delay before incrementing because we want the first
                    # delay to be exactly the initial delay
                    delay = calculate_exp_delay(attempts, initial_delay, max_delay, jitter)
                    await asyncio.sleep(delay)
                    attempts += 1

            # Max retries exceeded, raise or if a custom handler is provided, call it and then decide what to do
            retried_exc = AttemptsExceededError(attempts, max_attempts, last_exception)
            if on_permanent_failure is None:
                raise retried_exc
            perm_rv = await on_permanent_failure(attempts, retried_exc)
            if isinstance(perm_rv, Exception):
                raise perm_rv
            return perm_rv

        return wrapper

    return decorator
