# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unit tests for the AsyncWorkQueue class.

These tests verify the async work queue implementation including:
- Basic task submission and execution
- Args/kwargs handling
- Queue overflow behavior (reject vs block)
- Concurrency limiting
- Error propagation
- Cancellation handling
- Lifecycle management
"""

import asyncio
from typing import Any

import pytest

from nemoguardrails.guardrails.async_work_queue import AsyncWorkQueue


class TestBasicFunctionality:
    """Tests for basic AsyncWorkQueue functionality."""

    @pytest.mark.asyncio
    async def test_submit_with_args_only(self):
        """Test submitting a task with positional arguments only."""

        async def add(a, b):
            return a + b

        async with AsyncWorkQueue[int](name="test_queue", max_queue_size=10, max_concurrency=2) as queue:
            result = await queue.submit(add, 5, 3)
            assert result == 8

    @pytest.mark.asyncio
    async def test_submit_with_kwargs_only(self):
        """Test submitting a task with keyword arguments only."""

        async def multiply(x, y):
            return x * y

        async with AsyncWorkQueue[int](name="test_queue", max_queue_size=10, max_concurrency=2) as queue:
            result = await queue.submit(multiply, x=4, y=7)
            assert result == 28

    @pytest.mark.asyncio
    async def test_submit_with_args_and_kwargs(self):
        """Test submitting a task with both positional and keyword arguments."""

        async def compute(a, b, c, multiplier=1):
            return (a + b + c) * multiplier

        async with AsyncWorkQueue[int](name="test_queue", max_queue_size=10, max_concurrency=2) as queue:
            result = await queue.submit(compute, 1, 2, 3, multiplier=2)
            assert result == 12

    @pytest.mark.asyncio
    async def test_submit_no_args(self):
        """Test submitting a task with no arguments."""

        async def get_constant():
            return 42

        async with AsyncWorkQueue[int](name="test_queue", max_queue_size=10, max_concurrency=2) as queue:
            result = await queue.submit(get_constant)
            assert result == 42

    @pytest.mark.asyncio
    async def test_submit_with_string_result(self):
        """Test submitting a task that returns a string."""

        async def greet(name):
            return f"Hello, {name}!"

        async with AsyncWorkQueue[str](name="test_queue", max_queue_size=10, max_concurrency=2) as queue:
            result = await queue.submit(greet, "World")
            assert result == "Hello, World!"

    @pytest.mark.asyncio
    async def test_submit_with_complex_return_type(self):
        """Test submitting a task that returns a complex object."""

        async def create_dict(key, value):
            return {"key": key, "value": value, "processed": True}

        async with AsyncWorkQueue[dict](name="test_queue", max_queue_size=10, max_concurrency=2) as queue:
            result = await queue.submit(create_dict, "test_key", 123)
            assert result == {"key": "test_key", "value": 123, "processed": True}

    @pytest.mark.asyncio
    async def test_multiple_sequential_submissions(self):
        """Test submitting multiple tasks sequentially."""

        async def square(x):
            return x * x

        async with AsyncWorkQueue[int](name="test_queue", max_queue_size=10, max_concurrency=2) as queue:
            result1 = await queue.submit(square, 2)
            result2 = await queue.submit(square, 3)
            result3 = await queue.submit(square, 4)

            assert result1 == 4
            assert result2 == 9
            assert result3 == 16


class TestConcurrency:
    """Tests for concurrency control."""

    @pytest.mark.asyncio
    async def test_max_concurrency_limit(self):
        """Test that max_concurrency limits parallel execution."""
        execution_tracker = []
        lock = asyncio.Lock()

        async def track_execution(task_id):
            execution_tracker.append(("start", task_id))
            await asyncio.sleep(0.1)
            execution_tracker.append(("end", task_id))
            return task_id

        async with AsyncWorkQueue[int](name="test_queue", max_queue_size=10, max_concurrency=2) as queue:
            # Submit 4 tasks
            tasks = [asyncio.create_task(queue.submit(track_execution, i)) for i in range(4)]

            results = await asyncio.gather(*tasks)
            assert results == [0, 1, 2, 3]

            # Verify at most 2 tasks were executing concurrently
            concurrent_count = 0
            max_concurrent = 0
            for event_type, _ in execution_tracker:
                if event_type == "start":
                    concurrent_count += 1
                    max_concurrent = max(max_concurrent, concurrent_count)
                else:
                    concurrent_count -= 1

            assert max_concurrent <= 2

    @pytest.mark.asyncio
    async def test_parallel_submission(self):
        """Test submitting multiple tasks in parallel."""

        async def add_delayed(a, b):
            await asyncio.sleep(0.05)
            return a + b

        async with AsyncWorkQueue[int](name="test_queue", max_queue_size=10, max_concurrency=3) as queue:
            # Submit tasks in parallel
            tasks = [asyncio.create_task(queue.submit(add_delayed, i, i)) for i in range(5)]

            results = await asyncio.gather(*tasks)
            assert results == [0, 2, 4, 6, 8]


class TestQueueFull:
    """Tests for queue full behavior."""

    @pytest.mark.asyncio
    async def test_reject_on_full_raises_exception(self):
        """Test that reject_on_full=True raises QueueFull when queue is full."""

        async def slow_task():
            await asyncio.sleep(1)
            return "done"

        async with AsyncWorkQueue[str](
            name="test_queue", max_queue_size=2, max_concurrency=1, reject_on_full=True
        ) as queue:
            # Fill the queue
            task1 = asyncio.create_task(queue.submit(slow_task))
            await asyncio.sleep(0.01)  # Let it get picked up by worker
            task2 = asyncio.create_task(queue.submit(slow_task))
            task3 = asyncio.create_task(queue.submit(slow_task))
            await asyncio.sleep(0.01)

            # Try to submit one more - should raise QueueFull
            with pytest.raises(asyncio.QueueFull):
                await queue.submit(slow_task)

            # Cancel pending tasks to clean up
            for task in [task1, task2, task3]:
                task.cancel()
            await asyncio.gather(task1, task2, task3, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_reject_on_full_false_blocks_until_space(self):
        """Test that reject_on_full=False blocks until queue space is available."""
        results = []

        async def fast_task(value):
            await asyncio.sleep(0.05)
            return value

        async with AsyncWorkQueue[int](
            name="test_queue", max_queue_size=2, max_concurrency=1, reject_on_full=False
        ) as queue:
            # Submit tasks that will block when queue is full
            tasks = [asyncio.create_task(queue.submit(fast_task, i)) for i in range(5)]

            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks)

            # All tasks should complete successfully
            assert results == [0, 1, 2, 3, 4]


class TestErrorHandling:
    """Tests for error handling and exception propagation."""

    @pytest.mark.asyncio
    async def test_exception_propagation(self):
        """Test that exceptions in tasks are properly propagated to caller."""

        async def failing_task():
            raise ValueError("Task failed")

        async with AsyncWorkQueue[Any](name="test_queue", max_queue_size=10, max_concurrency=2) as queue:
            with pytest.raises(ValueError, match="Task failed"):
                await queue.submit(failing_task)

    @pytest.mark.asyncio
    async def test_exception_with_args(self):
        """Test exception propagation with task arguments."""

        async def divide(a, b):
            if b == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            return a / b

        async with AsyncWorkQueue[float](name="test_queue", max_queue_size=10, max_concurrency=2) as queue:
            # Should work fine
            result = await queue.submit(divide, 10, 2)
            assert result == 5.0

            # Should raise exception
            with pytest.raises(ZeroDivisionError, match="Cannot divide by zero"):
                await queue.submit(divide, 10, 0)

    @pytest.mark.asyncio
    async def test_multiple_tasks_with_mixed_results(self):
        """Test that errors in some tasks don't affect others."""

        async def conditional_fail(value):
            if value < 0:
                raise ValueError(f"Negative value: {value}")
            return value * 2

        async with AsyncWorkQueue[int](name="test_queue", max_queue_size=10, max_concurrency=2) as queue:
            # Submit mix of successful and failing tasks
            task1 = asyncio.create_task(queue.submit(conditional_fail, 5))
            task2 = asyncio.create_task(queue.submit(conditional_fail, -3))
            task3 = asyncio.create_task(queue.submit(conditional_fail, 7))

            # Wait for all tasks
            results = await asyncio.gather(task1, task2, task3, return_exceptions=True)

            # Check results
            assert results[0] == 10
            assert isinstance(results[1], ValueError)
            assert results[2] == 14


class TestCancellation:
    """Tests for task cancellation handling."""

    @pytest.mark.asyncio
    async def test_cancel_pending_task(self):
        """Test cancelling a task that hasn't started execution yet."""

        async def slow_task(value):
            await asyncio.sleep(1)
            return value

        async with AsyncWorkQueue[int](name="test_queue", max_queue_size=10, max_concurrency=1) as queue:
            # Submit first task to occupy the worker
            task1 = asyncio.create_task(queue.submit(slow_task, 1))
            await asyncio.sleep(0.01)  # Let it start

            # Submit second task (will be queued)
            task2 = asyncio.create_task(queue.submit(slow_task, 2))
            await asyncio.sleep(0.01)

            # Cancel the second task before it starts
            task2.cancel()

            # Wait for first task
            result1 = await task1
            assert result1 == 1

            # Second task should be cancelled
            with pytest.raises(asyncio.CancelledError):
                await task2

    @pytest.mark.asyncio
    async def test_cancel_multiple_tasks(self):
        """Test cancelling multiple queued tasks."""

        async def task(value):
            await asyncio.sleep(0.1)
            return value

        async with AsyncWorkQueue[int](name="test_queue", max_queue_size=10, max_concurrency=1) as queue:
            # Submit multiple tasks
            tasks = [asyncio.create_task(queue.submit(task, i)) for i in range(5)]

            await asyncio.sleep(0.05)

            # Cancel last 3 tasks
            for t in tasks[2:]:
                t.cancel()

            # First tasks should complete, others should be cancelled
            results = await asyncio.gather(*tasks, return_exceptions=True)

            assert results[0] == 0
            assert results[1] == 1
            assert isinstance(results[2], asyncio.CancelledError)
            assert isinstance(results[3], asyncio.CancelledError)
            assert isinstance(results[4], asyncio.CancelledError)


class TestLifecycle:
    """Tests for lifecycle management (start/stop)."""

    @pytest.mark.asyncio
    async def test_start_and_stop(self):
        """Test explicit start and stop calls."""

        async def simple_task(x):
            return x * 2

        queue = AsyncWorkQueue[int](name="test_queue", max_queue_size=10, max_concurrency=2)

        # Start the queue
        await queue.start()

        # Submit and execute task
        result = await queue.submit(simple_task, 5)
        assert result == 10

        # Stop the queue
        await queue.stop()

    @pytest.mark.asyncio
    async def test_stop_with_wait_for_completion(self):
        """Test stop with wait_for_completion=True."""
        results = []

        async def append_task(value):
            await asyncio.sleep(0.05)
            results.append(value)
            return value

        queue = AsyncWorkQueue[int](name="test_queue", max_queue_size=10, max_concurrency=2)

        await queue.start()

        # Submit tasks and await them before stopping
        tasks = [asyncio.create_task(queue.submit(append_task, i)) for i in range(5)]

        # Wait for all submissions to complete before stopping
        task_results = await asyncio.gather(*tasks)

        # Stop with wait_for_completion=True (default)
        await queue.stop(wait_for_completion=True)

        # All tasks should have completed
        assert len(results) == 5
        assert set(results) == {0, 1, 2, 3, 4}
        assert task_results == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_stop_without_wait_for_completion(self):
        """Test stop with wait_for_completion=False cancels workers immediately."""
        completed = []

        async def tracking_task(value):
            await asyncio.sleep(0.1)
            completed.append(value)
            return value

        queue = AsyncWorkQueue[int](name="test_queue", max_queue_size=10, max_concurrency=2)

        await queue.start()

        # Submit multiple tasks that will take time
        tasks = [asyncio.create_task(queue.submit(tracking_task, i)) for i in range(10)]

        # Give a moment for some tasks to start
        await asyncio.sleep(0.05)

        # Stop without waiting - this cancels workers immediately
        await queue.stop(wait_for_completion=False)

        # Not all tasks should have completed since we stopped early
        assert len(completed) < 10

    @pytest.mark.asyncio
    async def test_lazy_initialization_on_first_submit(self):
        """Test that queue auto-starts on first submission (lazy initialization)."""

        async def task():
            return 42

        queue = AsyncWorkQueue[int](name="test_queue", max_queue_size=10, max_concurrency=2)

        # Queue should not be running yet
        assert not queue._running

        # First submission should auto-start the queue
        result = await queue.submit(task)

        # Queue should now be running
        assert queue._running
        assert result == 42

        # Cleanup
        await queue.stop()

    @pytest.mark.asyncio
    async def test_double_start_is_idempotent(self):
        """Test that calling start twice doesn't cause issues."""

        async def task(x):
            return x + 1

        queue = AsyncWorkQueue[int](name="test_queue", max_queue_size=10, max_concurrency=2)

        await queue.start()
        await queue.start()  # Should be idempotent

        result = await queue.submit(task, 5)
        assert result == 6

        await queue.stop()

    @pytest.mark.asyncio
    async def test_double_stop_is_idempotent(self):
        """Test that calling stop twice doesn't cause issues."""
        queue = AsyncWorkQueue[int](name="test_queue", max_queue_size=10, max_concurrency=2)

        await queue.start()
        await queue.stop()
        await queue.stop()  # Should be idempotent


class TestContextManager:
    """Tests for context manager functionality."""

    @pytest.mark.asyncio
    async def test_context_manager_basic(self):
        """Test using AsyncWorkQueue as a context manager."""

        async def task(x):
            return x * 2

        async with AsyncWorkQueue[int](name="test_queue", max_queue_size=10, max_concurrency=2) as queue:
            result = await queue.submit(task, 21)
            assert result == 42

    @pytest.mark.asyncio
    async def test_context_manager_cleans_up(self):
        """Test that context manager properly cleans up resources."""
        executed = []

        async def track_task(value):
            await asyncio.sleep(0.05)
            executed.append(value)
            return value

        async with AsyncWorkQueue[int](name="test_queue", max_queue_size=10, max_concurrency=2) as queue:
            tasks = [asyncio.create_task(queue.submit(track_task, i)) for i in range(3)]

            await asyncio.gather(*tasks)

        # All tasks should have completed
        assert len(executed) == 3

    @pytest.mark.asyncio
    async def test_context_manager_with_exception(self):
        """Test context manager cleanup when exception occurs."""

        async def failing_task():
            raise ValueError("Test error")

        try:
            async with AsyncWorkQueue[Any](name="test_queue", max_queue_size=10, max_concurrency=2) as queue:
                await queue.submit(failing_task)
        except ValueError:
            pass  # Expected

        # Context manager should have cleaned up properly


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_queue_size_one(self):
        """Test queue with size of 1."""

        async def task(x):
            await asyncio.sleep(0.05)
            return x

        async with AsyncWorkQueue[int](name="test_queue", max_queue_size=1, max_concurrency=1) as queue:
            tasks = [asyncio.create_task(queue.submit(task, i)) for i in range(3)]

            results = await asyncio.gather(*tasks)
            assert results == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_single_worker(self):
        """Test queue with single worker (max_concurrency=1)."""
        execution_order = []

        async def ordered_task(value):
            execution_order.append(value)
            await asyncio.sleep(0.01)
            return value

        async with AsyncWorkQueue[int](name="test_queue", max_queue_size=10, max_concurrency=1) as queue:
            tasks = [asyncio.create_task(queue.submit(ordered_task, i)) for i in range(5)]

            await asyncio.gather(*tasks)

            # With single worker, tasks should execute in order
            assert execution_order == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_high_concurrency(self):
        """Test queue with high concurrency limit."""

        async def quick_task(x):
            return x * x

        async with AsyncWorkQueue[int](name="test_queue", max_queue_size=100, max_concurrency=20) as queue:
            tasks = [asyncio.create_task(queue.submit(quick_task, i)) for i in range(50)]

            results = await asyncio.gather(*tasks)
            assert results == [i * i for i in range(50)]

    @pytest.mark.asyncio
    async def test_empty_queue_operations(self):
        """Test operations on empty queue."""

        async def task(x):
            return x

        async with AsyncWorkQueue[int](name="test_queue", max_queue_size=10, max_concurrency=2) as queue:
            # Submit and complete a task
            result = await queue.submit(task, 42)
            assert result == 42

            # Queue should be empty now, submit another
            result = await queue.submit(task, 100)
            assert result == 100


class TestQueueStatusMethods:
    """Tests for queue status and monitoring methods."""

    @pytest.mark.asyncio
    async def test_num_busy_workers_idle(self):
        """Test that num_busy_workers returns 0 when no tasks are executing."""

        async def quick_task():
            return 42

        async with AsyncWorkQueue[int](name="test_queue", max_queue_size=10, max_concurrency=2) as queue:
            # Initially, no workers should be busy
            assert queue.num_busy_workers() == 0

            # Execute a quick task
            result = await queue.submit(quick_task)
            assert result == 42

            # After task completes, workers should be idle again
            assert queue.num_busy_workers() == 0

    @pytest.mark.asyncio
    async def test_num_busy_workers_during_execution(self):
        """Test that num_busy_workers correctly tracks executing tasks."""
        started = asyncio.Event()
        can_finish = asyncio.Event()

        async def blocking_task(task_id):
            started.set()
            await can_finish.wait()
            return task_id

        async with AsyncWorkQueue[int](name="test_queue", max_queue_size=10, max_concurrency=3) as queue:
            # Submit a task that will block
            task1 = asyncio.create_task(queue.submit(blocking_task, 1))

            # Wait for task to start executing
            await started.wait()
            await asyncio.sleep(0.01)  # Give time for busy_count to update

            # Should have 1 busy worker
            assert queue.num_busy_workers() == 1

            # Allow task to complete
            can_finish.set()
            result = await task1
            assert result == 1

            # Worker should be idle again
            await asyncio.sleep(0.01)
            assert queue.num_busy_workers() == 0

    @pytest.mark.asyncio
    async def test_num_busy_workers_multiple_concurrent(self):
        """Test busy worker count with multiple concurrent tasks."""
        started_count = []
        can_finish = asyncio.Event()

        async def concurrent_task(task_id):
            started_count.append(task_id)
            await can_finish.wait()
            return task_id

        async with AsyncWorkQueue[int](name="test_queue", max_queue_size=10, max_concurrency=3) as queue:
            # Submit 3 tasks that will block
            tasks = [asyncio.create_task(queue.submit(concurrent_task, i)) for i in range(3)]

            # Wait for all tasks to start
            await asyncio.sleep(0.1)

            # Should have 3 busy workers (up to max_concurrency)
            assert queue.num_busy_workers() == 3

            # Allow tasks to complete
            can_finish.set()
            results = await asyncio.gather(*tasks)
            assert results == [0, 1, 2]

            # All workers should be idle
            await asyncio.sleep(0.01)
            assert queue.num_busy_workers() == 0

    @pytest.mark.asyncio
    async def test_is_busy(self):
        """Test is_busy() method."""
        can_finish = asyncio.Event()

        async def blocking_task():
            await can_finish.wait()
            return True

        async with AsyncWorkQueue[bool](name="test_queue", max_queue_size=10, max_concurrency=2) as queue:
            # Queue should not be busy initially
            assert not queue.is_busy()

            # Submit a task
            task = asyncio.create_task(queue.submit(blocking_task))
            await asyncio.sleep(0.05)  # Let task start

            # Queue should be busy
            assert queue.is_busy()

            # Complete the task
            can_finish.set()
            await task

            # Queue should not be busy anymore
            await asyncio.sleep(0.01)
            assert not queue.is_busy()

    @pytest.mark.asyncio
    async def test_is_queue_empty(self):
        """Test is_queue_empty() method."""
        can_finish = asyncio.Event()

        async def blocking_task(value):
            await can_finish.wait()
            return value

        async with AsyncWorkQueue[int](name="test_queue", max_queue_size=5, max_concurrency=1) as queue:
            # Queue should be empty initially
            assert queue.is_queue_empty()

            # Submit first task (will be picked up by worker)
            task1 = asyncio.create_task(queue.submit(blocking_task, 1))
            await asyncio.sleep(0.05)

            # Queue should still be empty (task is being executed, not queued)
            assert queue.is_queue_empty()

            # Submit more tasks (these will queue)
            task2 = asyncio.create_task(queue.submit(blocking_task, 2))
            task3 = asyncio.create_task(queue.submit(blocking_task, 3))
            await asyncio.sleep(0.05)

            # Queue should not be empty now
            assert not queue.is_queue_empty()

            # Allow tasks to complete
            can_finish.set()
            await asyncio.gather(task1, task2, task3)

            # Queue should be empty again
            assert queue.is_queue_empty()

    @pytest.mark.asyncio
    async def test_is_queue_full(self):
        """Test is_queue_full() method."""
        can_finish = asyncio.Event()

        async def blocking_task(value):
            await can_finish.wait()
            return value

        async with AsyncWorkQueue[int](name="test_queue", max_queue_size=2, max_concurrency=1) as queue:
            # Queue should not be full initially
            assert not queue.is_queue_full()

            # Submit first task (picked up by worker)
            task1 = asyncio.create_task(queue.submit(blocking_task, 1))
            await asyncio.sleep(0.05)

            # Queue still not full (1 executing, 0 queued)
            assert not queue.is_queue_full()

            # Submit tasks to fill the queue
            task2 = asyncio.create_task(queue.submit(blocking_task, 2))
            task3 = asyncio.create_task(queue.submit(blocking_task, 3))
            await asyncio.sleep(0.05)

            # Queue should be full now (max_queue_size=2, 2 items queued)
            assert queue.is_queue_full()

            # Complete tasks
            can_finish.set()
            await asyncio.gather(task1, task2, task3)

            # Queue should not be full anymore
            assert not queue.is_queue_full()

    @pytest.mark.asyncio
    async def test_busy_count_reset_on_start(self):
        """Test that busy_count is reset to 0 when start() is called."""
        queue = AsyncWorkQueue[int](name="test_queue", max_queue_size=10, max_concurrency=2)

        await queue.start()
        assert queue.num_busy_workers() == 0
        await queue.stop()

    @pytest.mark.asyncio
    async def test_busy_count_reset_on_stop(self):
        """Test that busy_count is reset to 0 when stop() is called."""
        can_finish = asyncio.Event()

        async def blocking_task():
            await can_finish.wait()
            return True

        queue = AsyncWorkQueue[bool](name="test_queue", max_queue_size=10, max_concurrency=2)

        await queue.start()

        # Submit a task that will block
        _ = asyncio.create_task(queue.submit(blocking_task))
        await asyncio.sleep(0.05)

        # Should have a busy worker
        assert queue.num_busy_workers() > 0

        # Stop the queue
        await queue.stop(wait_for_completion=False)

        # Busy count should be reset
        assert queue.num_busy_workers() == 0


class TestWorkerErrorHandling:
    """Tests for enhanced worker error handling with logging and backoff."""

    @pytest.mark.asyncio
    async def test_worker_continues_after_exception_in_task(self):
        """Test that worker continues processing tasks after a task raises an exception."""
        call_count = []

        async def failing_task():
            call_count.append("fail")
            raise ValueError("Task failed")

        async def success_task():
            call_count.append("success")
            return "OK"

        async with AsyncWorkQueue[str](name="test_queue", max_queue_size=10, max_concurrency=1) as queue:
            # Submit failing task
            with pytest.raises(ValueError):
                await queue.submit(failing_task)

            # Submit success task - worker should still be operational
            result = await queue.submit(success_task)
            assert result == "OK"

            # Both tasks should have been attempted
            assert call_count == ["fail", "success"]

    @pytest.mark.asyncio
    async def test_worker_loop_exception_logging_and_backoff(self):
        """Test that worker loop exceptions are logged with backoff before retry."""
        from unittest.mock import patch

        execution_count = []

        async def normal_task():
            execution_count.append(1)
            return "OK"

        queue = AsyncWorkQueue[str](name="test_queue", max_queue_size=10, max_concurrency=1)

        # Patch the _queue.get() to raise an exception once, then work normally
        original_get = queue._queue.get
        get_call_count = [0]

        async def mock_get():
            get_call_count[0] += 1
            if get_call_count[0] == 1:
                # First call raises an exception (simulating infrastructure error)
                raise RuntimeError("Simulated queue infrastructure error")
            # Subsequent calls work normally
            return await original_get()

        with patch.object(queue._queue, "get", side_effect=mock_get):
            with patch("nemoguardrails.guardrails.async_work_queue.log") as mock_log:
                await queue.start()

                # Give worker time to hit the exception and recover
                await asyncio.sleep(0.2)

                # Submit a task after the worker has recovered
                result = await queue.submit(normal_task)
                assert result == "OK"
                assert len(execution_count) == 1

                # Verify critical logging was called
                assert mock_log.critical.called
                call_args = mock_log.critical.call_args
                assert "crashed due to" in call_args[0][0]
                assert "RuntimeError" in call_args[0][0]

                await queue.stop()
