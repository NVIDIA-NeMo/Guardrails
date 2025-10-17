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

from typing import Union


class LLMStats:
    """Simple class to store stats for the LLM usage."""

    def __init__(self):
        self._stats = self._get_empty_stats()

    @staticmethod
    def _get_empty_stats():
        return {
            "total_calls": 0,
            "total_time": 0,
            "total_tokens": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "latencies": [],
        }

    def inc(self, name: str, value: Union[float, int] = 1):
        """Increment a stat."""
        if name not in self._stats:
            self._stats[name] = 0

        self._stats[name] += value

        if name == "total_time":
            self._stats["latencies"].append(value)

    def get_stat(self, name):
        return self._stats[name]

    def get_stats(self):
        return self._stats

    def reset(self):
        self._stats = self._get_empty_stats()

    def __str__(self):
        return (
            f"{self._stats['total_calls']} total calls, "
            f"{round(self._stats['total_time'], 2)} total time, "
            f"{self._stats['total_tokens']} total tokens, "
            f"{self._stats['total_prompt_tokens']} total prompt tokens, "
            f"{self._stats['total_completion_tokens']} total completion tokens, "
            f"{[round(x, 2) for x in self._stats['latencies']]} as latencies"
        )
