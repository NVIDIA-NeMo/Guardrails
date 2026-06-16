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

import logging
from functools import wraps
from time import time

from nemoguardrails.context import explain_info_var, llm_call_info_var, llm_stats_var
from nemoguardrails.logging.explain import LLMCallInfo
from nemoguardrails.logging.processing_log import processing_log_var
from nemoguardrails.logging.stats import LLMStats
from nemoguardrails.utils import new_uuid

log = logging.getLogger(__name__)


def track_llm_call(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        llm_call_info = llm_call_info_var.get()
        if llm_call_info is None:
            llm_call_info = LLMCallInfo()
            llm_call_info_var.set(llm_call_info)

        llm_call_info.id = new_uuid()
        llm_call_info.started_at = time()

        explain_info = explain_info_var.get()
        if explain_info:
            explain_info.llm_calls.append(llm_call_info)

        llm_stats = llm_stats_var.get()
        if llm_stats is None:
            llm_stats = LLMStats()
            llm_stats_var.set(llm_stats)
        llm_stats.inc("total_calls")

        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            llm_call_info.finished_at = time()
            if llm_call_info.started_at:
                llm_call_info.duration = llm_call_info.finished_at - llm_call_info.started_at
                llm_stats.inc("total_time", llm_call_info.duration)

            processing_log = processing_log_var.get()
            if processing_log is not None:
                processing_log.append({"type": "llm_call_info", "timestamp": time(), "data": llm_call_info})

    return wrapper
