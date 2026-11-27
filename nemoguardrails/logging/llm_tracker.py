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
from contextvars import ContextVar
from functools import wraps
from time import time

from nemoguardrails.context import (
    explain_info_var,
    llm_call_info_var,
    llm_response_metadata_var,
    llm_stats_var,
    reasoning_trace_var,
    tool_calls_var,
)
from nemoguardrails.logging.explain import LLMCallInfo
from nemoguardrails.logging.processing_log import processing_log_var
from nemoguardrails.logging.stats import LLMStats
from nemoguardrails.utils import new_uuid

log = logging.getLogger(__name__)


_active_llm_call: ContextVar[bool] = ContextVar("active_llm_call", default=False)


def track_llm_call(func=None, *, isolated=False):
    """Track a fresh invocation while preserving the legacy decorator's view.

    Shared model calls opt into isolated records. Direct legacy decorator users
    retain a reusable post-call object, but log entries are snapshots of each
    invocation. Nested calls restore the outer invocation's context.
    """
    if func is None:
        return lambda wrapped: track_llm_call(wrapped, isolated=isolated)

    @wraps(func)
    async def wrapper(*args, **kwargs):
        previous = llm_call_info_var.get()
        nested = _active_llm_call.get()
        llm_call_info = LLMCallInfo(task=previous.task if previous is not None else None)
        llm_call_info.id = new_uuid()
        llm_call_info.started_at = time()
        explain_info = explain_info_var.get()
        explain_index = None
        if explain_info is not None:
            explain_index = len(explain_info.llm_calls)
            explain_info.llm_calls.append(llm_call_info)
        llm_stats = llm_stats_var.get()
        if llm_stats is None:
            llm_stats = LLMStats()
            llm_stats_var.set(llm_stats)
        llm_stats.inc("total_calls")
        info_token = llm_call_info_var.set(llm_call_info)
        active_token = _active_llm_call.set(True)
        reasoning_token = reasoning_trace_var.set(None)
        tools_token = tool_calls_var.set(None)
        metadata_token = llm_response_metadata_var.set(None)
        succeeded = False

        try:
            result = await func(*args, **kwargs)
            succeeded = True
            return result
        finally:
            try:
                llm_call_info.finished_at = time()
                if llm_call_info.started_at:
                    llm_call_info.duration = llm_call_info.finished_at - llm_call_info.started_at
                    llm_stats.inc("total_time", llm_call_info.duration)

                # The compatibility view replaces fields, so a shallow snapshot
                # isolates records without copying opaque provider payloads.
                recorded_info = llm_call_info.model_copy()
                if explain_info is not None and explain_index is not None:
                    explain_info.llm_calls[explain_index] = recorded_info
                processing_log = processing_log_var.get()
                if processing_log is not None:
                    processing_log.append({"type": "llm_call_info", "timestamp": time(), "data": recorded_info})
            except Exception:
                if succeeded:
                    succeeded = False
                    raise
                log.warning("Failed to record an interrupted model call", exc_info=True)
            finally:
                _active_llm_call.reset(active_token)
                llm_call_info_var.reset(info_token)
                if nested or not succeeded:
                    reasoning_trace_var.reset(reasoning_token)
                    tool_calls_var.reset(tools_token)
                    llm_response_metadata_var.reset(metadata_token)
                elif isolated or previous is None:
                    llm_call_info_var.set(llm_call_info)
                else:
                    for name in type(llm_call_info).model_fields:
                        setattr(previous, name, getattr(llm_call_info, name))
                    llm_call_info_var.set(previous)

    return wrapper
