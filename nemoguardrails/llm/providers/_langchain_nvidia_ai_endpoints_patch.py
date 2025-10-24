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

import inspect
import logging
from functools import wraps
from typing import Any, Dict, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import generate_from_stream
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_nvidia_ai_endpoints import ChatNVIDIA as ChatNVIDIAOriginal
from pydantic import Field

log = logging.getLogger(__name__)  # pragma: no cover


def stream_decorator(func):  # pragma: no cover
    @wraps(func)
    def wrapper(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)
        else:
            return func(self, messages, stop, run_manager, **kwargs)

    return wrapper


# NOTE: this needs to have the same name as the original class,
#   otherwise, there's a check inside `langchain-nvidia-ai-endpoints` that will fail.
class ChatNVIDIA(ChatNVIDIAOriginal):  # pragma: no cover
    streaming: bool = Field(
        default=False, description="Whether to use streaming or not"
    )
    custom_headers: Optional[Dict[str, str]] = Field(
        default=None, description="Custom HTTP headers to send with requests"
    )

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        if self.custom_headers:
            custom_headers_error = (
                "custom_headers requires langchain-nvidia-ai-endpoints >= 0.3.0. "
                "Your version does not support the required client structure or "
                "extra_headers parameter. Please upgrade: "
                "pip install --upgrade langchain-nvidia-ai-endpoints>=0.3.0"
            )
            if not hasattr(self._client, "get_req"):
                raise RuntimeError(custom_headers_error)

            sig = inspect.signature(self._client.get_req)
            if "extra_headers" not in sig.parameters:
                raise RuntimeError(custom_headers_error)

            self._wrap_client_methods()

    def _wrap_client_methods(self):
        original_get_req = self._client.get_req
        original_get_req_stream = self._client.get_req_stream

        def wrapped_get_req(payload: dict = None, extra_headers: dict = None):
            payload = payload or {}
            extra_headers = extra_headers or {}
            merged_headers = {**extra_headers, **self.custom_headers}
            return original_get_req(payload=payload, extra_headers=merged_headers)

        def wrapped_get_req_stream(payload: dict = None, extra_headers: dict = None):
            payload = payload or {}
            extra_headers = extra_headers or {}
            merged_headers = {**extra_headers, **self.custom_headers}
            return original_get_req_stream(
                payload=payload, extra_headers=merged_headers
            )

        object.__setattr__(self._client, "get_req", wrapped_get_req)
        object.__setattr__(self._client, "get_req_stream", wrapped_get_req_stream)

    @stream_decorator
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return super()._generate(
            messages=messages, stop=stop, run_manager=run_manager, **kwargs
        )


__all__ = ["ChatNVIDIA"]
