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

import logging
from functools import wraps
from typing import Any, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import generate_from_stream
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_nvidia_ai_endpoints import ChatNVIDIA as ChatNVIDIAOriginal
from pydantic.v1 import Field

log = logging.getLogger(__name__)


def stream_decorator(func):
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
class ChatNVIDIA(ChatNVIDIAOriginal):
    streaming: bool = Field(
        default=False, description="Whether to use streaming or not"
    )

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
