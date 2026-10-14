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

"""Synthetic Chat payloads shared by router and transport wiring tests.

These are protocol fixtures, not recorded provider responses. Keep Chat-specific
payload construction separate from provider-neutral checker doubles.
"""

import json


def request_body(**updates):
    payload = {
        "model": "gpt-example",
        "messages": [{"role": "user", "content": "question"}],
        "stream": True,
    }
    payload.update(updates)
    return json.dumps(payload, separators=(",", ":")).encode()


def stream_event(content=None, **delta_updates):
    delta = {"content": content} if content is not None else {"role": "assistant"}
    delta.update(delta_updates)
    payload = {
        "id": "chatcmpl-stream",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }
    return b"data: " + json.dumps(payload, separators=(",", ":")).encode() + b"\n\n"
