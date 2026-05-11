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

"""Public testing surface for NeMo Guardrails.

This subpackage ships utilities that help users write fast, deterministic tests
for their guardrails configurations. The two main building blocks are:

* :class:`FakeLLMModel`: a scriptable implementation of the ``LLMModel``
  protocol that returns canned responses.
* :class:`TestChat`: an ergonomic helper for asserting bot replies against a
  scripted conversation.

Pytest fixtures are exposed via the ``nemoguardrails.testing.fixtures`` plugin.
Add it to your ``conftest.py`` to opt in::

    pytest_plugins = ["nemoguardrails.testing.fixtures"]
"""

from nemoguardrails.testing.chat_harness import TestChat
from nemoguardrails.testing.fake_model import FakeLLMModel

__all__ = ["FakeLLMModel", "TestChat"]
