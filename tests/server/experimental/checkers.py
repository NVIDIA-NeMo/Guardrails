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

"""Provider-neutral checker doubles for proxy wiring and protocol tests.

These decisions are synthetic, not evidence of a configured Rails engine's
verdict. Capture the full check objects so tests can inspect text and context.
"""

from nemoguardrails.server.experimental._content_checker import (
    ContentAllowed,
    InputContentCheck,
)


class StaticChecker:
    """Return a fixed input verdict and successive scripted output verdicts.

    Output checks allow by default after the script is exhausted. Tests that
    require an exact sequence must assert the captured calls as well.
    """

    def __init__(self, policy, input_decision=ContentAllowed(), output_decisions=()):
        self.policy = policy
        self.input_decision = input_decision
        self.output_decisions = list(output_decisions)
        self.calls = []
        self.policy_reads = 0

    def inspection_policy(self):
        self.policy_reads += 1
        return self.policy

    async def check_input(self, check):
        self.calls.append(("input", check))
        return self.input_decision

    async def check_output(self, check):
        self.calls.append(("output", check))
        return self.output_decisions.pop(0) if self.output_decisions else ContentAllowed()

    @property
    def text_calls(self):
        """Expose text observations without assuming nonoverlapping windows."""
        return [
            (kind, check.message.content if isinstance(check, InputContentCheck) else check.output_content)
            for kind, check in self.calls
        ]
