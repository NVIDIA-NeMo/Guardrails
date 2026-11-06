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

"""Test declarations for buffered guarded operations."""

import pytest

from nemoguardrails.server.experimental._guarded_operation import BufferedGuardedOperation
from nemoguardrails.server.experimental.provider.types import GuardedMessage


def project_input(payload):
    return GuardedMessage("user", payload["input"])


def project_output(payload):
    return GuardedMessage("assistant", payload["output"])


def test_buffered_operation_accepts_private_dotted_name_and_projections():
    """Accept a dotted operation name and callable projections."""

    operation = BufferedGuardedOperation(
        name="test.chat.completions",
        input_projection=project_input,
        output_projection=project_output,
    )

    assert operation.input_projection({"input": "question"}).content == "question"
    assert operation.output_projection({"output": "answer"}).content == "answer"


@pytest.mark.parametrize("name", ["", ".chat", "chat.", "chat-completions", "chat completions"])
def test_buffered_operation_rejects_invalid_names(name):
    """Reject operation names that cannot serve as dotted identities."""

    with pytest.raises(ValueError, match="dotted identifier"):
        BufferedGuardedOperation(
            name=name,
            input_projection=project_input,
            output_projection=project_output,
        )


@pytest.mark.parametrize("field", ["input_projection", "output_projection"])
def test_buffered_operation_rejects_non_callable_projections(field):
    """Require both provider projection hooks to be callable."""

    values = {
        "name": "test.operation",
        "input_projection": project_input,
        "output_projection": project_output,
    }
    values[field] = object()

    with pytest.raises(TypeError, match=f"{field.removesuffix('_projection')} projection"):
        BufferedGuardedOperation(**values)
