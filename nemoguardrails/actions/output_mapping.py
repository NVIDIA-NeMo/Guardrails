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

from typing import Any, Tuple


def default_output_mapping(result: Any) -> bool:
    """A fallback output mapping if an action does not provide one.

    - For a boolean result: assume True means allowed (so block if False).
    - For a numeric result: use 0.5 as a threshold (block if the value is less).
    - Otherwise, assume the result is allowed.
    """
    if isinstance(result, bool):
        return not result
    elif isinstance(result, (int, float)):
        return result < 0.5
    else:
        return False


def is_output_blocked(result: Any, action_func: Any) -> bool:
    """Determines if an action result is not allowed using its attached mapping.

    Args:
        result: The value returned by the action.
        action_func: The action function (whose metadata contains the mapping).

    Returns:
        True if the mapping indicates that the output should be blocked, False otherwise.
    """
    mapping = getattr(action_func, "action_meta", {}).get("output_mapping")
    if mapping is None:
        mapping = default_output_mapping

    if not isinstance(result, Tuple):
        result = (result,)

    return mapping(result[0])
