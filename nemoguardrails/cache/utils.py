# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import re
from typing import List, Union

PROMPT_PATTERN_WHITESPACES = re.compile(r"\s+")


def create_normalized_cache_key(
    prompt: Union[str, List[str]], normalize_whitespace: bool = True
) -> str:
    """
    Create a normalized cache key from a prompt.

    Args:
        prompt: The prompt to normalize (string or list of strings)
        normalize_whitespace: Whether to normalize whitespace characters

    Returns:
        A normalized string suitable for use as a cache key
    """
    if isinstance(prompt, list):
        prompt_str = json.dumps(prompt, sort_keys=True)
    else:
        prompt_str = str(prompt)

    if normalize_whitespace:
        prompt_str = PROMPT_PATTERN_WHITESPACES.sub(" ", prompt_str).strip()

    return prompt_str
