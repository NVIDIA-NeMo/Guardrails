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
import re
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


def extract_model_name_and_base_url(
    serialized: Dict[str, Any]
) -> tuple[Optional[str], Optional[str]]:
    """Extract model name and base URL from serialized LLM parameters.

    Args:
        serialized: The serialized LLM configuration

    Returns:
        A tuple of (model_name, base_url). Either value can be None if not found
    """
    model_name = None
    base_url = None

    # Case 1: Try to extract from kwargs (we expect kwargs to be populated for the `ChatOpenAI` class).
    if "kwargs" in serialized:
        kwargs = serialized["kwargs"]

        # Check for model_name in kwargs (ChatOpenAI attribute)
        if "model_name" in kwargs and kwargs["model_name"]:
            model_name = str(kwargs["model_name"])

        # Check for openai_api_base in kwargs (ChatOpenAI attribute)
        if "openai_api_base" in kwargs and kwargs["openai_api_base"]:
            base_url = str(kwargs["openai_api_base"])

    # Case 2: For other providers, parse `repr`, a string representation of the provider class. Since we don't
    # have a reference to the actual class, we need to parse the string representation.
    if "repr" in serialized and isinstance(serialized["repr"], str):
        repr_str = serialized["repr"]

        # Extract model name. We expect the property to be formatted like model='...' or model_name='...',
        # and check for single and double quotes.
        if not model_name:
            match = re.search(r"model(?:_name)?=['\"]([^'\"]+)['\"]", repr_str)
            if match:
                model_name = match.group(1)

        # Extract base URL. The property name may vary between providers, so try common names.
        # We expect the property to be formatted like property_name='...', and check for single and double quotes.
        if not base_url:
            url_attrs = [
                "api_base",
                "api_host",
                "azure_endpoint",
                "base_url",
                "endpoint",
                "endpoint_url",
                "openai_api_base",
            ]
            for attr in url_attrs:
                match = re.search(rf"{attr}=['\"]([^'\"]+)['\"]", repr_str)
                if match:
                    base_url = match.group(1)
                    break

    return model_name, base_url
