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

try:
    from guardrails.errors import ValidationError

    GuardrailsAIValidationError = ValidationError
except ImportError:
    # create a fallback error class when guardrails is not installed
    class GuardrailsAIValidationError(Exception):
        """Fallback validation error when guardrails package is not available."""

        pass


class GuardrailsAIError(Exception):
    """Base exception for Guardrails AI integration."""

    pass


class GuardrailsAIConfigError(GuardrailsAIError):
    """Raised when configuration is invalid."""

    pass


__all__ = [
    "GuardrailsAIError",
    "GuardrailsAIValidationError",
    "GuardrailsAIConfigError",
]
