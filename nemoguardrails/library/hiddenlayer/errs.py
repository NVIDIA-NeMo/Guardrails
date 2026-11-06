# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Errors the HiddenLayer rail raises, all rooted at :class:`HiddenLayerRailError`."""

from typing import Optional


class HiddenLayerRailError(Exception):
    """Base for every error the HiddenLayer rail raises."""


class HiddenLayerTransportError(HiddenLayerRailError):
    """A HiddenLayer request failed. The message holds only status, content type and length."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class HiddenLayerActionError(HiddenLayerRailError):
    """The response carries no action to apply. The message never quotes the response."""


class HiddenLayerVerdictError(HiddenLayerRailError):
    """The action cannot be applied to what was sent. The message never quotes the response."""


class HiddenLayerCheckError(HiddenLayerRailError):
    """A check could not be completed and the rail is configured to fail closed."""
