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


class ClavataPluginError(Exception):
    """
    Base exception for all Clavata plugin errors.
    """


class ClavataPluginConfigurationError(ClavataPluginError):
    """
    Exception raised when the Clavata plugin is not configured correctly.
    """


class ClavataPluginValueError(ClavataPluginError):
    """
    Exception raised when the Clavata plugin is used incorrectly.
    """


class ClavataPluginTypeError(ClavataPluginError):
    """
    Exception raised when the Clavata plugin is used incorrectly due to type mismatches.
    """


class ClavataPluginAPIError(ClavataPluginError):
    """
    Exception raised when the Clavata plugin API returns an error.
    """


class ClavataPluginAPIRateLimitError(ClavataPluginAPIError):
    """
    Exception raised when the Clavata plugin API rate limit is exceeded.
    """
