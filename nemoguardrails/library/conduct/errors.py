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

"""Conduct-specific exceptions surfaced to Colang flows."""


class ConductPluginError(Exception):
    """Base class for anything raised out of the Conduct rail."""


class ConductPluginConfigurationError(ConductPluginError):
    """Raised when ``rails.conduct`` is missing a required setting
    (agent_token, api_url, etc.) or when a bad value is supplied."""


class ConductPluginImportError(ConductPluginError):
    """Raised at rail-load time when ``conduct-nemo-guard`` is not
    installed. Guides the user to::

        pip install conduct-nemo-guard
    """
