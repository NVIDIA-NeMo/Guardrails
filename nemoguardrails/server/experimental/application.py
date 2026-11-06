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

"""Python configuration and assembly for the transparent proxy.

Use ProxyConfig for reviewed built-ins, or pass explicit ProviderIntegration
objects to the same factory for custom Python composition. Both use the same
application lifespan, HTTP owner, routing, and Rails repository. This surface
does not load YAML, export contracts, or construct policy from generated files.
"""

from nemoguardrails.server.experimental._application import create_proxy_application
from nemoguardrails.server.experimental._configuration import ProviderConfig, ProxyConfig
from nemoguardrails.server.experimental.rails import ConfigEntry

__all__ = ["ConfigEntry", "ProviderConfig", "ProxyConfig", "create_proxy_application"]
