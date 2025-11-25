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


from nemoguardrails.llm.providers.providers import _llm_providers


def test_acall_method_added():
    for provider_name, provider_cls in _llm_providers.items():
        assert hasattr(provider_cls, "_acall"), f"_acall not added to {provider_name}"
        assert callable(getattr(provider_cls, "_acall")), f"_acall is not callable in {provider_name}"
