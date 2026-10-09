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

"""Pinned OpenAI OpenAPI source metadata.

These constants mirror contracts/openai/source.yaml; a test checks that
the values match.
"""

PROVIDER_DOCUMENT_URL = (
    "https://github.com/openai/openai-openapi/blob/df63773f69f542ef875b9f00c3837c25ba5f4f2a/openapi.yaml"
)
PROVIDER_DOWNLOAD_URL = (
    "https://raw.githubusercontent.com/openai/openai-openapi/df63773f69f542ef875b9f00c3837c25ba5f4f2a/openapi.yaml"
)
PROVIDER_REVISION = "df63773f69f542ef875b9f00c3837c25ba5f4f2a"
PROVIDER_DOCUMENT_VERSION = "2.3.0"
PROVIDER_DOCUMENT_SHA256 = "f2dae1a9aced09b91310db89edda51bf1e36ecbfb05230c3a50b239c07708469"
