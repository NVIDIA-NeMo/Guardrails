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
    "https://github.com/openai/openai-openapi/blob/3e288ba501d136b81181919a12a88495a0244488/openapi.yaml"
)
PROVIDER_DOWNLOAD_URL = (
    "https://raw.githubusercontent.com/openai/openai-openapi/3e288ba501d136b81181919a12a88495a0244488/openapi.yaml"
)
PROVIDER_REVISION = "3e288ba501d136b81181919a12a88495a0244488"
PROVIDER_DOCUMENT_VERSION = "2.3.0"
PROVIDER_DOCUMENT_SHA256 = "cf91be897ab55e4aefcfc44ec1eb7611eea1227509b67178be583244c2680ab3"
