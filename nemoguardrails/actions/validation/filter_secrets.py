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


def contains_secrets(resp):
    """Validate if response have any of the key present
    Refer https://github.com/Yelp/detect-secrets for detection process
    response is string of format
    AWSKeyDetector         : False
    ArtifactoryDetector    : False
    """
    try:
        import detect_secrets  # type: ignore (Assume user installs detect_secrets with instructions below)
    except ModuleNotFoundError:
        raise ValueError("Could not import detect_secrets. Please install using `pip install detect-secrets`")

    with detect_secrets.settings.default_settings():
        res = detect_secrets.scan_adhoc_string(resp)

    for secret_type in res.split("\n"):
        if "True" in secret_type:
            return True

    return False
