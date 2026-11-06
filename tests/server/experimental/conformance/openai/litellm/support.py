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

"""Load the upstream response schema independently of guard declarations."""

import json
from pathlib import Path

from jsonschema import Draft202012Validator

OPENAPI = json.loads((Path(__file__).parent / "fixtures/response.openapi.json").read_text())


def response_validator(operation):
    schema = OPENAPI["paths"]["/v1/" + operation]["post"]["responses"]["200"]["content"]["application/json"]["schema"]
    return Draft202012Validator({**schema, "components": OPENAPI["components"]})
