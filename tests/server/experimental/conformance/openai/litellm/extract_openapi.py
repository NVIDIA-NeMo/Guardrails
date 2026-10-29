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

"""Extract response schemas verbatim from LiteLLM's published OpenAPI snapshot.

Usage: python extract_openapi.py DOWNLOADED_OPENAPI_JSON OUTPUT_JSON
No network requests or provider credentials are used.
"""

import hashlib
import json
import sys
from pathlib import Path


def extract(source):
    document = json.loads(source)
    result = {
        "openapi": document["openapi"],
        "info": document["info"],
        "x-source": {
            "url": "https://docs.litellm.ai/openapi.json",
            "sha256": hashlib.sha256(source).hexdigest(),
            "scope": "POST 200 responses only; referenced component schemas copied unchanged",
        },
        "paths": {
            path: {"post": {"responses": {"200": document["paths"][path]["post"]["responses"]["200"]}}}
            for path in ("/v1/chat/completions", "/v1/responses")
        },
        "components": {"schemas": {}},
    }

    def collect(value):
        if isinstance(value, dict):
            ref = value.get("$ref")
            if ref:
                prefix = "#/components/schemas/"
                if not ref.startswith(prefix):
                    raise ValueError(f"Unsupported reference: {ref}")
                name = ref.removeprefix(prefix)
                if name not in result["components"]["schemas"]:
                    schema = document["components"]["schemas"][name]
                    result["components"]["schemas"][name] = schema
                    collect(schema)
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(result["paths"])
    return result


if __name__ == "__main__":
    Path(sys.argv[2]).write_text(json.dumps(extract(Path(sys.argv[1]).read_bytes()), indent=2) + "\n")
