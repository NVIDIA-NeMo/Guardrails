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

import argparse
import hashlib
import json
from pathlib import Path

import yaml

COMPONENTS = (
    "CreateResponse",
    "Response",
    "Reasoning",
    "ResponseTextParam",
    "ResponseFormatText",
    "OutputMessage",
    "OutputTextContent",
    "ReasoningItem",
    "ResponseTextDeltaEvent",
    "ResponseTextDoneEvent",
    "ResponseContentPartAddedEvent",
    "ResponseContentPartDoneEvent",
    "ResponseOutputItemAddedEvent",
    "ResponseOutputItemDoneEvent",
    "ResponseCreatedEvent",
    "ResponseInProgressEvent",
    "ResponseQueuedEvent",
    "ResponseCompletedEvent",
    "ResponseFailedEvent",
    "ResponseIncompleteEvent",
    "ResponseErrorEvent",
)


def extract_inventory(document, pin):
    if hashlib.sha256(document).hexdigest() != pin["sha256"]:
        raise ValueError("Source bytes do not match contracts/openai/source.yaml")
    source = yaml.safe_load(document)
    schemas = source["components"]["schemas"]

    def fields(node):
        result = set(node.get("properties", {}))
        if "$ref" in node:
            reference = node["$ref"]
            if not reference.startswith("#/components/schemas/"):
                raise ValueError("Only local schema references are supported")
            result.update(fields(schemas[reference.removeprefix("#/components/schemas/")]))
        for part in node.get("allOf", []):
            result.update(fields(part))
        return result

    examples = source["paths"]["/responses"]["post"]["x-oaiMeta"]["examples"]
    text_example = next(example for example in examples if example["title"] == "Text input")
    return {
        "source": pin,
        "schema_fields": {name: sorted(fields(schemas[name])) for name in COMPONENTS},
        "text_response_example": json.loads(text_example["response"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("descriptor", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    pin = yaml.safe_load(args.descriptor.read_text(encoding="utf-8"))
    inventory = extract_inventory(args.source.read_bytes(), pin)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
