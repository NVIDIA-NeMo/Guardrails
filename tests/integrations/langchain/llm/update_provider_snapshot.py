# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Regenerate the langchain provider drift snapshot.

Run after a langchain upgrade that the drift canary
(`test_langchain_provider_drift`) has flagged, once the new provider set has
been reviewed and accepted:

    poetry run python tests/integrations/langchain/llm/update_provider_snapshot.py
"""

import json
from importlib.metadata import version
from pathlib import Path

from nemoguardrails.integrations.langchain.providers.providers import (
    _discover_langchain_community_chat_providers,
    _discover_langchain_community_llm_providers,
    _discover_langchain_partner_chat_providers,
)

SNAPSHOT_PATH = Path(__file__).parent / "langchain_provider_snapshot.json"


def build_snapshot():
    # partner_chat_providers is sourced from the `langchain` core package (not
    # langchain-community) and includes our own custom providers such as "nim".
    return {
        "langchain_version": version("langchain"),
        "langchain_community_version": version("langchain-community"),
        "community_llm_providers": sorted(_discover_langchain_community_llm_providers().keys()),
        "community_chat_providers": sorted(_discover_langchain_community_chat_providers().keys()),
        "partner_chat_providers": sorted(_discover_langchain_partner_chat_providers()),
    }


if __name__ == "__main__":
    SNAPSHOT_PATH.write_text(json.dumps(build_snapshot(), indent=2) + "\n")
    print(f"wrote {SNAPSHOT_PATH}")
