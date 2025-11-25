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

"""NeMo Guardrails Toolkit."""

import os
from importlib.metadata import version

# If no explicit value is set for TOKENIZERS_PARALLELISM, we disable it
# to get rid of the annoying warning.
if not os.environ.get("TOKENIZERS_PARALLELISM"):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


import warnings

from . import patch_asyncio
from .rails import LLMRails, RailsConfig

patch_asyncio.apply()

# Ignore a warning message from torch.
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

__version__ = version("nemoguardrails")
__all__ = ["LLMRails", "RailsConfig"]
