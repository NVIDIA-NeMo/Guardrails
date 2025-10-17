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

import asyncio
import os

import nest_asyncio

# Keep track of whether the patch was applied or not
nest_asyncio_patch_applied = False


def apply():
    global nest_asyncio_patch_applied

    if os.environ.get("DISABLE_NEST_ASYNCIO", "true").lower() not in [
        "true",
        "1",
        "yes",
    ]:
        nest_asyncio.apply()
        nest_asyncio_patch_applied = True


def check_sync_call_from_async_loop():
    """Helper to check if a sync call is made from an async loop.

    Returns
        True if a sync call is made from an async loop.
    """
    if hasattr(asyncio, "_nest_patched"):
        return False

    if nest_asyncio_patch_applied:
        return False

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        return True

    return False
