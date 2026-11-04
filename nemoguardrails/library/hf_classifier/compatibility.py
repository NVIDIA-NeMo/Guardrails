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

"""Compatibility checks for local classifier overflow tokenization."""


def ensure_overflow_support() -> None:
    """Reject known overflow regressions only when windowing is required."""
    from tokenizers import __version__

    if __version__.split("+", 1)[0] in {"0.23.1", "0.23.2"}:
        raise ValueError(
            "Local HF classifier windowing is unsafe with tokenizers 0.23.1 or 0.23.2: "
            "these versions can silently discard overflow tokens. "
            'Upgrade compatible packages with: pip install --upgrade transformers "tokenizers>=0.23.3", '
            "then restart the process. "
            "See https://github.com/huggingface/tokenizers/issues/2091."
        )
