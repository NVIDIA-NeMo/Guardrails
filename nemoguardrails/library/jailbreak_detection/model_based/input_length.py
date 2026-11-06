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

import gzip
import hashlib
from functools import lru_cache
from importlib.resources import files

from .errors import JailbreakInputTooLongError, JailbreakInputValidationError

MAX_INPUT_TOKENS = 2048
TOKENIZER_SHA256 = "91f1def9b9391fdabe028cd3f3fcc4efd34e5d1f08c3bf2de513ebb5911a1854"


@lru_cache(maxsize=1)
def _input_tokenizer():
    from tokenizers import Tokenizer

    resource = files(__package__).joinpath("tokenizer").joinpath("tokenizer.json.gz")
    data = gzip.decompress(resource.read_bytes())
    if hashlib.sha256(data).hexdigest() != TOKENIZER_SHA256:
        raise ValueError("Bundled jailbreak tokenizer checksum mismatch.")
    tokenizer = Tokenizer.from_str(data.decode("utf-8"))
    tokenizer.no_padding()
    tokenizer.enable_truncation(max_length=MAX_INPUT_TOKENS + 1)
    return tokenizer


def validate_input_length(prompt: str) -> None:
    """Reject input the detector cannot fully inspect, including special tokens."""
    try:
        token_count = len(_input_tokenizer().encode(prompt).ids)
    except Exception as e:
        raise JailbreakInputValidationError("Unable to validate the jailbreak detector's input limit.") from e
    if token_count > MAX_INPUT_TOKENS:
        raise JailbreakInputTooLongError(
            f"Input exceeds the jailbreak detector's {MAX_INPUT_TOKENS}-token limit (including special tokens)."
        )
