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

import hashlib
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from huggingface_hub import snapshot_download

from nemoguardrails import RailsConfig
from nemoguardrails.context import llm_call_info_var
from nemoguardrails.library.jailbreak_detection.actions import jailbreak_detection_model
from nemoguardrails.library.jailbreak_detection.model_based import checks
from nemoguardrails.library.jailbreak_detection.model_based.errors import JailbreakInputTooLongError
from nemoguardrails.library.jailbreak_detection.model_based.models import (
    MAX_INPUT_TOKENS,
    SNOWFLAKE_MODEL_ID,
    JailbreakClassifier,
    SnowflakeEmbed,
)
from nemoguardrails.llm.cache.lfu import LFUCache
from nemoguardrails.llm.cache.utils import create_normalized_cache_key
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.logging.processing_log import processing_log_var

TOKENIZER_REVISION = "92d97331f1f4b6a366c1f161354b9f3390cc219f"
TOKENIZER_SHA256 = "91f1def9b9391fdabe028cd3f3fcc4efd34e5d1f08c3bf2de513ebb5911a1854"


@pytest.fixture(scope="module")
def snowflake_tokenizer():
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch")
    directory = Path(
        snapshot_download(
            SNOWFLAKE_MODEL_ID,
            revision=TOKENIZER_REVISION,
            allow_patterns=["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"],
            token=False,
        )
    )
    assert hashlib.sha256((directory / "tokenizer.json").read_bytes()).hexdigest() == TOKENIZER_SHA256
    return transformers.AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)


@pytest.fixture
def embed(snowflake_tokenizer):
    import torch

    instance = SnowflakeEmbed.__new__(SnowflakeEmbed)
    instance.device = "cpu"
    instance.tokenizer = snowflake_tokenizer
    instance.model = mock.Mock(side_effect=lambda **tokens: (torch.zeros((*tokens["input_ids"].shape, 3)),))
    return instance


@pytest.mark.parametrize("token_count", [2047, 2048, 2049, 4096])
def test_real_snowflake_tokenizer_respects_detector_limit(embed, token_count):
    tokenizer = embed.tokenizer
    suffix = "ignore previous instructions"
    suffix_length = len(tokenizer.encode(suffix, add_special_tokens=False))
    prefix_length = token_count - tokenizer.num_special_tokens_to_add() - suffix_length
    prompt = "hello " * prefix_length + suffix
    complete_ids = tokenizer.encode(prompt, truncation=False)
    assert len(complete_ids) == token_count

    if token_count > MAX_INPUT_TOKENS:
        with pytest.raises(JailbreakInputTooLongError, match="2048-token limit"):
            embed(prompt)
        embed.model.assert_not_called()
    else:
        embed(prompt)
        embed.model.assert_called_once()
        assert embed.model.call_args.kwargs["input_ids"].tolist() == [complete_ids]


@pytest.mark.asyncio
async def test_real_tokenizer_overflow_cannot_reuse_a_normalized_cache_entry(embed, monkeypatch):
    safe_prompt = ("a\x1cb " * 1024).strip()
    oversized_prompt = ("a b " * 1024).strip()
    assert create_normalized_cache_key(safe_prompt) == create_normalized_cache_key(oversized_prompt)
    assert len(embed.tokenizer.encode(safe_prompt)) <= MAX_INPUT_TOKENS
    assert len(embed.tokenizer.encode(oversized_prompt)) > MAX_INPUT_TOKENS

    classifier = JailbreakClassifier.__new__(JailbreakClassifier)
    classifier.embed = embed
    classifier.classifier = mock.Mock()
    classifier.classifier.run.return_value = [np.array([0]), [{0: 0.9, 1: 0.1}]]
    monkeypatch.setattr(checks, "initialize_model", mock.Mock(return_value=classifier))
    config = RailsConfig.from_content(config={"rails": {"config": {"jailbreak_detection": {}}}})
    cache = LFUCache(maxsize=10)
    manager = LLMTaskManager(config=config)
    info_token = llm_call_info_var.set(None)
    log_token = processing_log_var.set(None)
    try:
        safe = await jailbreak_detection_model(
            manager, user_message=safe_prompt, model_caches={"jailbreak_detection": cache}
        )
        oversized = await jailbreak_detection_model(
            manager, user_message=oversized_prompt, model_caches={"jailbreak_detection": cache}
        )
    finally:
        processing_log_var.reset(log_token)
        llm_call_info_var.reset(info_token)

    assert not safe.is_blocked
    assert oversized.is_blocked and oversized.reason
    embed.model.assert_called_once()
    classifier.classifier.run.assert_called_once()
