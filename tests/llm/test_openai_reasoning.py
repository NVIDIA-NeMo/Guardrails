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

from nemoguardrails.llm.openai_reasoning import apply_openai_reasoning_overrides


class TestDrops:
    def test_drops_temperature(self):
        result = apply_openai_reasoning_overrides({"temperature": 0.5, "model": "x"})
        assert "temperature" not in result
        assert result["model"] == "x"

    def test_drops_stop(self):
        result = apply_openai_reasoning_overrides({"stop": ["END"], "model": "x"})
        assert "stop" not in result
        assert result["model"] == "x"

    def test_drops_both(self):
        result = apply_openai_reasoning_overrides({"temperature": 0.5, "stop": ["END"]})
        assert result == {}

    def test_keeps_unrelated_params(self):
        # top_p, presence_penalty, etc. are sometimes accepted by reasoning
        # models in the family; this helper does not drop them.
        params = {
            "top_p": 0.9,
            "presence_penalty": 0.1,
            "frequency_penalty": 0.1,
            "logprobs": True,
            "logit_bias": {"50256": -100},
            "n": 1,
        }
        result = apply_openai_reasoning_overrides(params)
        assert result == params


class TestMaxTokensRename:
    def test_renames_max_tokens_to_max_completion_tokens(self):
        result = apply_openai_reasoning_overrides({"max_tokens": 100})
        assert result == {"max_completion_tokens": 100}

    def test_explicit_max_completion_tokens_wins(self):
        # Caller already used the canonical name; we must not stomp on it.
        result = apply_openai_reasoning_overrides({"max_tokens": 100, "max_completion_tokens": 50})
        assert result == {"max_completion_tokens": 50}

    def test_max_completion_tokens_alone_passes_through(self):
        result = apply_openai_reasoning_overrides({"max_completion_tokens": 100})
        assert result == {"max_completion_tokens": 100}

    def test_no_max_tokens_at_all(self):
        result = apply_openai_reasoning_overrides({"model": "gpt-5-mini"})
        assert result == {"model": "gpt-5-mini"}

    def test_max_tokens_none_does_not_promote(self):
        # `None` is the Python idiom for "value not set"; renaming it would
        # send a useless null to the wire. Drop it instead.
        result = apply_openai_reasoning_overrides({"max_tokens": None})
        assert result == {}


class TestPurity:
    def test_does_not_mutate_input(self):
        params = {"temperature": 0.5, "max_tokens": 100, "model": "x"}
        snapshot = dict(params)
        apply_openai_reasoning_overrides(params)
        assert params == snapshot

    def test_returns_new_dict(self):
        params = {"model": "x"}
        result = apply_openai_reasoning_overrides(params)
        assert result is not params


class TestCombined:
    def test_drop_and_rename_together(self):
        # Realistic action-code shape: temperature + max_tokens.
        result = apply_openai_reasoning_overrides({"temperature": 0.001, "max_tokens": 1024})
        assert result == {"max_completion_tokens": 1024}
