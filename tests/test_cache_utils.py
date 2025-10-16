# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest

from nemoguardrails.cache.utils import create_normalized_cache_key


class TestCacheUtils:
    @pytest.mark.parametrize(
        "prompt,expected",
        [
            ("Hello world", "Hello world"),
            ("", ""),
            ("   Hello world   ", "Hello world"),
            ("Hello      world      test", "Hello world test"),
            ("Hello\t\n\r world", "Hello world"),
            ("Hello    \n\t  world", "Hello world"),
        ],
    )
    def test_create_normalized_cache_key_with_whitespace_normalization(
        self, prompt, expected
    ):
        key = create_normalized_cache_key(prompt, normalize_whitespace=True)
        assert key == expected

    @pytest.mark.parametrize(
        "prompt,expected",
        [
            ("Hello world", "Hello world"),
            ("Hello    \n\t  world", "Hello    \n\t  world"),
            ("   spaces   ", "   spaces   "),
        ],
    )
    def test_create_normalized_cache_key_without_whitespace_normalization(
        self, prompt, expected
    ):
        key = create_normalized_cache_key(prompt, normalize_whitespace=False)
        assert key == expected

    @pytest.mark.parametrize(
        "prompt,expected",
        [
            (["Hello", "world"], '["Hello", "world"]'),
            ([], "[]"),
            (["Hello   world", "test"], '["Hello world", "test"]'),
        ],
    )
    def test_create_normalized_cache_key_list_input(self, prompt, expected):
        key = create_normalized_cache_key(prompt, normalize_whitespace=True)
        assert key == expected

    def test_create_normalized_cache_key_list_sorts_keys(self):
        prompt = [{"b": 2, "a": 1}, {"d": 4, "c": 3}]
        key = create_normalized_cache_key(prompt)
        assert '"a": 1' in key
        assert '"b": 2' in key

    @pytest.mark.parametrize(
        "prompt1,prompt2",
        [
            ("Hello   \n  world", "Hello     world"),
            ("test\t\nstring", "test  string"),
            ("   leading", "leading"),
        ],
    )
    def test_create_normalized_cache_key_consistent_for_same_input(
        self, prompt1, prompt2
    ):
        key1 = create_normalized_cache_key(prompt1, normalize_whitespace=True)
        key2 = create_normalized_cache_key(prompt2, normalize_whitespace=True)
        assert key1 == key2

    @pytest.mark.parametrize(
        "prompt1,prompt2",
        [
            ("Hello world", "Hello world!"),
            ("test", "testing"),
            ("case", "Case"),
        ],
    )
    def test_create_normalized_cache_key_different_for_different_input(
        self, prompt1, prompt2
    ):
        key1 = create_normalized_cache_key(prompt1)
        key2 = create_normalized_cache_key(prompt2)
        assert key1 != key2
