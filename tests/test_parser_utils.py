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

from nemoguardrails.colang.v1_0.lang.utils import split_args


def test_1():
    assert split_args("1") == ["1"]
    assert split_args('1, "a"') == ["1", '"a"']
    assert split_args("1, [1,2,3]") == ["1", "[1,2,3]"]
    assert split_args("1, numbers = [1,2,3]") == ["1", "numbers = [1,2,3]"]
    assert split_args("1, data = {'name': 'John'}") == ["1", "data = {'name': 'John'}"]
    assert split_args("'a,b, c'") == ["'a,b, c'"]

    assert split_args("1, 'a,b, c', x=[1,2,3], data = {'name': 'John'}") == [
        "1",
        "'a,b, c'",
        "x=[1,2,3]",
        "data = {'name': 'John'}",
    ]
