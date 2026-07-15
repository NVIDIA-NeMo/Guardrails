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

import pytest

from nemoguardrails.manifests import normalize_configured_surface_name, parse_configured_surface


def test_parse_configured_surface_dollar_form():
    name, parameters = parse_configured_surface("self check input $model=gpt-4o")

    assert name == "self check input"
    assert parameters == {"model": "gpt-4o"}
    assert normalize_configured_surface_name("self check input $model=gpt-4o") == "self check input"


def test_parse_configured_surface_bare_name():
    assert parse_configured_surface("\tplain  flow\t") == ("plain  flow", {})
    assert parse_configured_surface("vendor/check:v2") == ("vendor/check:v2", {})


def test_parse_configured_surface_does_not_interpret_parenthesized_custom_flows():
    configured = "custom flow(model=value)"

    assert parse_configured_surface(configured) == (configured, {})
    assert normalize_configured_surface_name(configured) == configured


@pytest.mark.parametrize(
    ("configured", "expected"),
    (
        (
            'content safety check input $model="content safety" $mode=strict',
            ("content safety check input", {"model": "content safety", "mode": "strict"}),
        ),
        (
            "content safety check input $model=gpt-4o $mode=strict",
            ("content safety check input", {"model": "gpt-4o", "mode": "strict"}),
        ),
        (
            'check $pattern="foo$bar" $mode=strict',
            ("check", {"pattern": "foo$bar", "mode": "strict"}),
        ),
        (
            'check $pattern="foo bar,$x=y()" $mode=strict',
            ("check", {"pattern": "foo bar,$x=y()", "mode": "strict"}),
        ),
        (
            "check $url=https://example.com/a=(b),c?x=y==z",
            ("check", {"url": "https://example.com/a=(b),c?x=y==z"}),
        ),
        (
            'check $pattern="  padded  " $mode=strict',
            ("check", {"pattern": "  padded  ", "mode": "strict"}),
        ),
        (
            "check $value='grüezi, $name=(test)'",
            ("check", {"value": "grüezi, $name=(test)"}),
        ),
        (r"check $path=C:\models\rail", ("check", {"path": r"C:\models\rail"})),
        (r'''check $path="C:\Program Files\rail"''', ("check", {"path": r"C:\Program Files\rail"})),
        ("check $_model=v1 $mode2=2.0", ("check", {"_model": "v1", "mode2": "2.0"})),
    ),
)
def test_configured_surface_parser_supports_string_parameters(configured, expected):
    assert parse_configured_surface(configured) == expected


@pytest.mark.parametrize(
    "configured",
    (
        "",
        "   ",
        "$model=value",
        "check$model=value",
        "content safety check input $model",
        "content safety check input $model=",
        "content safety check input $",
        "content safety check input $ model=first",
        "content safety check input $model =first",
        "content safety check input $model= first",
        "content safety check input $model = first",
        "content safety check input $model=first $model=second",
        "content safety check input $model=first$mode=strict",
        'content safety check input $model="unterminated',
        "content safety check input $model='unterminated",
        'content safety check input $model=""',
        "content safety check input $model=''",
        'content safety check input $model=" "',
        "check $model=value mode=strict",
        'check $model="value"trailing',
        'check $model=foo"bar"',
        'check $model="a""b"',
        "check $1model=value",
        "check $model-name=value",
        "check $model=value$mode=strict",
        'check $model="value"$mode=strict',
        "check $model=value $$mode=strict",
        "check $model=value $mode",
    ),
)
def test_configured_surface_parser_rejects_malformed_parameters(configured):
    with pytest.raises(ValueError):
        parse_configured_surface(configured)


@pytest.mark.parametrize(
    "configured",
    (
        "\ncheck",
        "check\n",
        'check $value="line\nbreak"',
        "check\x00name",
        "check\x7fname",
    ),
)
def test_configured_surface_parser_rejects_control_characters(configured):
    with pytest.raises(ValueError):
        parse_configured_surface(configured)


def test_configured_surface_errors_do_not_include_parameter_values():
    with pytest.raises(ValueError) as error:
        parse_configured_surface('check $model="not-for-error-output" trailing')

    assert "not-for-error-output" not in str(error.value)


def test_configured_surface_parser_handles_large_inputs():
    value = "x" * 100_000

    assert parse_configured_surface(f'check $value="{value}()"') == ("check", {"value": f"{value}()"})
    with pytest.raises(ValueError):
        parse_configured_surface("check" + " " * 100_000 + "$")


@pytest.mark.parametrize(
    ("configured", "expected"),
    (
        ("check $model=value", "check"),
        ('check $pattern="value(with-parentheses)"', "check"),
        ("check(model=value)", "check(model=value)"),
        ("unknown custom flow", "unknown custom flow"),
    ),
)
def test_normalize_configured_surface_name_is_lightweight(configured, expected):
    assert normalize_configured_surface_name(configured) == expected


def test_normalize_configured_surface_name_handles_large_inputs():
    assert normalize_configured_surface_name("check" + " " * 100_000 + "$model=value") == "check"
