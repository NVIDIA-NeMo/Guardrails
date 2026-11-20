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


"""Dogwood's reference examples, replayed through the binding the hooks server decides through.

Each bundle under `dogwood_conformance/` is a policy, its schemas, an event trace, and the verdicts the reference
`dogwood replay` CLI gives (see its README). Every one of the 56 must give the same verdict at every decision point.
"""

import json
import re
from pathlib import Path

import pytest

dogwood = pytest.importorskip("nemoguardrails_dogwood")

BUNDLES = sorted(p for p in (Path(__file__).parent / "dogwood_conformance").iterdir() if p.is_dir())


def _optional(bundle: Path, *names: str):
    return next(((bundle / name).read_text() for name in names if (bundle / name).exists()), None)


def test_every_reference_example_is_here():
    assert len(BUNDLES) == 56


@pytest.mark.parametrize("bundle", BUNDLES, ids=lambda p: p.name)
def test_the_binding_gives_the_reference_verdicts(bundle):
    report = json.loads(
        dogwood.replay(
            (bundle / "policy.dw").read_text(),
            (bundle / "schema.cedarschema").read_text(),
            (bundle / "trace.log").read_text(),
            _optional(bundle, "events.dwschema", "event.dwschema"),
            _optional(bundle, "providers.json"),
            _optional(bundle, "macros.dw"),
        )
    )
    got = [(verdict["timestamp"], verdict["verdict"].upper()) for verdict in report["verdicts"]]
    want = [
        (int(match.group(1)), match.group(2))
        for line in (bundle / "expected.out").read_text().splitlines()
        if (match := re.match(r"@(-?\d+) \(time point \d+\): (ALLOW|DENY)", line))
    ]
    assert want and got == want
