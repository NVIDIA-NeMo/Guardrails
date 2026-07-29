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

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def test_rail_dependencies_are_not_package_extras():
    project = tomllib.loads(Path("pyproject.toml").read_text())
    dependencies = {canonicalize_name(Requirement(value).name) for value in project["project"]["dependencies"]}
    extras = project["project"]["optional-dependencies"]
    rail_packages = {
        "presidio-analyzer",
        "presidio-anonymizer",
        "google-cloud-language",
        "yara-python",
        "fast-langdetect",
        "transformers",
        "torch",
    }

    assert rail_packages.isdisjoint(dependencies)
    assert {"sdd", "gcp", "jailbreak", "multilingual", "hf-classifier"}.isdisjoint(extras)
    assert rail_packages.isdisjoint(canonicalize_name(Requirement(value).name) for value in extras["all"])
