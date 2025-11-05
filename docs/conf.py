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

# Copyright (c) 2024, NVIDIA CORPORATION.

from datetime import date

from toml import load

project = "NVIDIA NeMo Guardrails"
this_year = date.today().year
copyright = f"2023-{this_year}, NVIDIA Corporation"
author = "NVIDIA Corporation"
release = "0.0.0"
with open("../pyproject.toml") as f:
    t = load(f)
    release = t.get("tool").get("poetry").get("version")

extensions = [
    "myst_parser",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_reredirects",
]

redirects = {
    "introduction": "index.html",
    "documentation": "index.html",
    "user-guides/advanced/nemotron-content-safety-multilingual-deployment": "nemotron-safety-guard-deployment.html",
    "user-guides/advanced/nemoguard-contentsafety-deployment": "nemotron-safety-guard-deployment.html",
}

copybutton_exclude = ".linenos, .gp, .go"

exclude_patterns = [
    "README.md",
]

myst_linkify_fuzzy_links = False
myst_heading_anchors = 4
myst_enable_extensions = [
    "deflist",
    "dollarmath",
    "fieldlist",
    "substitution",
]
myst_links_external_new_tab = True

myst_substitutions = {
    "version": release,
}

exclude_patterns = [
    "_build/**",
]

myst_url_schemes = {
    "http": None,
    "https": None,
    "pr": {
        "url": "https://github.com/NVIDIA/NeMo-Guardrails/pull/{{path}}",
        "title": "PR #{{path}}",
    },
}

# intersphinx_mapping = {
#     'gpu-op': ('https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest', None),
# }

# suppress_warnings = ["etoc.toctree", "myst.header", "misc.highlighting_failure"]

templates_path = ["_templates"]

html_theme = "nvidia_sphinx_theme"
html_copy_source = False
html_show_sourcelink = False
html_show_sphinx = False

html_domain_indices = False
html_use_index = False
html_extra_path = ["project.json", "versions1.json"]
highlight_language = "console"

html_theme_options = {
    "icon_links": [],
    "switcher": {
        "json_url": "../versions1.json",
        "version_match": release,
    },
}

html_baseurl = "https://docs.nvidia.com/nemo/guardrails/latest/"
