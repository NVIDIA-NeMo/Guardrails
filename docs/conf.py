# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import sys
from datetime import date
from pathlib import Path

from toml import load

# Add local extensions to path
sys.path.insert(0, str(Path(__file__).parent / "_extensions"))

# Add the project root to path so autodoc can import nemoguardrails
sys.path.insert(0, str(Path(__file__).parent.parent))

project = "NVIDIA NeMo Guardrails Library Developer Guide"
this_year = date.today().year
copyright = f"2023-{this_year}, NVIDIA Corporation"
author = "NVIDIA Corporation"
release = "0.0.0"
with open("../pyproject.toml") as f:
    t = load(f)
    release = t.get("tool").get("poetry").get("version")

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_reredirects",
    "sphinx_design",
    "sphinxcontrib.mermaid",
    "json_output",
    "search_assets",  # Enhanced search assets extension
    "validate_redirects",
]

# -- Autodoc configuration ---------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_class_signature = "separated"

# Mock imports for optional dependencies that may not be installed
autodoc_mock_imports = [
    "presidio_analyzer",
    "presidio_anonymizer",
    "spacy",
    "google.cloud",
    "yara",
    "fast_langdetect",
    "opentelemetry",
    "streamlit",
    "tqdm",
]

# -- Autosummary configuration -----------------------------------------------
autosummary_generate = True
autosummary_generate_overwrite = True
autosummary_imported_members = True

redirects = {
    "introduction": "index.html",
    "documentation": "index.html",
    "readme": "index.html",
    # Pre-revamp redirects
    "user-guides/advanced/nemotron-content-safety-multilingual-deployment": "getting-started/tutorials/nemotron-safety-guard-deployment.html",
    "user-guides/advanced/nemoguard-contentsafety-deployment": "getting-started/tutorials/nemotron-safety-guard-deployment.html",
    # Top-level pages
    "architecture": "about/how-it-works.html",
    "architecture/readme": "reference/colang-architecture-guide.html",
    "faqs": "resources/faqs.html",
    "glossary": "resources/glossary.html",
    "release-notes": "about/release-notes.html",
    "security/guidelines": "resources/security/guidelines.html",
    # Getting started
    "getting-started": "getting-started/installation-guide.html",
    "getting-started/readme": "getting-started/installation-guide.html",
    # Colang 1 tutorials (getting-started/N-* → configure-rails/colang/colang-1/tutorials/N-*)
    "getting-started/1-hello-world/readme": "configure-rails/colang/colang-1/tutorials/1-hello-world/readme.html",
    "getting-started/2-core-colang-concepts/readme": "configure-rails/colang/colang-1/tutorials/2-core-colang-concepts/readme.html",
    "getting-started/3-demo-use-case/readme": "configure-rails/colang/colang-1/tutorials/3-demo-use-case/readme.html",
    "getting-started/4-input-rails": "configure-rails/colang/colang-1/tutorials/4-input-rails/readme.html",
    "getting-started/4-input-rails/readme": "configure-rails/colang/colang-1/tutorials/4-input-rails/readme.html",
    "getting-started/5-output-rails/readme": "configure-rails/colang/colang-1/tutorials/5-output-rails/readme.html",
    "getting-started/6-topical-rails/readme": "configure-rails/colang/colang-1/tutorials/6-topical-rails/readme.html",
    "getting-started/7-rag/readme": "configure-rails/colang/colang-1/tutorials/7-rag/readme.html",
    # Colang 2 (colang-2/* → configure-rails/colang/colang-2/*)
    "colang-2": "configure-rails/colang/colang-2/index.html",
    "colang-2/overview": "configure-rails/colang/colang-2/index.html",
    "colang-2/getting-started": "configure-rails/colang/colang-2/getting-started/index.html",
    "colang-2/getting-started/dialog-rails": "configure-rails/colang/colang-2/getting-started/dialog-rails.html",
    "colang-2/getting-started/hello-world": "configure-rails/colang/colang-2/getting-started/hello-world.html",
    "colang-2/getting-started/input-rails": "configure-rails/colang/colang-2/getting-started/input-rails.html",
    "colang-2/getting-started/interaction-loop": "configure-rails/colang/colang-2/getting-started/interaction-loop.html",
    "colang-2/getting-started/llm-flows": "configure-rails/colang/colang-2/getting-started/llm-flows.html",
    "colang-2/getting-started/multimodal-rails": "configure-rails/colang/colang-2/getting-started/multimodal-rails.html",
    "colang-2/getting-started/recommended-next-steps": "configure-rails/colang/colang-2/getting-started/recommended-next-steps.html",
    "colang-2/whats-changed": "configure-rails/colang/colang-2/whats-changed.html",
    "colang-2/language-reference": "configure-rails/colang/colang-2/language-reference/index.html",
    "colang-2/language-reference/introduction": "configure-rails/colang/colang-2/language-reference/introduction.html",
    "colang-2/language-reference/defining-flows": "configure-rails/colang/colang-2/language-reference/defining-flows.html",
    "colang-2/language-reference/development-and-debugging": "configure-rails/colang/colang-2/language-reference/development-and-debugging.html",
    "colang-2/language-reference/event-generation-and-matching": "configure-rails/colang/colang-2/language-reference/event-generation-and-matching.html",
    "colang-2/language-reference/flow-control": "configure-rails/colang/colang-2/language-reference/flow-control.html",
    "colang-2/language-reference/make-use-of-llms": "configure-rails/colang/colang-2/language-reference/make-use-of-llms.html",
    "colang-2/language-reference/more-on-flows": "configure-rails/colang/colang-2/language-reference/more-on-flows.html",
    "colang-2/language-reference/python-actions": "configure-rails/colang/colang-2/language-reference/python-actions.html",
    "colang-2/language-reference/the-standard-library": "configure-rails/colang/colang-2/language-reference/the-standard-library.html",
    "colang-2/language-reference/working-with-actions": "configure-rails/colang/colang-2/language-reference/working-with-actions.html",
    "colang-2/language-reference/working-with-variables-and-expressions": "configure-rails/colang/colang-2/language-reference/working-with-variables-and-expressions.html",
    "colang-2/language-reference/csl/attention": "configure-rails/colang/colang-2/language-reference/csl/attention.html",
    "colang-2/language-reference/csl/avatars": "configure-rails/colang/colang-2/language-reference/csl/avatars.html",
    "colang-2/language-reference/csl/core": "configure-rails/colang/colang-2/language-reference/csl/core.html",
    "colang-2/language-reference/csl/guardrails": "configure-rails/colang/colang-2/language-reference/csl/guardrails.html",
    "colang-2/language-reference/csl/lmm": "configure-rails/colang/colang-2/language-reference/csl/lmm.html",
    "colang-2/language-reference/csl/timing": "configure-rails/colang/colang-2/language-reference/csl/timing.html",
    # Configuration guide (user-guides/configuration-guide/* → configure-rails/*)
    "user-guides/configuration-guide": "configure-rails/index.html",
    "user-guides/configuration-guide/custom-initialization": "configure-rails/custom-initialization/index.html",
    "user-guides/configuration-guide/exceptions": "configure-rails/exceptions.html",
    "user-guides/configuration-guide/general-options": "configure-rails/configuration-reference.html",
    "user-guides/configuration-guide/guardrails-configuration": "configure-rails/yaml-schema/guardrails-configuration/index.html",
    "user-guides/configuration-guide/knowledge-base": "configure-rails/other-configurations/knowledge-base.html",
    "user-guides/configuration-guide/llm-configuration": "configure-rails/yaml-schema/model-configuration.html",
    "user-guides/configuration-guide/tracing-configuration": "configure-rails/yaml-schema/tracing-configuration.html",
    # LangChain (user-guides/langchain/* → integration/langchain/*)
    "user-guides/langchain": "integration/langchain/index.html",
    "user-guides/langchain/langchain-integration": "integration/langchain/langchain-integration.html",
    "user-guides/langchain/langgraph-integration": "integration/langchain/langgraph-integration.html",
    "user-guides/langchain/runnable-rails": "integration/langchain/runnable-rails.html",
    "user-guides/langchain/runnable-as-action": "integration/langchain/runnable-as-action/index.html",
    "user-guides/langchain/runnable-as-action/readme": "integration/langchain/runnable-as-action/index.html",
    "user-guides/langchain/chain-with-guardrails": "integration/langchain/chain-with-guardrails/index.html",
    "user-guides/langchain/chain-with-guardrails/readme": "integration/langchain/chain-with-guardrails/index.html",
    # Tracing (user-guides/tracing/* → observability/tracing/*)
    "user-guides/tracing": "observability/tracing/index.html",
    "user-guides/tracing/adapter-configurations": "observability/tracing/adapter-configurations.html",
    "user-guides/tracing/opentelemetry-integration": "observability/tracing/opentelemetry-integration.html",
    "user-guides/tracing/quick-start": "observability/tracing/quick-start.html",
    "user-guides/tracing/troubleshooting": "observability/tracing/troubleshooting.html",
    # Logging (user-guides/detailed-logging/* → observability/logging/*)
    "user-guides/detailed-logging": "observability/logging/index.html",
    "user-guides/detailed-logging/readme": "observability/logging/readme.html",
    # Advanced user guides → various new locations
    "user-guides/advanced/bot-message-instructions": "configure-rails/colang/usage-examples/bot-message-instructions.html",
    "user-guides/advanced/bot-thinking-guardrails": "configure-rails/colang/colang-1/bot-thinking-guardrails.html",
    "user-guides/advanced/embedding-search-providers": "configure-rails/other-configurations/embedding-search-providers.html",
    "user-guides/advanced/event-based-api": "run-rails/using-python-apis/event-based-api.html",
    "user-guides/advanced/extract-user-provided-values": "configure-rails/colang/usage-examples/extract-user-provided-values.html",
    "user-guides/advanced/generation-options": "run-rails/using-python-apis/generation-options.html",
    "user-guides/advanced/jailbreak-detection-deployment": "getting-started/tutorials/nemoguard-jailbreakdetect-deployment.html",
    "user-guides/advanced/kv-cache-reuse": "configure-rails/caching/kv-cache-reuse.html",
    "user-guides/advanced/llama-guard-deployment": "configure-rails/guardrail-catalog/community/llama-guard.html",
    "user-guides/advanced/model-memory-cache": "configure-rails/caching/model-memory-cache.html",
    "user-guides/advanced/nemoguard-jailbreakdetect-deployment": "getting-started/tutorials/nemoguard-jailbreakdetect-deployment.html",
    "user-guides/advanced/nemoguard-topiccontrol-deployment": "getting-started/tutorials/nemoguard-topiccontrol-deployment.html",
    "user-guides/advanced/nemotron-safety-guard-deployment": "getting-started/tutorials/nemotron-safety-guard-deployment.html",
    "user-guides/advanced/nested-async-loop": "troubleshooting.html",
    "user-guides/advanced/streaming": "run-rails/using-python-apis/streaming.html",
    "user-guides/advanced/tools-integration": "integration/tools-integration.html",
    "user-guides/advanced/using-docker": "deployment/using-docker.html",
    "user-guides/advanced/vertexai-setup": "about/supported-llms.html",
    # Other user guides
    "user-guides/cli": "reference/cli/index.html",
    "user-guides/colang-language-syntax-guide": "configure-rails/colang/colang-1/colang-language-syntax-guide.html",
    "user-guides/guardrails-library": "configure-rails/guardrail-catalog/index.html",
    "user-guides/guardrails-process": "about/how-it-works.html",
    "user-guides/llm-support": "about/supported-llms.html",
    "user-guides/llm": "about/supported-llms.html",
    "user-guides/llm/nvidia-ai-endpoints": "about/supported-llms.html",
    "user-guides/llm/nvidia-ai-endpoints/readme": "about/supported-llms.html",
    "user-guides/llm/vertexai": "about/supported-llms.html",
    "user-guides/llm/vertexai/readme": "about/supported-llms.html",
    "user-guides/migration-guide": "configure-rails/colang/colang-2/migration-guide.html",
    "user-guides/multi-config-api": "run-rails/using-python-apis/index.html",
    "user-guides/multi-config-api/readme": "run-rails/using-python-apis/index.html",
    "user-guides/multimodal": "getting-started/tutorials/multimodal.html",
    "user-guides/python-api": "reference/python-api/index.html",
    "user-guides/server-guide": "run-rails/using-fastapi-server/index.html",
    # API reference (api/* → reference/python-api)
    "api/nemoguardrails.rails.llm.config": "reference/python-api/index.html",
    "api/nemoguardrails.rails.llm.llmrails": "reference/python-api/index.html",
    "api/nemoguardrails.streaming": "reference/python-api/index.html",
}

copybutton_exclude = ".linenos, .gp, .go"

exclude_patterns = [
    "README.md",
    "_build/**",
    "_extensions/**",
    "LIVE_DOCS.md",
    "research.md",
    "scripts/**",
    "docs-structure-context.md",
    "user-guides/**",
    "**/.cursor/**",
]

myst_linkify_fuzzy_links = False
myst_heading_anchors = 4
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "substitution",
]
myst_links_external_new_tab = True

myst_substitutions = {
    "version": release,
}

myst_url_schemes = {
    "http": None,
    "https": None,
    "mailto": None,
    "pr": {
        "url": "https://github.com/NVIDIA-NeMo/Guardrails/pull/{{path}}",
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
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/NVIDIA-NeMo/Guardrails",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/nemoguardrails/",
            "icon": "fa-brands fa-python",
            "type": "fontawesome",
        },
    ],
    "switcher": {
        "json_url": "../versions1.json",
        "version_match": release,
    },
}

html_baseurl = "https://docs.nvidia.com/nemo/guardrails/latest/"

# JSON output extension settings
json_output_settings = {
    "enabled": True,
    "verbose": True,
}
