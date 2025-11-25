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

"""This module wraps LangChain tools as actions."""

import os

from nemoguardrails.actions import action
from nemoguardrails.actions.langchain.safetools import (
    ApifyWrapperSafe,
    BingSearchAPIWrapperSafe,
    GoogleSearchAPIWrapperSafe,
    GoogleSerperAPIWrapperSafe,
    OpenWeatherMapAPIWrapperSafe,
    SearxSearchWrapperSafe,
    SerpAPIWrapperSafe,
    WikipediaAPIWrapperSafe,
    WolframAlphaAPIWrapperSafe,
    ZapierNLAWrapperSafe,
)

# TODO: Document this env variable.
if os.environ.get("NEMO_GUARDRAILS_DEMO_ACTIONS"):
    apify = action(name="apify")(ApifyWrapperSafe)
    bing_search = action(name="bing_search")(BingSearchAPIWrapperSafe)
    google_search = action(name="google_search")(GoogleSearchAPIWrapperSafe)
    searx_search = action(name="searx_search")(SearxSearchWrapperSafe)
    google_serper = action(name="google_serper")(GoogleSerperAPIWrapperSafe)
    openweather_query = action(name="openweather_query")(OpenWeatherMapAPIWrapperSafe)
    serp_api_query = action(name="serp_api_query")(SerpAPIWrapperSafe)
    wikipedia_query = action(name="wikipedia_query")(WikipediaAPIWrapperSafe)
    wolframalpha_query = action(name="wolframalpha_query")(WolframAlphaAPIWrapperSafe)
    zapier_nla_query = action(name="zapier_nla_query")(ZapierNLAWrapperSafe)
