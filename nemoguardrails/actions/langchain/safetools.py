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

"""This module contains safer versions for some of the most used LangChain tools.

The same validation logic can be applied to others as well.
"""

import logging
from typing import TYPE_CHECKING

from nemoguardrails.actions.validation import validate_input, validate_response

log = logging.getLogger(__name__)

# Include these outside the try .. except so the Type-checker knows they're always imported
if TYPE_CHECKING:
    from langchain_community.utilities import (
        ApifyWrapper,
        BingSearchAPIWrapper,
        GoogleSearchAPIWrapper,
        GoogleSerperAPIWrapper,
        OpenWeatherMapAPIWrapper,
        SearxSearchWrapper,
        SerpAPIWrapper,
        WikipediaAPIWrapper,
        WolframAlphaAPIWrapper,
        ZapierNLAWrapper,
    )

try:
    from langchain_community.utilities import (
        ApifyWrapper,
        BingSearchAPIWrapper,
        GoogleSearchAPIWrapper,
        GoogleSerperAPIWrapper,
        OpenWeatherMapAPIWrapper,
        SearxSearchWrapper,
        SerpAPIWrapper,
        WikipediaAPIWrapper,
        WolframAlphaAPIWrapper,
        ZapierNLAWrapper,
    )
except ImportError:
    log.warning(
        "The langchain_community module is not installed. Please install it using pip: pip install langchain_community"
    )


MAX_QUERY_LEN = 50
MAX_LOCATION_LEN = 50


@validate_input("actor_id", validators=["length"], max_len=MAX_QUERY_LEN)
class ApifyWrapperSafe(ApifyWrapper):
    """Safer version for the ApifyWrapper."""


@validate_input("query", validators=["length"], max_len=MAX_QUERY_LEN)
@validate_response(validators=["ip_filter", "is_default_resp"])
class BingSearchAPIWrapperSafe(BingSearchAPIWrapper):
    """Safer version for the BingSearch API wrapper."""


@validate_input("query", validators=["length"], max_len=MAX_QUERY_LEN)
@validate_response(validators=["ip_filter", "is_default_resp"])
class GoogleSearchAPIWrapperSafe(GoogleSearchAPIWrapper):
    """Safer version for the Google Search API wrapper."""


@validate_input("query", validators=["length"], max_len=MAX_QUERY_LEN)
@validate_response(validators=["ip_filter", "is_default_resp"])
class SearxSearchWrapperSafe(SearxSearchWrapper):
    """Safer version for the Searx Search wrapper"""


@validate_input("query", validators=["length"], max_len=MAX_QUERY_LEN)
@validate_response(validators=["ip_filter", "is_default_resp"])
class GoogleSerperAPIWrapperSafe(GoogleSerperAPIWrapper):
    """Safer version for the Google Serper API wrapper."""


@validate_input("location", validators=["length"], max_len=MAX_LOCATION_LEN)
@validate_response(validators=["ip_filter", "is_default_resp"])
class OpenWeatherMapAPIWrapperSafe(OpenWeatherMapAPIWrapper):
    """Safer version for the OpenWeatherMap API wrapper."""


@validate_input("query", validators=["length"], max_len=MAX_QUERY_LEN)
@validate_response(validators=["ip_filter", "is_default_resp"])
class SerpAPIWrapperSafe(SerpAPIWrapper):
    """Safer version for the SerpAPI wrapper."""


@validate_input("query", validators=["length"], max_len=MAX_QUERY_LEN)
@validate_response(validators=["ip_filter", "is_default_resp"])
class WikipediaAPIWrapperSafe(WikipediaAPIWrapper):
    """Safer version for the Wikipedia API wrapper."""


@validate_input("query", validators=["length"], max_len=MAX_QUERY_LEN)
@validate_response(validators=["ip_filter", "is_default_resp"])
class WolframAlphaAPIWrapperSafe(WolframAlphaAPIWrapper):
    """Safer version for the Wolfram Alpha API wrapper."""


@validate_input("instructions", validators=["length"], max_len=MAX_QUERY_LEN)
@validate_input("action_id")
@validate_response(validators=["ip_filter"])
class ZapierNLAWrapperSafe(ZapierNLAWrapper):
    """Safer version for the Zapier NLA Wrapper."""
