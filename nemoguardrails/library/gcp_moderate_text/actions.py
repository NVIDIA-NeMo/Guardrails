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

import logging
from typing import Optional

try:
    from google.cloud import language_v2
except ImportError:
    # The exception about installing google-cloud-language will be on the first call to the moderation api
    pass


from nemoguardrails.actions import action

log = logging.getLogger(__name__)


def gcp_text_moderation_mapping(result: dict) -> bool:
    """Determines whether the output from the GCP text moderation API should be blocked.

    The mapping checks two things:
      1. If the overall 'max_risk_score' exceeds 0.8.
      2. If any individual violation score (from the 'violations' dictionary) exceeds its defined threshold.
         - Toxic: > 0.8
         - Insult: > 0.7
         - Profanity: > 0.6
         - Derogatory: > 0.4
         - Violent: > 0.8
         - Sexual: > 0.7
         - Death, Harm & Tragedy: > 0.8
         - Firearms & Weapons: > 0.8
         - Illicit Drugs: > 0.8
         - Public Safety: > 0.8
         - Health: > 0.8
         - Religion & Belief: > 0.8
         - War & Conflict: > 0.8
         - Politics: > 0.8
         - Finance: > 0.8
         - Legal: > 0.8

    Returns True (i.e. block the output) if any of the conditions are met.
    """
    # Check overall risk score if available.
    max_risk_score = result.get("max_risk_score", 0)
    if max_risk_score > 0.8:
        return True

    # Check detailed violations, if provided.
    violations = result.get("violations", {})

    if violations.get("Toxic", 0) > 0.8:
        return True
    if violations.get("Insult", 0) > 0.7:
        return True
    if violations.get("Profanity", 0) > 0.6:
        return True
    if violations.get("Derogatory", 0) > 0.4:
        return True
    if violations.get("Violent", 0) > 0.8:
        return True
    if violations.get("Sexual", 0) > 0.7:
        return True
    if violations.get("Death, Harm & Tragedy", 0) > 0.8:
        return True
    if violations.get("Firearms & Weapons", 0) > 0.8:
        return True
    if violations.get("Illicit Drugs", 0) > 0.8:
        return True
    if violations.get("Public Safety", 0) > 0.8:
        return True
    if violations.get("Health", 0) > 0.8:
        return True
    if violations.get("Religion & Belief", 0) > 0.8:
        return True
    if violations.get("War & Conflict", 0) > 0.8:
        return True
    if violations.get("Politics", 0) > 0.8:
        return True
    if violations.get("Finance", 0) > 0.8:
        return True
    if violations.get("Legal", 0) > 0.8:
        return True

    # If none of the thresholds are exceeded, allow the output.
    return False


@action(
    name="call gcpnlp api",
    is_system_action=True,
    output_mapping=gcp_text_moderation_mapping,
)
async def call_gcp_text_moderation_api(
    context: Optional[dict] = None, **kwargs
) -> dict:
    """
    Application Default Credentials (ADC) is a strategy used by the GCP authentication libraries to automatically
    find credentials based on the application environment. ADC searches for credentials in the following locations (Search order):
    1. GOOGLE_APPLICATION_CREDENTIALS environment variable
    2. User credentials set up by using the Google Cloud CLI
    3. The attached service account, returned by the metadata server

    For more information check https://cloud.google.com/docs/authentication/application-default-credentials
    """
    try:
        from google.cloud import language_v2

    except ImportError:
        raise ImportError(
            "Could not import google.cloud.language_v2, please install it with "
            "`pip install google-cloud-language`."
        )

    user_message = context.get("user_message")
    client = language_v2.LanguageServiceAsyncClient()

    # Initialize request argument(s)
    document = language_v2.Document()
    document.content = user_message
    document.type_ = language_v2.Document.Type.PLAIN_TEXT

    response = await client.moderate_text(document=document)

    violations_dict = {}
    max_risk_score = 0.0
    for violation in response.moderation_categories:
        if violation.confidence > max_risk_score:
            max_risk_score = violation.confidence
        violations_dict[violation.name] = violation.confidence

    return {"max_risk_score": max_risk_score, "violations": violations_dict}
