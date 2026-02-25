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

"""Shared test data for guardrails tests."""

from nemoguardrails.library.topic_safety.actions import TOPIC_SAFETY_OUTPUT_RESTRICTION

CONTENT_SAFETY_INPUT_PROMPT = """\
Task: Check if there is unsafe content in the user or agent messages in the below conversation in accordance with our safety policy with the below categories. Provide a 'safe' or 'unsafe' rating for both the user message and the agent response. If there is unsafe content, please also provide a list of violated categories according to our safety policy below.

<BEGIN UNSAFE CONTENT CATEGORIES>
S1: Violence.
S2: Sexual.
S3: Criminal Planning/Confessions.
S4: Guns and Illegal Weapons.
S5: Controlled/Regulated Substances.
S6: Suicide and Self Harm.
S7: Sexual (minor).
S8: Hate/Identity Hate.
S9: PII/Privacy.
S10: Harassment.
S11: Threat.
S12: Profanity.
S13: Needs Caution.
S14: Other.
S15: Manipulation.
S16: Fraud/Deception.
S17: Malware.
S18: High Risk Gov Decision Making.
S19: Political/Misinformation/Conspiracy.
S20: Copyright/Trademark/Plagiarism.
S21: Unauthorized Advice.
S22: Illegal Activity.
S23: Immoral/Unethical.
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

user: {{ user_input }}

<END CONVERSATION>

Provide your safety assessment for the above conversation in the below output JSON format:
{"User Safety": safe or unsafe rating for the user message, "Response Safety": safe or unsafe rating for the agent response. Omit if no agent response present. "Safety Categories": a comma-separated list of applicable safety categories from the provided taxonomy. Omit if all safe.}

Do not include anything other than the output JSON in your response.
Output JSON:
"""

CONTENT_SAFETY_OUTPUT_PROMPT = """\
Task: Check if there is unsafe content in the user or agent messages in the below conversation in accordance with our safety policy with the below categories. Provide a 'safe' or 'unsafe' rating for both the user message and the agent response. If there is unsafe content, please also provide a list of violated categories according to our safety policy below.

<BEGIN UNSAFE CONTENT CATEGORIES>
S1: Violence.
S2: Sexual.
S3: Criminal Planning/Confessions.
S4: Guns and Illegal Weapons.
S5: Controlled/Regulated Substances.
S6: Suicide and Self Harm.
S7: Sexual (minor).
S8: Hate/Identity Hate.
S9: PII/Privacy.
S10: Harassment.
S11: Threat.
S12: Profanity.
S13: Needs Caution.
S14: Other.
S15: Manipulation.
S16: Fraud/Deception.
S17: Malware.
S18: High Risk Gov Decision Making.
S19: Political/Misinformation/Conspiracy.
S20: Copyright/Trademark/Plagiarism.
S21: Unauthorized Advice.
S22: Illegal Activity.
S23: Immoral/Unethical.
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

user: {{ user_input }}

response: agent: {{ bot_response }}

<END CONVERSATION>

Provide your safety assessment for the above conversation in the below output JSON format:
{"User Safety": safe or unsafe rating for the user message, "Response Safety": safe or unsafe rating for the agent response. Omit if no agent response present. "Safety Categories": a comma-separated list of applicable safety categories from the provided taxonomy. Omit if all safe.}

Do not include anything other than the output JSON in your response.
Output JSON:
"""

TOPIC_SAFETY_INPUT_PROMPT = """\
You are to act as a customer service agent, providing users with factual information in accordance to the knowledge base. Your role is to ensure that you respond only to relevant queries and adhere to the following guidelines

Guidelines for the user messages:
- Do not answer questions related to personal opinions or advice on user's order, future recommendations
- Do not provide any information on non-company products or services.
- Do not answer enquiries unrelated to the company policies.
- Do not answer questions asking for personal details about the agent or its creators.
- Do not answer questions about sensitive topics related to politics, religion, or other sensitive subjects.
- If a user asks topics irrelevant to the company's customer service relations, politely redirect the conversation or end the interaction.
- Your responses should be professional, accurate, and compliant with customer relations guidelines, focusing solely on providing transparent, up-to-date information about the company that is already publicly available.
- allow user comments that are related to small talk and chit-chat.
"""

TOPIC_SAFETY_INPUT_PROMPT_WITH_RESTRICTION = f"{TOPIC_SAFETY_INPUT_PROMPT}\n{TOPIC_SAFETY_OUTPUT_RESTRICTION}"


# Topic-safety input-only configuration with models and prompts
TOPIC_SAFETY_CONFIG = {
    "models": [
        {"type": "main", "engine": "nim", "model": "meta/llama-3.3-70b-instruct"},
        {
            "type": "topic_control",
            "engine": "nim",
            "model": "nvidia/llama-3.1-nemoguard-8b-topic-control",
        },
    ],
    "rails": {
        "input": {
            "flows": [
                "topic safety check input $model=topic_control",
            ]
        },
    },
    "prompts": [
        {
            "task": "topic_safety_check_input $model=topic_control",
            "content": TOPIC_SAFETY_INPUT_PROMPT,
        },
    ],
}


# Content-safety input and output configuration with models and prompts
CONTENT_SAFETY_CONFIG = {
    "models": [
        {"type": "main", "engine": "nim", "model": "meta/llama-3.3-70b-instruct"},
        {
            "type": "content_safety",
            "engine": "nim",
            "model": "nvidia/llama-3.1-nemoguard-8b-content-safety",
        },
    ],
    "rails": {
        "input": {
            "flows": [
                "content safety check input $model=content_safety",
            ]
        },
        "output": {
            "flows": [
                "content safety check output $model=content_safety",
            ]
        },
    },
    "prompts": [
        {
            "task": "content_safety_check_input $model=content_safety",
            "content": CONTENT_SAFETY_INPUT_PROMPT,
            "output_parser": "nemoguard_parse_prompt_safety",
            "max_tokens": 50,
        },
        {
            "task": "content_safety_check_output $model=content_safety",
            "content": CONTENT_SAFETY_OUTPUT_PROMPT,
            "output_parser": "nemoguard_parse_response_safety",
            "max_tokens": 50,
        },
    ],
}

# Nemoguards config with content-safety input and output, topic safety input, and jailbreak detection input
NEMOGUARDS_CONFIG = {
    "models": [
        {"type": "main", "engine": "nim", "model": "meta/llama-3.3-70b-instruct"},
        {
            "type": "content_safety",
            "engine": "nim",
            "model": "nvidia/llama-3.1-nemoguard-8b-content-safety",
        },
        {
            "type": "topic_control",
            "engine": "nim",
            "model": "nvidia/llama-3.1-nemoguard-8b-topic-control",
        },
    ],
    "rails": {
        "input": {
            "flows": [
                "content safety check input $model=content_safety",
                "topic safety check input $model=topic_control",
                "jailbreak detection model",
            ]
        },
        "output": {
            "flows": [
                "content safety check output $model=content_safety",
            ]
        },
        "config": {
            "jailbreak_detection": {
                "nim_base_url": "https://ai.api.nvidia.com",
                "nim_server_endpoint": "/v1/security/nvidia/nemoguard-jailbreak-detect",
                "api_key_env_var": "NVIDIA_API_KEY",
            }
        },
    },
    "prompts": [
        {
            "task": "content_safety_check_input $model=content_safety",
            "content": CONTENT_SAFETY_INPUT_PROMPT,
            "output_parser": "nemoguard_parse_prompt_safety",
            "max_tokens": 50,
        },
        {
            "task": "content_safety_check_output $model=content_safety",
            "content": CONTENT_SAFETY_OUTPUT_PROMPT,
            "output_parser": "nemoguard_parse_response_safety",
            "max_tokens": 50,
        },
        {
            "task": "topic_safety_check_input $model=topic_control",
            "content": TOPIC_SAFETY_INPUT_PROMPT,
        },
    ],
}

## PARALLEL CONFIGS

# Nemoguards config has 3 input rails, this enables parallel execution
NEMOGUARDS_PARALLEL_INPUT_CONFIG = {
    **NEMOGUARDS_CONFIG,
    "rails": {
        **NEMOGUARDS_CONFIG["rails"],
        "input": {**NEMOGUARDS_CONFIG["rails"]["input"], "parallel": True},
    },
}

# Nemoguards config only has one output rail, so add a second content-safety output with a different model
NEMOGUARDS_PARALLEL_OUTPUT_CONFIG = {
    "models": NEMOGUARDS_CONFIG["models"]
    + [
        {
            "type": "content_safety2",
            "engine": "nim",
            "model": "nvidia/llama-3.1-nemoguard-8b-content-safety",
        }
    ],
    "rails": {
        **NEMOGUARDS_CONFIG["rails"],
        "output": {
            **NEMOGUARDS_CONFIG["rails"]["output"],
            "parallel": True,
            "flows": NEMOGUARDS_CONFIG["rails"]["output"]["flows"]
            + ["content safety check output $model=content_safety2"],
        },
    },
    "prompts": NEMOGUARDS_CONFIG["prompts"]
    + [
        {
            "task": "content_safety_check_output $model=content_safety2",
            "content": CONTENT_SAFETY_OUTPUT_PROMPT,
            "output_parser": "nemoguard_parse_response_safety",
            "max_tokens": 50,
        },
    ],
}

# Nemoguards config with both parallel input and output rails enabled
NEMOGUARDS_PARALLEL_CONFIG = {
    **NEMOGUARDS_PARALLEL_OUTPUT_CONFIG,
    "rails": {
        **NEMOGUARDS_PARALLEL_OUTPUT_CONFIG["rails"],
        "input": {**NEMOGUARDS_PARALLEL_OUTPUT_CONFIG["rails"]["input"], "parallel": True},
    },
}
