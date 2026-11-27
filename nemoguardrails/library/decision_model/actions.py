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

from nemoguardrails.actions import action
from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.decisions.model import DecisionModel
from nemoguardrails.decisions.types import ChoiceAnswer, ChoiceQuestion, NoulAnswer, NoulQuestion
from nemoguardrails.http import HTTPClient
from nemoguardrails.library.decision_model.rail_config import (
    ChoiceQuestionConfig,
    ChoiceRule,
    DecisionRule,
    ProbabilityRule,
)
from nemoguardrails.rails.llm.config import RailsConfig


@action(is_system_action=True)
async def decision_model_check_output(
    rule: str,
    model_name: str,
    config: RailsConfig,
    decision_models: dict[str, DecisionModel],
    bot_message: str,
    http_client: HTTPClient | None = None,
) -> RailOutcome:
    definition: DecisionRule = config.rails.config.decision_model.rules[rule]
    state = {"bot_message": bot_message}
    spec = definition.question
    if isinstance(spec, ChoiceQuestionConfig):
        question = ChoiceQuestion(spec.instructions, spec.choices)
    else:
        question = NoulQuestion(spec.instructions)
    answer = await decision_models[model_name].decide(rule, question, state, http_client=http_client)
    condition = definition.block_if
    if isinstance(answer, NoulAnswer) and isinstance(condition, ProbabilityRule):
        blocked = answer.probability >= condition.probability_at_least
    elif isinstance(answer, ChoiceAnswer) and isinstance(condition, ChoiceRule):
        blocked = answer.choice in condition.choice_in
    else:
        raise ValueError("Decision answer does not match the configured rule")
    outcome = RailOutcome.block if blocked else RailOutcome.allow
    return outcome(metadata={"rule": rule})
