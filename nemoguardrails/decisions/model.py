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

"""Native decision model adapters. Transport is shared, while each adapter owns its wire format."""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from nemoguardrails.decisions.types import ChoiceAnswer, ChoiceQuestion, NoulAnswer, NoulQuestion, Question
from nemoguardrails.http import HTTPClient, http_call

if TYPE_CHECKING:
    from nemoguardrails.rails.llm.config import Model

_PROBABILITY = TypeAdapter(Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)])
Answer = NoulAnswer | ChoiceAnswer


class DecisionError(RuntimeError):
    """The provider could not produce a usable judgment."""


class DecisionParameters(BaseModel):
    """Transport settings are consumed locally. Extra fields are provider body options."""

    model_config = ConfigDict(extra="allow", hide_input_in_errors=True)
    base_url: str | None = Field(default=None, min_length=1)
    timeout: float = Field(default=30.0, gt=0, allow_inf_nan=False, strict=True)

    @classmethod
    def from_deployment(cls, entry: "Model") -> "DecisionParameters":
        evidence_field = "state" if entry.engine == "system_one" else "input"
        reserved = {"model", "model_name", "questions", evidence_field}
        if conflicts := reserved.intersection(entry.parameters):
            raise ValueError(f"Decision deployment {entry.type!r} supplies rail-owned parameters: {sorted(conflicts)}")
        if "api_key" in entry.parameters:
            raise ValueError("Use api_key_env_var to configure the decision deployment credential")
        transport_only = {"default_headers", "default_query", "max_attempts", "timeout_connect"}
        if unsupported := transport_only.intersection(entry.parameters):
            raise ValueError(f"Unsupported decision transport parameters for {entry.type!r}: {sorted(unsupported)}")
        if entry.engine == "openai_decisions":
            return OpenAIDecisionParameters.model_validate(entry.parameters)
        return cls.model_validate(entry.parameters)


class OpenAIDecisionParameters(DecisionParameters):
    input_format: Literal["input_json", "text"] = "input_json"


def _choice_answer(question: ChoiceQuestion, label: str, probabilities: dict | None) -> ChoiceAnswer:
    if label not in question.choices:
        raise DecisionError("Decision model returned an undeclared choice")
    if probabilities is None:
        return ChoiceAnswer(label, None)
    if not isinstance(probabilities, dict) or probabilities.keys() != question.choices.keys():
        raise DecisionError("Decision distribution does not match the declared choices")
    return ChoiceAnswer(label, {key: _PROBABILITY.validate_python(value) for key, value in probabilities.items()})


@dataclass(frozen=True)
class DecisionModel(ABC):
    model: str
    url: str
    api_key_env_var: str
    timeout: float = 30.0
    body_defaults: dict[str, Any] = field(default_factory=dict, repr=False)

    @abstractmethod
    def _payload(self, name: str, question: Question, state: dict[str, str]) -> dict[str, Any]: ...

    @abstractmethod
    def _answer(self, name: str, question: Question, body: dict[str, Any]) -> Answer: ...

    async def decide(
        self, name: str, question: Question, state: dict[str, str], *, http_client: HTTPClient | None = None
    ) -> Answer:
        api_key = os.environ.get(self.api_key_env_var)
        if not api_key:
            raise DecisionError(f"Missing decision model credential in {self.api_key_env_var}")
        response = await http_call(
            http_client,
            "POST",
            self.url,
            headers={"Authorization": f"Bearer {api_key}"},
            json={**self.body_defaults, "model": self.model, **self._payload(name, question, state)},
            timeout=self.timeout,
        )
        if not response.is_success:
            raise DecisionError(f"Unexpected decision model status {response.status_code}")
        try:
            return self._answer(name, question, response.json())
        except (KeyError, TypeError, ValueError, AttributeError):
            # Validation errors can quote provider output containing private data.
            raise DecisionError("Invalid native decision model answer") from None


class SystemOneModel(DecisionModel):
    def _payload(self, name: str, question: Question, state: dict[str, str]) -> dict[str, Any]:
        wire_question: dict[str, Any] = {"type": question.type, "instructions": question.instructions}
        if isinstance(question, ChoiceQuestion):
            wire_question["criteria"] = question.choices
        return {"state": state, "questions": {name: wire_question}}

    def _answer(self, name: str, question: Question, body: dict[str, Any]) -> Answer:
        answer = body["answers"][name]
        if answer["type"] != question.type:
            raise DecisionError("Unexpected System One answer type")
        if isinstance(question, NoulQuestion):
            return NoulAnswer(_PROBABILITY.validate_python(answer["noul"]))
        return _choice_answer(question, answer["choice"], answer.get("probabilities"))


@dataclass(frozen=True)
class OpenAIDecisionsModel(DecisionModel):
    input_format: Literal["input_json", "text"] = "input_json"

    def _payload(self, name: str, question: Question, state: dict[str, str]) -> dict[str, Any]:
        wire_question: dict[str, Any] = {"name": name, "type": "predicate", "instructions": question.instructions}
        if isinstance(question, ChoiceQuestion):
            wire_question.update(
                type="choice",
                choices=[
                    {"value": label, "description": description} for label, description in question.choices.items()
                ],
            )
        evidence = (
            json.dumps(state, ensure_ascii=False)
            if self.input_format == "text"
            else [{"role": "user", "content": [{"type": "input_json", "data": state}]}]
        )
        return {"input": evidence, "questions": [wire_question]}

    def _answer(self, name: str, question: Question, body: dict[str, Any]) -> Answer:
        if body.get("status") not in (None, "completed"):
            raise DecisionError("OpenAI decision request did not complete")
        (answer,) = body["answers"]
        if answer["name"] != name:
            raise DecisionError("Mismatched decision answer name")
        if isinstance(question, NoulQuestion) and answer["type"] == "predicate":
            return NoulAnswer(_PROBABILITY.validate_python(answer["probability"]))
        if isinstance(question, ChoiceQuestion) and answer["type"] == "choice":
            items = answer.get("probabilities")
            distribution = None
            if items is not None:
                distribution = {item["value"]: item["probability"] for item in items}
                if len(distribution) != len(items):
                    raise DecisionError("Duplicate entries in decision distribution")
            return _choice_answer(question, answer["choice"], distribution)
        raise DecisionError("Unexpected or refused OpenAI decision answer")


DECISION_ADAPTERS = {
    "system_one": (SystemOneModel, "https://api.typesafe.ai", "systemone", "TYPESAFE_API_KEY"),
    "openai_decisions": (OpenAIDecisionsModel, "https://api.openai.com", "decisions", "OPENAI_API_KEY"),
}


def decision_models(models: list["Model"]) -> dict[str, DecisionModel]:
    result: dict[str, DecisionModel] = {}
    for entry in models:
        settings = DECISION_ADAPTERS.get(entry.engine)
        if settings is None:
            continue
        adapter, root, resource, credential = settings
        parameters = DecisionParameters.from_deployment(entry)
        base_url = (parameters.base_url or root).rstrip("/").removesuffix("/v1")
        if not entry.model:
            raise ValueError("Native decision models require a model ID")
        result[entry.type] = adapter(
            entry.model,
            f"{base_url}/v1/{resource}",
            entry.api_key_env_var or credential,
            timeout=parameters.timeout,
            body_defaults=parameters.model_extra or {},
            **({"input_format": parameters.input_format} if isinstance(parameters, OpenAIDecisionParameters) else {}),
        )
    return result
