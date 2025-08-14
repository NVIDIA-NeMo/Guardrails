# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""OpenTelemetry constants and semantic conventions for NeMo Guardrails."""


class SpanKind:
    """String constants for span kinds."""

    SERVER = "server"
    CLIENT = "client"
    INTERNAL = "internal"


class SpanTypes:
    """Internal span type identifiers used in span mapping.

    These are internal identifiers used to categorize spans before mapping
    to actual span names. They represent the type of operation being traced.

    Note: 'llm_call' maps to various GenAI semantic convention span types
    like inference (gen_ai.inference.client), embeddings, etc.
    """

    # NeMo Guardrails-specific internal types
    INTERACTION = "interaction"  # Entry point to guardrails
    RAIL = "rail"  # Rail execution
    ACTION = "action"  # Action execution

    # GenAI-related type (maps to official semantic conventions)
    LLM_CALL = "llm_call"  # maps to gen_ai.inference.client

    # NOTE: might use more specific types in the future
    # could add more specific types that align with semantic conventions:
    # INFERENCE = "inference"  # for gen_ai.inference.client spans
    # EMBEDDING = "embedding"  # for gen_ai.embeddings.client spans


class SpanNamePatterns:
    """Patterns used for identifying span types from span names."""

    # patterns that indicate SERVER spans
    INTERACTION = "interaction"
    GUARDRAILS_REQUEST_PATTERN = "guardrails.request"

    # patterns that indicate CLIENT spans
    GEN_AI_PREFIX = "gen_ai."
    LLM = "llm"
    COMPLETION = "completion"


class SystemConstants:
    """System-level constants for NeMo Guardrails."""

    SYSTEM_NAME = "nemo-guardrails"
    UNKNOWN = "unknown"


class GenAIAttributes:
    """GenAI semantic convention attributes following the draft specification.

    Note: These are based on the experimental OpenTelemetry GenAI semantic conventions
    since they are not yet available in the stable semantic conventions package.

    See: https://opentelemetry.io/docs/specs/semconv/gen-ai/
    """

    GEN_AI_SYSTEM = "gen_ai.system"  # @deprecated

    GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
    GEN_AI_OPERATION_NAME = "gen_ai.operation.name"

    GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
    GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
    GEN_AI_REQUEST_TOP_K = "gen_ai.request.top_k"
    GEN_AI_REQUEST_FREQUENCY_PENALTY = "gen_ai.request.frequency_penalty"
    GEN_AI_REQUEST_PRESENCE_PENALTY = "gen_ai.request.presence_penalty"
    GEN_AI_REQUEST_STOP_SEQUENCES = "gen_ai.request.stop_sequences"

    GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
    GEN_AI_RESPONSE_ID = "gen_ai.response.id"
    GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"

    GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
    GEN_AI_USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"


class CommonAttributes:
    """Common OpenTelemetry attributes used across spans."""

    SPAN_KIND = "span.kind"


class GuardrailsAttributes:
    """NeMo Guardrails-specific attributes for spans."""

    # rail attributes
    RAIL_TYPE = "rail.type"
    RAIL_NAME = "rail.name"
    RAIL_STOP = "rail.stop"
    RAIL_DECISIONS = "rail.decisions"

    # action attributes
    ACTION_NAME = "action.name"
    ACTION_HAS_LLM_CALLS = "action.has_llm_calls"
    ACTION_LLM_CALLS_COUNT = "action.llm_calls_count"
    ACTION_PARAM_PREFIX = "action.param."  # For dynamic action parameters


class SpanNames:
    """Standard span names following OpenTelemetry GenAI semantic conventions.

    Based on: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/

    IMPORTANT: Span names must be low cardinality to avoid performance issues.
    Variable/high cardinality data (like specific rail types, model names, etc.)
    should go in attributes instead of the span name.
    """

    # server spans (entry points); NeMo Guardrails specific
    GUARDRAILS_REQUEST = "guardrails.request"  # Entry point for guardrails processing

    # internal spans; NeMo Guardrails specific
    GUARDRAILS_RAIL = "guardrails.rail"  # Use attributes for rail type/name
    GUARDRAILS_ACTION = "guardrails.action"  # Use attributes for action name

    # client spans (LLM calls), following official GenAI semantic conventions
    # "Span name SHOULD be `{gen_ai.operation.name} {gen_ai.request.model}`"
    # since model names are high cardinality, we'll build these dynamically
    # these are fallback operation names when model is unknown
    GEN_AI_COMPLETION = "completion"
    GEN_AI_CHAT = "chat"
    GEN_AI_EMBEDDING = "embedding"


class OperationNames:
    """Standard operation names for GenAI semantic conventions.

    Note: This only defines standard LLM operations. Custom actions and tasks
    should be passed through as-is since they are dynamic and user-defined.
    """

    # standard LLM operations (from GenAI semantic conventions)
    COMPLETION = "completion"
    CHAT = "chat"
    EMBEDDING = "embedding"

    # default operation for guardrails interactions
    GUARDRAILS = "guardrails"


class EventNames:
    """Standard event names for OpenTelemetry GenAI semantic conventions.

    Based on official spec at:
    https://github.com/open-telemetry/semantic-conventions/blob/main/model/gen-ai/events.yaml
    """

    GEN_AI_SYSTEM_MESSAGE = "gen_ai.system.message"
    GEN_AI_USER_MESSAGE = "gen_ai.user.message"
    GEN_AI_ASSISTANT_MESSAGE = "gen_ai.assistant.message"
    # GEN_AI_TOOL_MESSAGE = "gen_ai.tool.message"

    GEN_AI_CHOICE = "gen_ai.choice"

    GEN_AI_CONTENT_PROMPT = "gen_ai.content.prompt"  # @deprecated ; use GEN_AI_USER_MESSAGE  instead, as we are still using text completions we should use it!
    GEN_AI_CONTENT_COMPLETION = "gen_ai.content.completion"  # @deprecated ; use GEN_AI_ASSISTANT_MESSAGE, but as we are still using text completions we should use it!


class GuardrailsEventNames:
    """NeMo Guardrails-specific event names (not OTel GenAI conventions).

    These events represent internal guardrails state changes, not LLM API calls.
    They use a guardrails-specific namespace to avoid confusion with OTel GenAI semantic conventions.
    """

    UTTERANCE_USER_FINISHED = "guardrails.utterance.user.finished"
    UTTERANCE_BOT_STARTED = "guardrails.utterance.bot.started"
    UTTERANCE_BOT_FINISHED = "guardrails.utterance.bot.finished"

    USER_MESSAGE = "guardrails.user_message"


class GuardrailsEventTypes:
    """NeMo Guardrails internal event type constants.

    These are the type values from internal guardrails events.
    """

    UTTERANCE_USER_ACTION_FINISHED = "UtteranceUserActionFinished"
    USER_MESSAGE = "UserMessage"

    START_UTTERANCE_BOT_ACTION = "StartUtteranceBotAction"
    UTTERANCE_BOT_ACTION_FINISHED = "UtteranceBotActionFinished"

    SYSTEM_MESSAGE = "SystemMessage"
