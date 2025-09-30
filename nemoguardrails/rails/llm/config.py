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

"""Module for the configuration of rails."""

import logging
import os
import warnings
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    model_validator,
    root_validator,
    validator,
)
from pydantic.fields import Field

from nemoguardrails import utils
from nemoguardrails.colang import parse_colang_file, parse_flow_elements
from nemoguardrails.colang.v2_x.lang.utils import format_colang_parsing_error_message
from nemoguardrails.colang.v2_x.runtime.errors import ColangParsingError
from nemoguardrails.llm.types import Task

log = logging.getLogger(__name__)

# Load the default config values from the file
with open(os.path.join(os.path.dirname(__file__), "default_config.yml")) as _fc:
    _default_config = yaml.safe_load(_fc)

with open(os.path.join(os.path.dirname(__file__), "default_config_v2.yml")) as _fc:
    _default_config_v2 = yaml.safe_load(_fc)


# Extract the COLANGPATH directories.
colang_path_dirs = [
    _path.strip()
    for _path in os.environ.get("COLANGPATH", "").split(os.pathsep)
    if _path.strip() != ""
]

# We also make sure that the standard library is in the COLANGPATH.
standard_library_path = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "colang", "v2_x", "library")
)

# nemoguardrails/library
guardrails_stdlib_path = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
colang_path_dirs.append(standard_library_path)
colang_path_dirs.append(guardrails_stdlib_path)


class Model(BaseModel):
    """Configuration of a model used by the rails engine.

    Typically, the main model is configured e.g.:
    {
        "type": "main",
        "engine": "openai",
        "model": "gpt-3.5-turbo-instruct"
    }
    """

    type: str
    engine: str
    model: Optional[str] = Field(
        default=None,
        description="The name of the model. If not specified, it should be specified through the parameters attribute.",
    )
    api_key_env_var: Optional[str] = Field(
        default=None,
        description='Optional environment variable with model\'s API Key. Do not include "$".',
    )
    parameters: Dict[str, Any] = Field(default_factory=dict)

    mode: Literal["chat", "text"] = Field(
        default="chat",
        description="Whether the mode is 'text' completion or 'chat' completion. Allowed values are 'chat' or 'text'.",
    )

    @model_validator(mode="before")
    @classmethod
    def set_and_validate_model(cls, data: Any) -> Any:
        if isinstance(data, dict):
            parameters = data.get("parameters")
            if parameters is None:
                return data
            model_field = data.get("model")
            model_from_params = parameters.get("model_name") or parameters.get("model")

            if model_field and model_from_params:
                raise ValueError(
                    "Model name must be specified in exactly one place: either in the 'model' field or in parameters, not both."
                )
            if not model_field and model_from_params:
                data["model"] = model_from_params
                if (
                    "model_name" in parameters
                    and parameters["model_name"] == model_from_params
                ):
                    parameters.pop("model_name")
                elif "model" in parameters and parameters["model"] == model_from_params:
                    parameters.pop("model")
            return data

    @model_validator(mode="after")
    def model_must_be_none_empty(self) -> "Model":
        """Validate that a model name is present either directly or in parameters."""
        if not self.model or not self.model.strip():
            raise ValueError(
                "Model name must be specified either directly in the 'model' field or through 'model_name'/'model' in parameters"
            )
        return self


class Instruction(BaseModel):
    """Configuration for instructions in natural language that should be passed to the LLM."""

    type: str
    content: str


class Document(BaseModel):
    """Configuration for documents that should be used for question answering."""

    format: str
    content: str


class InjectionDetection(BaseModel):
    injections: List[str] = Field(
        default_factory=list,
        description="The list of injection types to detect. Options are 'sqli', 'template', 'code', 'xss'."
        "Currently, only SQL injection, template injection, code injection, "
        "and markdown cross-site scripting are supported. "
        "Custom rules can be added, provided they are in the `yara_path` and have a `.yara` file extension.",
    )
    action: str = Field(
        default="reject",
        pattern=r"^(reject|omit)$",
        description="Action to take. Options are 'reject' to offer a rejection message, "
        "'omit' to mask the offending content, and 'sanitize' to pass the content as-is in the safest way. "
        "These options are listed in descending order of relative safety. 'sanitize' is not implemented at this time.",
    )
    yara_path: Optional[str] = Field(
        default="",
        description="Location on disk where YARA rules are located. If this parameter is an empty string, "
        "the default location defined in injection_detection's actions.py file will be used.",
    )
    yara_rules: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        description="Dictionary mapping rule names to YARA rule strings. If provided, these rules will be used "
        "instead of loading rules from yara_path. Each rule should be a valid YARA rule string.",
    )


class SensitiveDataDetectionOptions(BaseModel):
    entities: List[str] = Field(
        default_factory=list,
        description="The list of entities that should be detected. "
        "Check out https://microsoft.github.io/presidio/supported_entities/ for"
        "the list of supported entities.",
    )
    # TODO: this is not currently in use.
    mask_token: str = Field(
        default="*",
        description="The token that should be used to mask the sensitive data.",
    )

    score_threshold: float = Field(
        default=0.2,
        description="The score threshold that should be used to detect the sensitive data.",
    )


class SensitiveDataDetection(BaseModel):
    """Configuration of what sensitive data should be detected."""

    recognizers: List[dict] = Field(
        default_factory=list,
        description="Additional custom recognizers. "
        "Check out https://microsoft.github.io/presidio/tutorial/08_no_code/ for more details.",
    )
    input: SensitiveDataDetectionOptions = Field(
        default_factory=SensitiveDataDetectionOptions,
        description="Configuration of the entities to be detected on the user input.",
    )
    output: SensitiveDataDetectionOptions = Field(
        default_factory=SensitiveDataDetectionOptions,
        description="Configuration of the entities to be detected on the bot output.",
    )
    retrieval: SensitiveDataDetectionOptions = Field(
        default_factory=SensitiveDataDetectionOptions,
        description="Configuration of the entities to be detected on retrieved relevant chunks.",
    )


class PrivateAIDetectionOptions(BaseModel):
    """Configuration options for Private AI."""

    entities: List[str] = Field(
        default_factory=list,
        description="The list of entities that should be detected.",
    )


class PrivateAIDetection(BaseModel):
    """Configuration for Private AI."""

    server_endpoint: str = Field(
        default="http://localhost:8080/process/text",
        description="The endpoint for the private AI detection server.",
    )
    input: PrivateAIDetectionOptions = Field(
        default_factory=PrivateAIDetectionOptions,
        description="Configuration of the entities to be detected on the user input.",
    )
    output: PrivateAIDetectionOptions = Field(
        default_factory=PrivateAIDetectionOptions,
        description="Configuration of the entities to be detected on the bot output.",
    )
    retrieval: PrivateAIDetectionOptions = Field(
        default_factory=PrivateAIDetectionOptions,
        description="Configuration of the entities to be detected on retrieved relevant chunks.",
    )


class FiddlerGuardrails(BaseModel):
    """Configuration for Fiddler Guardrails."""

    fiddler_endpoint: str = Field(
        default="http://localhost:8080/process/text",
        description="The global endpoint for Fiddler Guardrails requests.",
    )
    safety_threshold: float = Field(
        default=0.1,
        description="Fiddler Guardrails safety detection threshold.",
    )
    faithfulness_threshold: float = Field(
        default=0.05,
        description="Fiddler Guardrails faithfulness detection threshold.",
    )


class MessageTemplate(BaseModel):
    """Template for a message structure."""

    type: str = Field(
        description="The type of message, e.g., 'assistant', 'user', 'system'."
    )
    content: str = Field(description="The content of the message.")


class TaskPrompt(BaseModel):
    """Configuration for prompts that will be used for a specific task."""

    task: str = Field(description="The id of the task associated with this prompt.")
    content: Optional[str] = Field(
        default=None, description="The content of the prompt, if it's a string."
    )
    messages: Optional[List[Union[MessageTemplate, str]]] = Field(
        default=None,
        description="The list of messages included in the prompt. Used for chat models.",
    )
    models: Optional[List[str]] = Field(
        default=None,
        description="If specified, the prompt will be used only for the given LLM engines/models. "
        "The format is a list of strings with the format: <engine> or <engine>/<model>.",
    )
    output_parser: Optional[str] = Field(
        default=None,
        description="The name of the output parser to use for this prompt.",
    )
    max_length: Optional[int] = Field(
        default=16000,
        description="The maximum length of the prompt in number of characters.",
        ge=1,
    )
    mode: Optional[str] = Field(
        default=_default_config["prompting_mode"],
        description="Corresponds to the `prompting_mode` for which this prompt is fetched. Default is 'standard'.",
    )
    stop: Optional[List[str]] = Field(
        default=None,
        description="If specified, will be configure stop tokens for models that support this.",
    )

    max_tokens: Optional[int] = Field(
        default=None,
        description="The maximum number of tokens that can be generated in the chat completion.",
        ge=1,
    )

    @root_validator(pre=True, allow_reuse=True)
    def check_fields(cls, values):
        if not values.get("content") and not values.get("messages"):
            raise ValueError("One of `content` or `messages` must be provided.")

        if values.get("content") and values.get("messages"):
            raise ValueError("Only one of `content` or `messages` must be provided.")

        return values


class LogAdapterConfig(BaseModel):
    name: str = Field(default="FileSystem", description="The name of the adapter.")
    model_config = ConfigDict(extra="allow")


class SpanFormat(str, Enum):
    legacy = "legacy"
    opentelemetry = "opentelemetry"


class TracingConfig(BaseModel):
    enabled: bool = False
    adapters: List[LogAdapterConfig] = Field(
        default_factory=lambda: [LogAdapterConfig()],
        description="The list of tracing adapters to use. If not specified, the default adapters are used.",
    )
    span_format: str = Field(
        default=SpanFormat.opentelemetry,
        description="The span format to use. Options are 'legacy' (simple metrics) or 'opentelemetry' (OpenTelemetry semantic conventions).",
    )
    enable_content_capture: bool = Field(
        default=False,
        description=(
            "Capture prompts and responses (user/assistant/tool message content) in tracing/telemetry events. "
            "Disabled by default for privacy and alignment with OpenTelemetry GenAI semantic conventions. "
            "WARNING: Enabling this may include PII and sensitive data in your telemetry backend."
        ),
    )


class EmbeddingsCacheConfig(BaseModel):
    """Configuration for the caching embeddings."""

    enabled: bool = Field(
        default=False,
        description="Whether caching of the embeddings should be enabled or not.",
    )
    key_generator: str = Field(
        default="sha256",
        description="The method to use for generating the cache keys.",
    )
    store: str = Field(
        default="filesystem",
        description="What type of store to use for the cached embeddings.",
    )
    store_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional configuration options required for the store. "
        "For example, path for `filesystem` or `host`/`port`/`db` for redis.",
    )

    def to_dict(self):
        return self.dict()


class EmbeddingSearchProvider(BaseModel):
    """Configuration of a embedding search provider."""

    name: str = Field(
        default="default",
        description="The name of the embedding search provider. If not specified, default is used.",
    )
    parameters: Dict[str, Any] = Field(default_factory=dict)
    cache: EmbeddingsCacheConfig = Field(default_factory=EmbeddingsCacheConfig)


class KnowledgeBaseConfig(BaseModel):
    folder: str = Field(
        default="kb",
        description="The folder from which the documents should be loaded.",
    )
    embedding_search_provider: EmbeddingSearchProvider = Field(
        default_factory=EmbeddingSearchProvider,
        description="The search provider used to search the knowledge base.",
    )


class CoreConfig(BaseModel):
    """Settings for core internal mechanics."""

    embedding_search_provider: EmbeddingSearchProvider = Field(
        default_factory=EmbeddingSearchProvider,
        description="The search provider used to search the most similar canonical forms/flows.",
    )


class InputRails(BaseModel):
    """Configuration of input rails."""

    parallel: Optional[bool] = Field(
        default=False,
        description="If True, the input rails are executed in parallel.",
    )

    flows: List[str] = Field(
        default_factory=list,
        description="The names of all the flows that implement input rails.",
    )


class OutputRailsStreamingConfig(BaseModel):
    """Configuration for managing streaming output of LLM tokens."""

    enabled: bool = Field(
        default=False, description="Enables streaming mode when True."
    )
    chunk_size: int = Field(
        default=200,
        description="The number of tokens in each processing chunk. This is the size of the token block on which output rails are applied.",
    )
    context_size: int = Field(
        default=50,
        description="The number of tokens carried over from the previous chunk to provide context for continuity in processing.",
    )
    stream_first: bool = Field(
        default=True,
        description="If True, token chunks are streamed immediately before output rails are applied.",
    )
    model_config = ConfigDict(extra="allow")


class OutputRails(BaseModel):
    """Configuration of output rails."""

    parallel: Optional[bool] = Field(
        default=False,
        description="If True, the output rails are executed in parallel.",
    )

    flows: List[str] = Field(
        default_factory=list,
        description="The names of all the flows that implement output rails.",
    )

    streaming: OutputRailsStreamingConfig = Field(
        default_factory=OutputRailsStreamingConfig,
        description="Configuration for streaming output rails.",
    )


class RetrievalRails(BaseModel):
    """Configuration of retrieval rails."""

    flows: List[str] = Field(
        default_factory=list,
        description="The names of all the flows that implement retrieval rails.",
    )


class ActionRails(BaseModel):
    """Configuration of action rails.

    Action rails control various options related to the execution of actions.
    Currently, only

    In the future multiple options will be added, e.g., what input validation should be
    performed per action, output validation, throttling, disabling, etc.
    """

    instant_actions: Optional[List[str]] = Field(
        default=None,
        description="The names of all actions which should finish instantly.",
    )


class ToolOutputRails(BaseModel):
    """Configuration of tool output rails.

    Tool output rails are applied to tool calls before they are executed.
    They can validate tool names, parameters, and context to ensure safe tool usage.
    """

    flows: List[str] = Field(
        default_factory=list,
        description="The names of all the flows that implement tool output rails.",
    )
    parallel: Optional[bool] = Field(
        default=False,
        description="If True, the tool output rails are executed in parallel.",
    )


class ToolInputRails(BaseModel):
    """Configuration of tool input rails.

    Tool input rails are applied to tool results before they are processed.
    They can validate, filter, or transform tool outputs for security and safety.
    """

    flows: List[str] = Field(
        default_factory=list,
        description="The names of all the flows that implement tool input rails.",
    )
    parallel: Optional[bool] = Field(
        default=False,
        description="If True, the tool input rails are executed in parallel.",
    )


class SingleCallConfig(BaseModel):
    """Configuration for the single LLM call option for topical rails."""

    enabled: bool = False
    fallback_to_multiple_calls: bool = Field(
        default=True,
        description="Whether to fall back to multiple calls if a single call is not possible.",
    )


class UserMessagesConfig(BaseModel):
    """Configuration for how the user messages are interpreted."""

    embeddings_only: bool = Field(
        default=False,
        description="Whether to use only embeddings for computing the user canonical form messages.",
    )
    embeddings_only_similarity_threshold: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="The similarity threshold to use when using only embeddings for computing the user canonical form messages.",
    )
    embeddings_only_fallback_intent: Optional[str] = Field(
        default=None,
        description="Defines the fallback intent when the similarity is below the threshold. If set to None, the user intent is computed normally using the LLM. If set to a string value, that string is used as the intent.",
    )


class DialogRails(BaseModel):
    """Configuration of topical rails."""

    single_call: SingleCallConfig = Field(
        default_factory=SingleCallConfig,
        description="Configuration for the single LLM call option.",
    )
    user_messages: UserMessagesConfig = Field(
        default_factory=UserMessagesConfig,
        description="Configuration for how the user messages are interpreted.",
    )


class FactCheckingRailConfig(BaseModel):
    """Configuration data for the fact-checking rail."""

    parameters: Dict[str, Any] = Field(default_factory=dict)
    fallback_to_self_check: bool = Field(
        default=False,
        description="Whether to fall back to self-check if another method fail.",
    )


class JailbreakDetectionConfig(BaseModel):
    """Configuration data for jailbreak detection."""

    server_endpoint: Optional[str] = Field(
        default=None,
        description="The endpoint for the jailbreak detection heuristics/model container.",
    )
    length_per_perplexity_threshold: float = Field(
        default=89.79, description="The length/perplexity threshold."
    )
    prefix_suffix_perplexity_threshold: float = Field(
        default=1845.65, description="The prefix/suffix perplexity threshold."
    )
    nim_base_url: Optional[str] = Field(
        default=None,
        description="Base URL for jailbreak detection model. Example: http://localhost:8000/v1",
    )
    nim_server_endpoint: Optional[str] = Field(
        default="classify",
        description="Classification path uri. Defaults to 'classify' for NemoGuard JailbreakDetect.",
    )
    api_key: Optional[SecretStr] = Field(
        default=None,
        description="Secret String with API key for use in Jailbreak requests. Takes precedence over api_key_env_var",
    )
    api_key_env_var: Optional[str] = Field(
        default=None,
        description="Environment variable containing API key for jailbreak detection model",
    )
    # legacy fields, keep for backward comp with deprecation warnings
    nim_url: Optional[str] = Field(
        default=None,
        deprecated="Use 'nim_base_url' instead. This field will be removed in a future version.",
        description="DEPRECATED: Use nim_base_url instead",
    )
    nim_port: Optional[int] = Field(
        default=None,
        deprecated="Include port in 'nim_base_url' instead. This field will be removed in a future version.",
        description="DEPRECATED: Include port in nim_base_url instead",
    )
    embedding: Optional[str] = Field(
        default=None,
        deprecated="This field is no longer used.",
    )

    @model_validator(mode="after")
    def migrate_deprecated_fields(self) -> "JailbreakDetectionConfig":
        """Migrate deprecated nim_url/nim_port fields to nim_base_url format."""
        if self.nim_url and not self.nim_base_url:
            port = self.nim_port or 8000
            self.nim_base_url = f"http://{self.nim_url}:{port}/v1"
        return self

    def get_api_key(self) -> Optional[str]:
        """Helper to return an API key (if it exists) from a Jailbreak configuration.
          This can come from (in descending order of priority):

        1. The `api_key` field, a Pydantic SecretStr from which we extract the full string.
        2. The `api_key_env_var` field, a string stored in this environment variable.

        If neither is found, None is returned.
        """

        if self.api_key:
            return self.api_key.get_secret_value()

        if self.api_key_env_var:
            nim_auth_token = os.getenv(self.api_key_env_var)
            if nim_auth_token:
                return nim_auth_token

            log.warning(
                "Specified a value for jailbreak config api_key_env var at %s but the environment variable was not set!"
                % self.api_key_env_var
            )

        return None


class AutoAlignOptions(BaseModel):
    """List of guardrails that are activated"""

    guardrails_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="The guardrails configuration that is passed to the AutoAlign endpoint",
    )


class AutoAlignRailConfig(BaseModel):
    """Configuration data for the AutoAlign API"""

    parameters: Dict[str, Any] = Field(default_factory=dict)
    input: AutoAlignOptions = Field(
        default_factory=AutoAlignOptions,
        description="Input configuration for AutoAlign guardrails",
    )
    output: AutoAlignOptions = Field(
        default_factory=AutoAlignOptions,
        description="Output configuration for AutoAlign guardrails",
    )


class PatronusEvaluationSuccessStrategy(str, Enum):
    """
    Strategy for determining whether a Patronus Evaluation API
    request should pass, especially when multiple evaluators
    are called in a single request.
    ALL_PASS requires all evaluators to pass for success.
    ANY_PASS requires only one evaluator to pass for success.
    """

    ALL_PASS = "all_pass"
    ANY_PASS = "any_pass"


class PatronusEvaluateApiParams(BaseModel):
    """Config to parameterize the Patronus Evaluate API call"""

    success_strategy: Optional[PatronusEvaluationSuccessStrategy] = Field(
        default=PatronusEvaluationSuccessStrategy.ALL_PASS,
        description="Strategy to determine whether the Patronus Evaluate API Guardrail passes or not.",
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters to the Patronus Evaluate API",
    )


class PatronusEvaluateConfig(BaseModel):
    """Config for the Patronus Evaluate API call"""

    evaluate_config: PatronusEvaluateApiParams = Field(
        default_factory=PatronusEvaluateApiParams,
        description="Configuration passed to the Patronus Evaluate API",
    )


class PatronusRailConfig(BaseModel):
    """Configuration data for the Patronus Evaluate API"""

    input: Optional[PatronusEvaluateConfig] = Field(
        default_factory=PatronusEvaluateConfig,
        description="Patronus Evaluate API configuration for an Input Guardrail",
    )
    output: Optional[PatronusEvaluateConfig] = Field(
        default_factory=PatronusEvaluateConfig,
        description="Patronus Evaluate API configuration for an Output Guardrail",
    )


class ClavataRailOptions(BaseModel):
    """Configuration data for the Clavata API"""

    policy: str = Field(
        description="The policy alias to use when evaluating inputs or outputs.",
    )

    labels: List[str] = Field(
        default_factory=list,
        description="""A list of labels to match against the policy.
        If no labels are provided, the overall policy result will be returned.
        If labels are provided, only hits on the provided labels will be considered a hit.""",
    )


class ClavataRailConfig(BaseModel):
    """Configuration data for the Clavata API"""

    server_endpoint: str = Field(
        default="https://gateway.app.clavata.ai:8443",
        description="The endpoint for the Clavata API",
    )

    policies: Dict[str, str] = Field(
        default_factory=dict,
        description="A dictionary of policy aliases and their corresponding IDs.",
    )

    label_match_logic: Literal["ANY", "ALL"] = Field(
        default="ANY",
        description="""The logic to use when deciding whether the evaluation matched.
        If ANY, only one of the configured labels needs to be found in the input or output.
        If ALL, all configured labels must be found in the input or output.""",
    )

    input: Optional[ClavataRailOptions] = Field(
        default=None,
        description="Clavata configuration for an Input Guardrail",
    )
    output: Optional[ClavataRailOptions] = Field(
        default=None,
        description="Clavata configuration for an Output Guardrail",
    )


class PangeaRailOptions(BaseModel):
    """Configuration data for the Pangea AI Guard API"""

    recipe: str = Field(
        description="""Recipe key of a configuration of data types and settings defined in the Pangea User Console. It
        specifies the rules that are to be applied to the text, such as defang malicious URLs."""
    )


class PangeaRailConfig(BaseModel):
    """Configuration data for the Pangea AI Guard API"""

    input: Optional[PangeaRailOptions] = Field(
        default=None,
        description="Pangea configuration for an Input Guardrail",
    )
    output: Optional[PangeaRailOptions] = Field(
        default=None,
        description="Pangea configuration for an Output Guardrail",
    )


class GuardrailsAIValidatorConfig(BaseModel):
    """Configuration for a single Guardrails AI validator."""

    name: str = Field(
        description="Unique identifier or import path for the Guardrails AI validator (e.g., 'toxic_language', 'pii', 'regex_match', or 'guardrails/competitor_check')."
    )

    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters to pass to the validator during initialization (e.g., threshold, regex pattern).",
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata to pass to the validator during validation (e.g., valid_topics, context).",
    )


class GuardrailsAIRailConfig(BaseModel):
    """Configuration data for Guardrails AI integration."""

    validators: List[GuardrailsAIValidatorConfig] = Field(
        default_factory=list,
        description="List of Guardrails AI validators to apply. Each validator can have its own parameters and metadata.",
    )

    def get_validator_config(self, name: str) -> Optional[GuardrailsAIValidatorConfig]:
        """Get a specific validator configuration by name."""
        for _validator in self.validators:
            if _validator.name == name:
                return _validator
        return None


class TrendMicroRailConfig(BaseModel):
    """Configuration data for the Trend Micro AI Guard API"""

    v1_url: Optional[str] = Field(
        default="https://api.xdr.trendmicro.com/beta/aiSecurity/guard",
        description="The endpoint for the Trend Micro AI Guard API",
    )

    api_key_env_var: Optional[str] = Field(
        default=None,
        description="Environment variable containing API key for Trend Micro AI Guard",
    )

    def get_api_key(self) -> Optional[str]:
        """Helper to return an API key (if it exists) from a Trend Micro configuration.
        The `api_key_env_var` field, a string stored in this environment variable.

        If the environment variable is not found None is returned.
        """

        if self.api_key_env_var:
            v1_api_key = os.getenv(self.api_key_env_var)
            if v1_api_key:
                return v1_api_key

            log.warning(
                "Specified a value for Trend Micro config api_key_env var at %s but the environment variable was not set!"
                % self.api_key_env_var
            )

        return None


class RailsConfigData(BaseModel):
    """Configuration data for specific rails that are supported out-of-the-box."""

    fact_checking: FactCheckingRailConfig = Field(
        default_factory=FactCheckingRailConfig,
        description="Configuration data for the fact-checking rail.",
    )

    autoalign: AutoAlignRailConfig = Field(
        default_factory=AutoAlignRailConfig,
        description="Configuration data for the AutoAlign guardrails API.",
    )

    patronus: Optional[PatronusRailConfig] = Field(
        default_factory=PatronusRailConfig,
        description="Configuration data for the Patronus Evaluate API.",
    )

    sensitive_data_detection: Optional[SensitiveDataDetection] = Field(
        default_factory=SensitiveDataDetection,
        description="Configuration for detecting sensitive data.",
    )

    jailbreak_detection: Optional[JailbreakDetectionConfig] = Field(
        default_factory=JailbreakDetectionConfig,
        description="Configuration for jailbreak detection.",
    )

    injection_detection: Optional[InjectionDetection] = Field(
        default_factory=InjectionDetection,
        description="Configuration for injection detection.",
    )

    privateai: Optional[PrivateAIDetection] = Field(
        default_factory=PrivateAIDetection,
        description="Configuration for Private AI.",
    )

    fiddler: Optional[FiddlerGuardrails] = Field(
        default_factory=FiddlerGuardrails,
        description="Configuration for Fiddler Guardrails.",
    )

    clavata: Optional[ClavataRailConfig] = Field(
        default_factory=ClavataRailConfig,
        description="Configuration for Clavata.",
    )

    pangea: Optional[PangeaRailConfig] = Field(
        default_factory=PangeaRailConfig,
        description="Configuration for Pangea.",
    )

    guardrails_ai: Optional[GuardrailsAIRailConfig] = Field(
        default_factory=GuardrailsAIRailConfig,
        description="Configuration for Guardrails AI validators.",
    )

    trend_micro: Optional[TrendMicroRailConfig] = Field(
        default_factory=TrendMicroRailConfig,
        description="Configuration for Trend Micro.",
    )


class Rails(BaseModel):
    """Configuration of specific rails."""

    config: RailsConfigData = Field(
        default_factory=RailsConfigData,
        description="Configuration data for specific rails that are supported out-of-the-box.",
    )
    input: InputRails = Field(
        default_factory=InputRails, description="Configuration of the input rails."
    )
    output: OutputRails = Field(
        default_factory=OutputRails, description="Configuration of the output rails."
    )
    retrieval: RetrievalRails = Field(
        default_factory=RetrievalRails,
        description="Configuration of the retrieval rails.",
    )
    dialog: DialogRails = Field(
        default_factory=DialogRails, description="Configuration of the dialog rails."
    )
    actions: ActionRails = Field(
        default_factory=ActionRails, description="Configuration of action rails."
    )
    tool_output: ToolOutputRails = Field(
        default_factory=ToolOutputRails,
        description="Configuration of tool output rails.",
    )
    tool_input: ToolInputRails = Field(
        default_factory=ToolInputRails,
        description="Configuration of tool input rails.",
    )


def merge_two_dicts(dict_1: dict, dict_2: dict, ignore_keys: Set[str]) -> None:
    """Merges the fields of two dictionaries recursively."""
    for key, value in dict_2.items():
        if key not in ignore_keys:
            if key in dict_1:
                if isinstance(dict_1[key], dict) and isinstance(value, dict):
                    merge_two_dicts(dict_1[key], value, set())
                elif dict_1[key] != value:
                    log.warning(
                        "Conflicting fields with same name '%s' in yaml config files detected!",
                        key,
                    )
            else:
                dict_1[key] = value


def _join_config(dest_config: dict, additional_config: dict):
    """Helper to join two configuration."""

    dest_config["user_messages"] = {
        **dest_config.get("user_messages", {}),
        **additional_config.get("user_messages", {}),
    }

    dest_config["bot_messages"] = {
        **dest_config.get("bot_messages", {}),
        **additional_config.get("bot_messages", {}),
    }

    dest_config["instructions"] = dest_config.get(
        "instructions", []
    ) + additional_config.get("instructions", [])

    dest_config["flows"] = dest_config.get("flows", []) + additional_config.get(
        "flows", []
    )

    dest_config["models"] = dest_config.get("models", []) + additional_config.get(
        "models", []
    )

    dest_config["prompts"] = dest_config.get("prompts", []) + additional_config.get(
        "prompts", []
    )

    dest_config["docs"] = dest_config.get("docs", []) + additional_config.get(
        "docs", []
    )

    dest_config["actions_server_url"] = dest_config.get(
        "actions_server_url", None
    ) or additional_config.get("actions_server_url", None)

    dest_config["sensitive_data_detection"] = {
        **dest_config.get("sensitive_data_detection", {}),
        **additional_config.get("sensitive_data_detection", {}),
    }

    dest_config["embedding_search_provider"] = dest_config.get(
        "embedding_search_provider", {}
    ) or additional_config.get("embedding_search_provider", {})

    # We join the arrays and keep only unique elements for import paths.
    dest_config["import_paths"] = dest_config.get("import_paths", [])
    for import_path in additional_config.get("import_paths", []):
        if import_path not in dest_config["import_paths"]:
            dest_config["import_paths"].append(import_path)

    additional_fields = [
        "sample_conversation",
        "lowest_temperature",
        "enable_multi_step_generation",
        "colang_version",
        "event_source_uid",
        "custom_data",
        "prompting_mode",
        "knowledge_base",
        "core",
        "rails",
        "streaming",
        "passthrough",
        "raw_llm_call_action",
        "enable_rails_exceptions",
        "tracing",
    ]

    for field in additional_fields:
        if field in additional_config:
            dest_config[field] = additional_config[field]

    # TODO: Rethink the best way to parse and load yaml config files
    ignore_fields = set(additional_fields).union(
        {
            "user_messages",
            "bot_messages",
            "instructions",
            "flows",
            "models",
            "prompts",
            "docs",
            "actions_server_url",
            "sensitive_data_detection",
            "embedding_search_provider",
            "import_paths",
        }
    )

    # Reads all the other fields and merges them with the custom_data field
    merge_two_dicts(
        dest_config.get("custom_data", {}), additional_config, ignore_fields
    )


def _load_path(
    config_path: str,
) -> Tuple[dict, List[Tuple[str, str]]]:
    """Load a configuration object from the specified path.

    Args:
        config_path: The path from which to load.

    Returns:
        (raw_config, colang_files) The raw config object and the list of colang files.
    """
    raw_config = {}

    # The names of the colang files.
    colang_files = []

    if not os.path.exists(config_path):
        raise ValueError(f"Could not find config path: {config_path}")

    # the first .railsignore file found from cwd down to its subdirectories
    railsignore_path = utils.get_railsignore_path(config_path)
    ignore_patterns = (
        utils.get_railsignore_patterns(railsignore_path) if railsignore_path else set()
    )

    if os.path.isdir(config_path):
        for root, _, files in os.walk(config_path, followlinks=True):
            # Followlinks to traverse symlinks instead of ignoring them.

            for file in files:
                # Verify railsignore to skip loading
                ignored_by_railsignore = utils.is_ignored_by_railsignore(
                    file, ignore_patterns
                )

                if ignored_by_railsignore:
                    continue

                # This is the raw configuration that will be loaded from the file.
                _raw_config = {}

                # Extract the full path for the file and compute relative path
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, config_path)

                # If it's a file in the `kb` folder we need to append it to the docs
                if rel_path.startswith("kb"):
                    _raw_config = {"docs": []}
                    if rel_path.endswith(".md"):
                        with open(full_path, encoding="utf-8") as f:
                            _raw_config["docs"].append(
                                {"format": "md", "content": f.read()}
                            )

                elif file.endswith(".yml") or file.endswith(".yaml"):
                    with open(full_path, "r", encoding="utf-8") as f:
                        _raw_config = yaml.safe_load(f.read())

                elif file.endswith(".co"):
                    colang_files.append((file, full_path))

                _join_config(raw_config, _raw_config)

    # If it's just a .co file, we append it as is to the config.
    elif config_path.endswith(".co"):
        colang_files.append((config_path, config_path))

    return raw_config, colang_files


def _load_imported_paths(raw_config: dict, colang_files: List[Tuple[str, str]]):
    """Load recursively all the imported path in the specified raw_config.

    Args:
        raw_config: The starting raw configuration (i.e., a dict)
        colang_files: The current set of colang files which will be extended as new
            configurations are loaded.
    """
    # We also keep a temporary array of all the paths that have been imported
    if "imported_paths" not in raw_config:
        raw_config["imported_paths"] = {}

    while len(raw_config["imported_paths"]) != len(raw_config["import_paths"]):
        for import_path in raw_config["import_paths"]:
            if import_path in raw_config["imported_paths"]:
                continue

            log.info(f"Loading imported path: {import_path}")

            # If the path does not exist, we try to resolve it using COLANGPATH
            actual_path = None
            if not os.path.exists(import_path):
                for root in colang_path_dirs:
                    if os.path.exists(os.path.join(root, import_path)):
                        actual_path = os.path.join(root, import_path)
                        break

                    # We also check if we can load it as a file.
                    if not import_path.endswith(".co") and os.path.exists(
                        os.path.join(root, import_path + ".co")
                    ):
                        actual_path = os.path.join(root, import_path + ".co")
                        break
            else:
                actual_path = import_path

            if actual_path is None:
                formated_import_path = import_path.replace("/", ".")
                raise ValueError(
                    f"Import path '{formated_import_path}' could not be resolved.",
                )

            _raw_config, _colang_files = _load_path(actual_path)

            # Join them.
            _join_config(raw_config, _raw_config)
            colang_files.extend(_colang_files)

            # And mark the path as imported.
            raw_config["imported_paths"][import_path] = actual_path


def _parse_colang_files_recursively(
    raw_config: dict,
    colang_files: List[Tuple[str, str]],
    parsed_colang_files: List[dict],
):
    """Helper function to parse all the Colang files.

    If there are imports, they will be imported recursively
    """
    colang_version = raw_config.get("colang_version", "1.0")
    _rails_parsed_config = None

    # We start parsing the colang files one by one, and if we have
    # new import paths, we continue to update
    while len(parsed_colang_files) != len(colang_files):
        current_file, current_path = colang_files[len(parsed_colang_files)]

        with open(current_path, "r", encoding="utf-8") as f:
            content = f.read()
            try:
                _parsed_config = parse_colang_file(
                    current_file, content=content, version=colang_version
                )
            except ValueError as e:
                raise ColangParsingError(
                    f"Unsupported colang version {colang_version} for file: {current_path}"
                ) from e
            except Exception as e:
                raise ColangParsingError(
                    f"Error while parsing Colang file: {current_path}\n"
                    + format_colang_parsing_error_message(e, content)
                ) from e

            # We join only the "import_paths" field in the config for now
            _join_config(
                raw_config,
                {"import_paths": _parsed_config.get("import_paths", [])},
            )

            parsed_colang_files.append(_parsed_config)

        # If there are any new imports, we load them
        if raw_config.get("import_paths"):
            _load_imported_paths(raw_config, colang_files)

    if colang_version == "2.x" and _has_input_output_config_rails(raw_config):
        # raise deprecation warning

        rails_flows = _get_rails_flows(raw_config)
        flow_definitions = "\n".join(_generate_rails_flows(rails_flows))

        current_file = "INTRINSIC_FLOW_GENERATION"

        _rails_parsed_config = parse_colang_file(
            current_file, content=flow_definitions, version=colang_version
        )

        _DOCUMENTATION_LINK = "https://docs.nvidia.com/nemo/guardrails/colang-2/getting-started/dialog-rails.html"  # Replace with the actual documentation link

        warnings.warn(
            "Configuring input/output rails in config.yml is deprecated. "
            "Please use the new flow-based configuration instead. "
            f"For more information, please refer to the documentation at {_DOCUMENTATION_LINK}. "
            f"Here is the expected usage:\n{flow_definitions}",
            FutureWarning,
        )

    if _rails_parsed_config:
        parsed_colang_files.append(_rails_parsed_config)
    # To allow overriding of elements from imported paths, we need to merge the
    # parsed data in reverse order.
    for file_parsed_data in reversed(parsed_colang_files):
        _join_config(raw_config, file_parsed_data)


class RailsConfig(BaseModel):
    """Configuration object for the models and the rails.

    TODO: add typed config for user_messages, bot_messages, and flows.
    """

    models: List[Model] = Field(
        description="The list of models used by the rails configuration."
    )

    user_messages: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="The list of user messages that should be used for the rails.",
    )

    bot_messages: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="The list of bot messages that should be used for the rails.",
    )

    # NOTE: the Any below is used to get rid of a warning with pydantic 1.10.x;
    #   The correct typing should be List[Dict, Flow]. To be updated when
    #   support for pydantic 1.10.x is dropped.
    flows: List[Union[Dict, Any]] = Field(
        default_factory=list,
        description="The list of flows that should be used for the rails.",
    )

    instructions: Optional[List[Instruction]] = Field(
        default=[Instruction.parse_obj(obj) for obj in _default_config["instructions"]],
        description="List of instructions in natural language that the LLM should use.",
    )

    docs: Optional[List[Document]] = Field(
        default=None,
        description="List of documents that should be used for question answering.",
    )

    actions_server_url: Optional[str] = Field(
        default=None,
        description="The URL of the actions server that should be used for the rails.",
    )  # consider as conflict

    sample_conversation: Optional[str] = Field(
        default=_default_config["sample_conversation"],
        description="The sample conversation that should be used inside the prompts.",
    )

    prompts: Optional[List[TaskPrompt]] = Field(
        default=None,
        description="The prompts that should be used for the various LLM tasks.",
    )

    prompting_mode: Optional[str] = Field(
        default=_default_config["prompting_mode"],
        description="Allows choosing between different prompting strategies.",
    )

    config_path: Optional[str] = Field(
        default=None, description="The path from which the configuration was loaded."
    )

    import_paths: Optional[List[str]] = Field(
        default_factory=list,
        description="A list of additional paths from which configuration elements (colang flows, .yml files, actions)"
        " should be loaded.",
    )
    imported_paths: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        description="The mapping between the imported paths and the actual full path to which they were resolved.",
    )

    # Some tasks need to be as deterministic as possible. The lowest possible temperature
    # will be used for those tasks. Models like dolly don't allow for a temperature of 0.0,
    # for example, in which case a custom one can be set.
    lowest_temperature: Optional[float] = Field(
        default=0.001,
        description="The lowest temperature that should be used for the LLM.",
    )

    # This should only be enabled for highly capable LLMs i.e. gpt-3.5-turbo-instruct or similar.
    enable_multi_step_generation: Optional[bool] = Field(
        default=False,
        description="Whether to enable multi-step generation for the LLM.",
    )

    colang_version: str = Field(default="1.0", description="The Colang version to use.")

    custom_data: Dict = Field(
        default_factory=dict,
        description="Any custom configuration data that might be needed.",
    )

    knowledge_base: KnowledgeBaseConfig = Field(
        default_factory=KnowledgeBaseConfig,
        description="Configuration for the built-in knowledge base support.",
    )

    core: CoreConfig = Field(
        default_factory=CoreConfig,
        description="Configuration for core internal mechanics.",
    )

    rails: Rails = Field(
        default_factory=Rails,
        description="Configuration for the various rails (input, output, etc.).",
    )

    streaming: bool = Field(
        default=False,
        description="Whether this configuration should use streaming mode or not.",
    )

    enable_rails_exceptions: bool = Field(
        default=False,
        description="If set, the pre-defined guardrails raise exceptions instead of returning pre-defined messages.",
    )

    passthrough: Optional[bool] = Field(
        default=None,
        description="Weather the original prompt should pass through the guardrails configuration as is. "
        "This means it will not be altered in any way. ",
    )

    event_source_uid: str = Field(
        default="NeMoGuardrails-Colang-2.x",
        description="The source ID of events sent by the Colang Runtime. Useful to identify the component that has sent an event.",
    )

    tracing: TracingConfig = Field(
        default_factory=TracingConfig,
        description="Configuration for tracing.",
    )

    @root_validator(pre=True, allow_reuse=True)
    def check_prompt_exist_for_self_check_rails(cls, values):
        rails = values.get("rails", {})
        prompts = values.get("prompts", []) or []

        enabled_input_rails = rails.get("input", {}).get("flows", [])
        enabled_output_rails = rails.get("output", {}).get("flows", [])
        provided_task_prompts = [
            prompt.task if hasattr(prompt, "task") else prompt.get("task")
            for prompt in prompts
        ]

        # Input moderation prompt verification
        if (
            "self check input" in enabled_input_rails
            and "self_check_input" not in provided_task_prompts
        ):
            raise ValueError("You must provide a `self_check_input` prompt template.")
        if (
            "llama guard check input" in enabled_input_rails
            and "llama_guard_check_input" not in provided_task_prompts
        ):
            raise ValueError(
                "You must provide a `llama_guard_check_input` prompt template."
            )

        # Output moderation prompt verification
        if (
            "self check output" in enabled_output_rails
            and "self_check_output" not in provided_task_prompts
        ):
            raise ValueError("You must provide a `self_check_output` prompt template.")
        if (
            "llama guard check output" in enabled_output_rails
            and "llama_guard_check_output" not in provided_task_prompts
        ):
            raise ValueError(
                "You must provide a `llama_guard_check_output` prompt template."
            )
        if (
            "patronus lynx check output hallucination" in enabled_output_rails
            and "patronus_lynx_check_output_hallucination" not in provided_task_prompts
        ):
            raise ValueError(
                "You must provide a `patronus_lynx_check_output_hallucination` prompt template."
            )

        if (
            "self check facts" in enabled_output_rails
            and "self_check_facts" not in provided_task_prompts
        ):
            raise ValueError("You must provide a `self_check_facts` prompt template.")

        return values

    @root_validator(pre=True, allow_reuse=True)
    def check_output_parser_exists(cls, values):
        tasks_requiring_output_parser = [
            "self_check_input",
            "self_check_facts",
            "self_check_output",
            # "content_safety_check input $model",
            # "content_safety_check output $model",
        ]
        prompts = values.get("prompts") or []
        for prompt in prompts:
            task = prompt.task if hasattr(prompt, "task") else prompt.get("task")
            output_parser = (
                prompt.output_parser
                if hasattr(prompt, "output_parser")
                else prompt.get("output_parser")
            )

            if (
                any(
                    task.startswith(task_prefix)
                    for task_prefix in tasks_requiring_output_parser
                )
                and not output_parser
            ):
                log.info(
                    f"Deprecation Warning: Output parser is not registered for the task. "
                    f"The correct way is to register the 'output_parser' in the prompts.yml for '{task}' task. "
                    "It uses 'is_content safe' as the default output parser."
                    "This behavior will be deprecated in future versions."
                )
        return values

    @root_validator(pre=True, allow_reuse=True)
    def fill_in_default_values_for_v2_x(cls, values):
        instructions = values.get("instructions", {})
        sample_conversation = values.get("sample_conversation")
        colang_version = values.get("colang_version", "1.0")

        if colang_version == "2.x":
            if not instructions:
                values["instructions"] = _default_config_v2["instructions"]

            if not sample_conversation:
                values["sample_conversation"] = _default_config_v2[
                    "sample_conversation"
                ]

        return values

    @validator("models")
    def validate_models_api_key_env_var(cls, models):
        """Model API Key Env var must be set to make LLM calls"""
        api_keys = [m.api_key_env_var for m in models]
        for api_key in api_keys:
            if api_key and not os.environ.get(api_key):
                raise ValueError(
                    f"Model API Key environment variable '{api_key}' not set."
                )
        return models

    raw_llm_call_action: Optional[str] = Field(
        default="raw llm call",
        description="The name of the action that would execute the original raw LLM call. ",
    )

    @classmethod
    def from_path(
        cls,
        config_path: str,
    ):
        """Loads a configuration from a given path.

        Supports loading a from a single file, or from a directory.
        """
        # If the config path is a file, we load the YAML content.
        # Otherwise, if it's a folder, we iterate through all files.
        if os.path.isfile(config_path) and config_path.endswith((".yaml", ".yml")):
            with open(config_path) as f:
                raw_config = yaml.safe_load(f.read())

        elif os.path.isdir(config_path):
            raw_config, colang_files = _load_path(config_path)

            # If we have import paths, we also need to load them.
            if raw_config.get("import_paths"):
                _load_imported_paths(raw_config, colang_files)

            # Parse the colang files after we know the colang version
            _parse_colang_files_recursively(
                raw_config, colang_files, parsed_colang_files=[]
            )

        else:
            raise ValueError(f"Invalid config path {config_path}.")

        # If there are no instructions, we use the default ones.
        if len(raw_config.get("instructions", [])) == 0:
            raw_config["instructions"] = _default_config["instructions"]

        raw_config["config_path"] = config_path

        return cls.parse_object(raw_config)

    @classmethod
    def from_content(
        cls,
        colang_content: Optional[str] = None,
        yaml_content: Optional[str] = None,
        config: Optional[dict] = None,
    ):
        """Loads a configuration from the provided colang/YAML content/config dict."""
        raw_config = {}

        if config:
            _join_config(raw_config, config)

        if yaml_content:
            _join_config(raw_config, yaml.safe_load(yaml_content))

        # Parse the colang files after we know the colang version
        colang_version = raw_config.get("colang_version", "1.0")

        # We start parsing the colang files one by one, and if we have
        # new import paths, we continue to update
        colang_files = []
        parsed_colang_files = []

        # First, we parse the starting content.
        if colang_content:
            colang_files.append(("main.co", "main.co"))

            _parsed_config = parse_colang_file(
                "main.co",
                content=colang_content,
                version=colang_version,
            )

            # We join only the "import_paths" field in the config for now
            _join_config(
                raw_config,
                {"import_paths": _parsed_config.get("import_paths", [])},
            )

            parsed_colang_files.append(_parsed_config)

        # Load any new colang files potentially coming from imports
        if raw_config.get("import_paths"):
            _load_imported_paths(raw_config, colang_files)

        # Next, we parse any additional files recursively
        _parse_colang_files_recursively(raw_config, colang_files, parsed_colang_files)

        # If there are no instructions, we use the default ones.
        if len(raw_config.get("instructions", [])) == 0:
            raw_config["instructions"] = _default_config["instructions"]

        return cls.parse_object(raw_config)

    @classmethod
    def parse_object(cls, obj):
        """Parses a configuration object from a given dictionary."""
        # If we have flows, we need to process them further from CoYML to CIL, but only for
        # version 1.0.

        if obj.get("colang_version", "1.0") == "1.0":
            for flow_data in obj.get("flows", []):
                # If the first element in the flow does not have a "_type", we need to convert
                if flow_data.get("elements") and not flow_data["elements"][0].get(
                    "_type"
                ):
                    flow_data["elements"] = parse_flow_elements(flow_data["elements"])

        return cls.parse_obj(obj)

    @property
    def streaming_supported(self):
        """Whether the current config supports streaming or not."""

        if len(self.rails.output.flows) > 0:
            # if we have output rails streaming enabled
            # we keep it in case it was needed when we have
            # support per rails
            if self.rails.output.streaming and self.rails.output.streaming.enabled:
                return True
            return False

        return True

    def __add__(self, other):
        """Adds two RailsConfig objects."""
        return _join_rails_configs(self, other)


def _join_dict(dict1, dict2):
    """
    Joins two dictionaries recursively.
    - If values are dictionaries, it applies _join_dict recursively.
    - If values are lists, it concatenates them, ensuring unique elements.
    - For other types, values from dict2 overwrite dict1.
    """
    result = dict(dict1)  # Create a copy of dict1 to avoid modifying the original

    for key, value in dict2.items():
        # If key is in both dictionaries and both values are dictionaries, apply _join_dict recursively
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
            result[key] = _join_dict(dict1[key], value)
        # If key is in both dictionaries and both values are lists, concatenate unique elements
        elif key in dict1 and isinstance(dict1[key], list) and isinstance(value, list):
            # Since we want values from dict2 to take precedence, we concatenate dict2 first
            result[key] = _unique_list_concat(value, dict1[key])
        # Otherwise, simply overwrite the value from dict2
        else:
            result[key] = value

    return result


def _unique_list_concat(list1, list2):
    """
    Concatenates two lists ensuring all elements are unique.
    Handles unhashable types like dictionaries.
    """
    result = list(list1)
    for item in list2:
        if item not in result:
            result.append(item)
    return result


def _join_rails_configs(
    base_rails_config: RailsConfig, updated_rails_config: RailsConfig
):
    """Helper to join two rails configuration."""

    config_old_types = {}
    for model_old in base_rails_config.models:
        config_old_types[model_old.type] = model_old

    for model_new in updated_rails_config.models:
        if model_new.type in config_old_types:
            if model_new.engine != config_old_types[model_new.type].engine:
                raise ValueError(
                    "Both config files should have the same engine for the same model type"
                )
            if model_new.model != config_old_types[model_new.type].model:
                raise ValueError(
                    "Both config files should have the same model for the same model type"
                )

    if base_rails_config.actions_server_url != updated_rails_config.actions_server_url:
        raise ValueError("Both config files should have the same actions_server_url")

    combined_rails_config_dict = _join_dict(
        base_rails_config.dict(), updated_rails_config.dict()
    )
    # filter out empty strings to avoid leading/trailing commas
    config_paths = [
        base_rails_config.dict()["config_path"] or "",
        updated_rails_config.dict()["config_path"] or "",
    ]
    combined_rails_config_dict["config_path"] = ",".join(filter(None, config_paths))
    combined_rails_config = RailsConfig(**combined_rails_config_dict)
    return combined_rails_config


def _has_input_output_config_rails(raw_config):
    """Checks if the raw configuration has input/output rails configured."""

    has_input_rails = (
        len(raw_config.get("rails", {}).get("input", {}).get("flows", [])) > 0
    )
    has_output_rails = (
        len(raw_config.get("rails", {}).get("output", {}).get("flows", [])) > 0
    )
    return has_input_rails or has_output_rails


def _get_rails_flows(raw_config):
    """Extracts the list of flows from the raw_config dictionary.

    Args:
        raw_config (dict): The raw configuration dictionary.

    Returns:
        list: The list of flows.
    """
    from collections import defaultdict

    flows = defaultdict(list)

    for key in raw_config["rails"]:
        if "flows" in raw_config["rails"][key]:
            flows[key].extend(raw_config["rails"][key]["flows"])
    return flows


def _generate_rails_flows(flows):
    """Generates flow definitions from the list of flows.
    Args:
        flows (dict): The dictionary of flows.
    Returns:
        str: The flow definitions.
    """
    _MAPPING = {
        "input": "flow input rails $input_text",
        "output": "flow output rails $output_text",
    }

    _GUARDRAILS_IMPORT = "import guardrails"
    _LIBRARY_IMPORT = "import nemoguardrails.library"

    flow_definitions = []
    _INDENT = "    "  # 4 spaces for indentation
    _NEWLINE = "\n"

    for key, value in flows.items():
        flow_definitions.append(_MAPPING[key] + _NEWLINE)
        for v in value:
            flow_definitions.append(_INDENT + v + _NEWLINE)
        flow_definitions.append(_NEWLINE)  # Add an empty line after each flow

    if flow_definitions:
        flow_definitions.insert(0, _GUARDRAILS_IMPORT + _NEWLINE)
        flow_definitions.insert(1, _LIBRARY_IMPORT + _NEWLINE * 2)

    return flow_definitions
