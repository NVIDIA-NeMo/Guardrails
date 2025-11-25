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

"""Module for initializing LangChain models with proper error handling."""

import logging
import warnings
from importlib.metadata import version
from typing import Any, Callable, Dict, Literal, Optional, Union

from langchain.chat_models import init_chat_model

# from langchain_core._api.beta_decorator import LangChainBetaWarning
# from langchain_core._api.deprecation import LangChainDeprecationWarning
from langchain_core.language_models import BaseChatModel, BaseLLM

from nemoguardrails.llm.providers.providers import (
    _get_chat_completion_provider,
    _get_text_completion_provider,
    _parse_version,
)

log = logging.getLogger(__name__)


# Suppress specific LangChain warnings
# warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
# warnings.filterwarnings("ignore", category=LangChainBetaWarning)
warnings.filterwarnings("ignore", module="langchain_nvidia_ai_endpoints._common")


class ModelInitializationError(Exception):
    """Raised when model initialization fails."""

    pass


ModelInitMethod = Callable[[str, str, Dict[str, Any]], Optional[Union[BaseChatModel, BaseLLM]]]


class ModelInitializer:
    """A method for initializing a model with its supported modes."""

    def __init__(
        self,
        init_method: ModelInitMethod,
        supported_modes: list[Literal["chat", "text"]],
    ):
        self.init_method = init_method
        self.supported_modes = supported_modes

    def supports_mode(self, mode: Literal["chat", "text"]) -> bool:
        """Check if this initializer supports the given mode."""
        return mode in self.supported_modes

    def execute(
        self, model_name: str, provider_name: str, kwargs: Dict[str, Any]
    ) -> Optional[Union[BaseChatModel, BaseLLM]]:
        """Execute this initializer to initialize a model."""
        return self.init_method(model_name, provider_name, kwargs)

    def __str__(self) -> str:
        return f"{self.init_method.__name__}(modes={self.supported_modes})"


def try_initialization_method(
    initializer: ModelInitializer,
    model_name: str,
    provider_name: str,
    mode: Literal["chat", "text"],
    kwargs: Dict[str, Any],
):
    """Wrap an initialization method execution with a try/except to capture errors.

    1. Wraps the call to `try_initialization_method` in a try/except block
    2. Catches any exceptions that might be thrown
    3. Logs them and continues with the next initializer
    4. Only fails at the end if all initializers have been tried
    """
    # skip initializers that don't support the requested mode
    if not initializer.supports_mode(mode):
        log.debug(
            f"Skipping initializer: {initializer.init_method.__name__} for model: {model_name} "
            f"and provider: {provider_name} as it doesn't support mode: {mode}"
        )
        return None

    try:
        log.debug(
            f"Trying initializer: {initializer.init_method.__name__} for model: {model_name} and provider: {provider_name}"
        )
        result = initializer.execute(
            model_name=model_name,
            provider_name=provider_name,
            kwargs=kwargs,
        )
        log.debug(f"Initializer {initializer.init_method.__name__} returned: {result}")
        if result is not None:
            return result
    except ValueError as e:
        raise ValueError(
            f"ValueError encountered in initializer {initializer} "
            f"for model: {model_name} and provider: {provider_name}: {e}"
        )
    return None


def init_langchain_model(
    model_name: str,
    provider_name: str,
    mode: Literal["chat", "text"],
    kwargs: Dict[str, Any],
) -> Union[BaseChatModel, BaseLLM]:
    """Initialize a LangChain model using a series of initialization methods.

    This function tries multiple initialization methods in sequence until one succeeds.
    Each method is attempted only if it supports the requested mode.
    """

    if mode not in ["chat", "text"]:
        raise ValueError(f"Unsupported mode: {mode}")
    if not model_name:
        raise ModelInitializationError(f"Model name is required for provider {provider_name}")

    # Define initialization methods in order of preference
    initializers: list[ModelInitializer] = [
        # Try special case handlers first (handles both chat and text)
        ModelInitializer(_handle_model_special_cases, ["chat", "text"]),
        # For chat mode, first try the standard chat completion API
        ModelInitializer(_init_chat_completion_model, ["chat"]),
        # For chat mode, fall back to community chat models
        ModelInitializer(_init_community_chat_models, ["chat"]),
        # FIXME: is text and chat a good idea?
        # For text mode, use text completion, we are using both text and chat as the last resort
        ModelInitializer(_init_text_completion_model, ["text", "chat"]),
    ]

    # Track the last exception for better error reporting
    last_exception = None
    # but also remember the first import‐error we see
    first_import_error: Optional[ImportError] = None
    # Try each initializer in sequence
    for initializer in initializers:
        try:
            result = try_initialization_method(
                initializer=initializer,
                model_name=model_name,
                provider_name=provider_name,
                mode=mode,
                kwargs=kwargs,
            )
            if result is not None:
                return result
        except ImportError as e:
            # remember only the first import‐error we encounter
            if first_import_error is None:
                first_import_error = e
            last_exception = e
            log.debug(f"Initialization import‐failure in {initializer}: {e}")
        except Exception as e:
            last_exception = e
            log.debug(f"Initialization failed with {initializer}: {e}")
    # build the final message, preferring that first ImportError if we saw one
    base = f"Failed to initialize model {model_name!r} with provider {provider_name!r} in {mode!r} mode"

    # if we ever hit an ImportError, surface its message:
    if first_import_error is not None:
        base += f": {first_import_error}"
        # chain from that importer
        raise ModelInitializationError(base) from first_import_error

    # otherwise fall back to the last exception we saw
    if last_exception is not None:
        base += f": {last_exception}"
        raise ModelInitializationError(base) from last_exception

    # (should never happen—no initializer claimed success, no exception thrown)
    raise ModelInitializationError(base)


def _init_chat_completion_model(model_name: str, provider_name: str, kwargs: Dict[str, Any]) -> BaseChatModel:  # noqa #type: ignore
    """Initialize a chat completion model.

    Args:
        model_name: Name of the model to initialize
        provider_name: Name of the provider to use
        kwargs: Additional arguments to pass to the model initialization

    Returns:
        An initialized chat completion model

    Raises:
        ValueError: If the model cannot be initialized as a chat model
    """

    # just to document the expected behavior
    # we don't support pre-0.2.7 versions of langchain-core it is in
    # line with our pyproject.toml
    package_version = version("langchain-core")

    if _parse_version(package_version) < (0, 2, 7):
        raise RuntimeError(
            "this feature is supported from v0.2.7 of langchain-core."
            " Please upgrade it with `pip install langchain-core --upgrade`."
        )
    try:
        return init_chat_model(
            model=model_name,
            model_provider=provider_name,
            **kwargs,
        )
    except ValueError:
        raise


def _init_text_completion_model(model_name: str, provider_name: str, kwargs: Dict[str, Any]) -> BaseLLM:
    """Initialize a text completion model.

    Args:
        model_name: Name of the model to initialize
        provider_name: Name of the provider to use
        kwargs: Additional arguments to pass to the model initialization

    Returns:
        An initialized text completion model

    Raises:
        RuntimeError: If the provider is not found
    """
    provider_cls = _get_text_completion_provider(provider_name)
    if provider_cls is None:
        raise ValueError()
    kwargs = _update_model_kwargs(provider_cls, model_name, kwargs)
    # remove stream_usage parameter as it's not supported by text completion APIs
    # (e.g., OpenAI's AsyncCompletions.create() doesn't accept this parameter)
    kwargs.pop("stream_usage", None)
    return provider_cls(**kwargs)


def _init_community_chat_models(model_name: str, provider_name: str, kwargs: Dict[str, Any]) -> BaseChatModel:
    """Initialize community chat models.

    Args:
        provider_name: Name of the provider to use
        model_name: Name of the model to initialize
        kwargs: Additional arguments to pass to the model initialization

    Returns:
        An initialized chat model

    Raises:
        ImportError: If langchain_community is not installed
        ModelInitializationError: If model initialization fails
    """
    provider_cls = _get_chat_completion_provider(provider_name)
    if provider_cls is None:
        raise ValueError()
    kwargs = _update_model_kwargs(provider_cls, model_name, kwargs)
    return provider_cls(**kwargs)


def _init_gpt35_turbo_instruct(model_name: str, provider_name: str, kwargs: Dict[str, Any]) -> BaseLLM:
    """Initialize GPT-3.5 Turbo Instruct model.

    Currently init_chat_model from langchain infers this as a chat model.
    This is a bug in langchain, and we need to handle it here.

    This model requires text completion initialization.

    Args:
        model_name: Name of the model to initialize
        provider_name: Name of the provider to use
        kwargs: Additional arguments to pass to the model initialization

    Returns:
        An initialized text completion model

    Raises:
        ModelInitializationError: If model initialization fails
    """
    try:
        return _init_text_completion_model(
            model_name=model_name,
            provider_name=provider_name,
            kwargs=kwargs,
        )
    except Exception as e:
        raise ModelInitializationError(f"Failed to initialize text completion model {model_name}: {str(e)}")


def _init_nvidia_model(model_name: str, provider_name: str, kwargs: Dict[str, Any]) -> BaseChatModel:
    """Initialize NVIDIA AI Endpoints model.

    Args:
        model_name: Name of the model to initialize
        provider_name: Name of the provider to use
        **kwargs: Additional arguments to pass to the model initialization

    Returns:
        An initialized chat model

    Raises:
        ImportError: If langchain_nvidia_ai_endpoints is not installed
        ModelInitializationError: If model initialization fails
    """
    try:
        from nemoguardrails.llm.providers._langchain_nvidia_ai_endpoints_patch import (
            ChatNVIDIA,
        )

        package_version = version("langchain_nvidia_ai_endpoints")

        if _parse_version(package_version) < (0, 2, 0):
            raise ValueError(
                "langchain_nvidia_ai_endpoints version must be 0.2.0 or above."
                " Please upgrade it with `pip install langchain-nvidia-ai-endpoints --upgrade`."
            )

        return ChatNVIDIA(model=model_name, **kwargs)
    except ImportError as e:
        raise ImportError(
            "Could not import langchain_nvidia_ai_endpoints, please install it with "
            "`pip install langchain-nvidia-ai-endpoints`."
        )


# special model handlers
_SPECIAL_MODEL_INITIALIZERS = {
    "gpt-3.5-turbo-instruct": _init_gpt35_turbo_instruct,
}

# provider-specific handlers
_PROVIDER_INITIALIZERS = {
    "nvidia_ai_endpoints": _init_nvidia_model,
    "nim": _init_nvidia_model,
}


def _handle_model_special_cases(
    model_name: str, provider_name: str, kwargs: Dict[str, Any]
) -> Optional[Union[BaseChatModel, BaseLLM]]:
    """Handle model initialization for special cases that need custom logic.

    This function handles edge cases where standard initialization methods
    don't work properly. It looks up initializers in the registry and dispatches
    to the appropriate initialization function.

    Args:
        provider_name: Name of the provider to use
        model_name: Name of the model to initialize
        kwargs: Additional arguments to pass to the model initialization

    Returns:
        An initialized model for special cases, or None if no special initializer exists
    """
    initializer = None

    for pattern, model_initializer in _SPECIAL_MODEL_INITIALIZERS.items():
        if pattern in model_name:
            initializer = model_initializer
            break

    if initializer is None and provider_name in _PROVIDER_INITIALIZERS:
        initializer = _PROVIDER_INITIALIZERS[provider_name]

    if initializer is None:
        return None

    result = initializer(model_name, provider_name, kwargs)
    if not isinstance(result, (BaseChatModel, BaseLLM)):
        raise TypeError("Initializer returned an invalid type")
    return result


def _update_model_kwargs(provider_cls: type, model_name: str, kwargs: dict) -> Dict:
    """Update kwargs with the model name based on the provider's expected fields.

    If provider_cls.model_fields contains 'model' or 'model_name',
    sets the corresponding key in kwargs to model_name.
    """
    for key in ("model", "model_name"):
        if key in getattr(provider_cls, "model_fields", {}):
            kwargs[key] = model_name
    return kwargs
