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

"""
Module for providing a context manager to temporarily adjust parameters of a language model.

Also allows registration of custom parameter managers for different language model types.

.. deprecated:: 0.17.0
    This module is deprecated and will be removed in version 0.19.0.
    Instead of using the context manager approach, pass parameters directly to `llm_call()`
    using the `llm_params` argument:

    Old way (deprecated):
        from nemoguardrails.llm.params import llm_params
        with llm_params(llm, temperature=0.7):
            result = await llm_call(llm, prompt)

    New way (recommended):
        result = await llm_call(llm, prompt, llm_params={"temperature": 0.7})

    See: https://github.com/NVIDIA/NeMo-Guardrails/issues/1387
"""

import logging
import warnings
from typing import Any, Dict, Type

from langchain.base_language import BaseLanguageModel

log = logging.getLogger(__name__)

_DEPRECATION_MESSAGE = (
    "The nemoguardrails.llm.params module is deprecated and will be removed in version 0.19.0. "
    "Instead of using llm_params context manager, pass parameters directly to llm_call() "
    "using the llm_params argument. "
    "See: https://github.com/NVIDIA/NeMo-Guardrails/issues/1387"
)


class LLMParams:
    """Context manager to temporarily modify the parameters of a language model.

    .. deprecated:: 0.17.0
        Use llm_call() with llm_params argument instead.
    """

    def __init__(self, llm: BaseLanguageModel, **kwargs):
        warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)
        self.llm = llm
        self.altered_params = kwargs
        self.original_params: dict[str, Any] = {}

    def __enter__(self):
        # Here we can access and modify the global language model parameters.
        for param, value in self.altered_params.items():
            if hasattr(self.llm, param):
                self.original_params[param] = getattr(self.llm, param)
                setattr(self.llm, param, value)

            elif hasattr(self.llm, "model_kwargs"):
                model_kwargs = getattr(self.llm, "model_kwargs", {})
                if param not in model_kwargs:
                    log.warning(
                        "Parameter %s does not exist for %s. Passing to model_kwargs",
                        param,
                        self.llm.__class__.__name__,
                    )

                    self.original_params[param] = None
                else:
                    self.original_params[param] = model_kwargs[param]

                model_kwargs[param] = value
                setattr(self.llm, "model_kwargs", model_kwargs)

            else:
                log.warning(
                    "Parameter %s does not exist for %s",
                    param,
                    self.llm.__class__.__name__,
                )

    def __exit__(self, exc_type, value, traceback):
        # Restore original parameters when exiting the context
        for param, value in self.original_params.items():
            if hasattr(self.llm, param):
                setattr(self.llm, param, value)
            elif hasattr(self.llm, "model_kwargs"):
                model_kwargs = getattr(self.llm, "model_kwargs", {})
                if param in model_kwargs:
                    model_kwargs[param] = value
                    setattr(self.llm, "model_kwargs", model_kwargs)


# The list of registered param managers. This will allow us to override the param manager
# for a new LLM.
_param_managers: Dict[Type[BaseLanguageModel], Type[LLMParams]] = {}


def register_param_manager(llm_type: Type[BaseLanguageModel], manager: Type[LLMParams]):
    """Register a parameter manager.

    .. deprecated:: 0.17.0
        This function is deprecated and will be removed in version 0.19.0.
    """
    warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)
    _param_managers[llm_type] = manager


def llm_params(llm: BaseLanguageModel, **kwargs):
    """Returns a parameter manager for the given language model.

    .. deprecated:: 0.17.0
        Use llm_call() with llm_params argument instead.
    """
    warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)
    _llm_params = _param_managers.get(llm.__class__, LLMParams)

    return _llm_params(llm, **kwargs)
