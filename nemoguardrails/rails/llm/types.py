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

"""Structural typing contracts for the extracted ``rails.llm`` helper modules.

``LLMRails`` is decomposed into focused helper modules (``startup``, ``runtime``,
``conversation``, ``generation``, ``streaming``, ``checks``). Each helper depends
on only a narrow slice of the ``LLMRails`` instance.

This module expresses those slices along two axes:

* **Capabilities** (``Has*`` / ``Supports*``): one single-role Protocol per
  member-or-method an extracted helper touches, declared exactly once.
* **Contexts**: one Protocol per helper role, assembled by composing the
  capabilities it needs (Protocol multiple inheritance), covering exactly what
  that helper reads/writes/calls and nothing more.

These are deliberately **structural**, not pinned to the concrete ``RailsConfig``
/ ``Runtime`` / ``ExplainInfo`` classes: a helper unit treats ``config`` and
``runtime`` as opaque collaborators, so a lightweight stand-in (a
``SimpleNamespace`` config, a minimal fake runtime) that is valid for the unit
also satisfies its protocol. The concrete types are enforced at the production
call site (``LLMRails``), where ``config`` and ``runtime`` are already statically
typed.

Variance is deliberate. A member a helper *reassigns* (``rails.x = ...``) is a
plain attribute annotation (``x: Any``); ``Any`` keeps the invariant-attribute
match permissive. A member a helper only *reads* is a read-only ``@property`` so
it is covariant: a concrete attribute, a ``property``, or a fake all satisfy it,
which a mutable (invariant) attribute annotation would reject.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Union

from nemoguardrails.rails.llm.options import GenerationOptions
from nemoguardrails.streaming import StreamingHandler

__all__ = [
    "ColangTurnSurface",
    "ConversationEventSurface",
    "GenerationActionsSurface",
    "GenerationStreamSurface",
    "KnowledgeBaseSurface",
    "LLMActionCacheSurface",
    "LLMActionModelsSurface",
    "RailsCheckSurface",
    "StandardGenerationSurface",
    "StreamingOutputSurface",
]


# capabilities: one role each, declared exactly once.


class HasConfig(Protocol):
    """Reads the rails configuration (treated as an opaque collaborator)."""

    @property
    def config(self) -> Any: ...


class HasRuntime(Protocol):
    """Reaches the Colang runtime (action registration, dispatch, turn execution)."""

    @property
    def runtime(self) -> Any: ...


class HasLLM(Protocol):
    """Holds the main LLM; reassigned by the startup model wiring."""

    llm: Any


class HasGenerationActions(Protocol):
    """Holds the LLM generation actions object; reassigned during startup."""

    _llm_generation_actions: Any


class HasKnowledgeBase(Protocol):
    """Holds the knowledge base; assigned by knowledge-base init."""

    _kb: Any


class SupportsEmbeddingSearchProvider(Protocol):
    """Exposes the embedding-search provider factory seam (patchable/overridable)."""

    def _get_embeddings_search_provider_instance(self, esp_config: Any = None) -> Any: ...


class HasExplainInfo(Protocol):
    """Holds the per-instance ExplainInfo; assigned by the generation workflow."""

    _explain_info: Any


class SupportsEnsureExplainInfo(Protocol):
    """Lazily materializes the request-scoped ExplainInfo; result is stored/forwarded."""

    def _ensure_explain_info(self) -> Any: ...


class HasLogAdapters(Protocol):
    """Reads the tracing log adapters configured at startup."""

    @property
    def _log_adapters(self) -> Any: ...


class HasEventsHistoryCache(Protocol):
    """Reads the events-history cache (entries are mutated in place, not rebound)."""

    @property
    def events_history_cache(self) -> Dict[str, List[dict]]: ...


class HasVerbose(Protocol):
    """Reads the verbose flag (a ``property`` on the concrete class)."""

    @property
    def verbose(self) -> bool: ...


class SupportsGenerateAsync(Protocol):
    """Drives a full async generation pass; mirrors ``LLMRails.generate_async``."""

    async def generate_async(
        self,
        *,
        prompt: Optional[str] = None,
        messages: Optional[List[dict]] = None,
        options: Optional[Union[dict, GenerationOptions]] = None,
        state: Optional[Any] = None,
        streaming_handler: Optional[StreamingHandler] = None,
    ) -> Any: ...


# contexts: one per helper role, composed from the capabilities it uses.


class LLMActionModelsSurface(HasConfig, HasRuntime, HasLLM, HasGenerationActions, Protocol):
    """Surface used by ``startup.llm_action_models`` (load + sync model bindings)."""


class LLMActionCacheSurface(HasConfig, HasRuntime, Protocol):
    """Surface used by ``startup.llm_action_caches``."""


class GenerationActionsSurface(
    HasConfig, HasRuntime, HasLLM, HasGenerationActions, SupportsEmbeddingSearchProvider, Protocol
):
    """Surface used by ``startup.generation_actions`` to register LLM actions."""


class KnowledgeBaseSurface(HasConfig, HasRuntime, SupportsEmbeddingSearchProvider, HasKnowledgeBase, Protocol):
    """Surface used by ``startup.knowledge_base`` to build and register the KB."""


class ConversationEventSurface(HasConfig, HasEventsHistoryCache, Protocol):
    """Surface used by ``conversation.conversation_events``."""


class ColangTurnSurface(HasConfig, HasRuntime, HasVerbose, Protocol):
    """Surface used by ``colang_turns.colang_turns`` to run a Colang turn."""


class StandardGenerationSurface(ConversationEventSurface, ColangTurnSurface, HasExplainInfo, HasLogAdapters, Protocol):
    """Surface used by ``generation.generation_workflow`` (the standard generate pass)."""


class StreamingOutputSurface(HasConfig, HasRuntime, HasLLM, HasExplainInfo, SupportsEnsureExplainInfo, Protocol):
    """Surface used by ``streaming.streaming_output_rails`` (output rails over a stream)."""


class GenerationStreamSurface(StreamingOutputSurface, SupportsGenerateAsync, Protocol):
    """Surface used by ``streaming.generation_stream`` (token stream + output rails)."""


class RailsCheckSurface(Protocol):
    """Surface used by ``checks.rails_check``.

    Narrower than ``SupportsGenerateAsync``: the check path only ever drives
    ``generate_async`` with ``messages``/``options``, so a double that supports
    just those satisfies the contract without growing the streaming arguments.
    """

    async def generate_async(self, *, messages: List[dict], options: dict) -> Any: ...
