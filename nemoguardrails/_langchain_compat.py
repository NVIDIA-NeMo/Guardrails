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

"""Compatibility shim for langchain on Python 3.14+ (PEP 649).

Python 3.14 introduces PEP 649 (deferred evaluation of annotations).
Under PEP 649, pydantic resolves type annotations using vars(cls),
which means a class method named ``dict()`` shadows the builtin
``dict`` type.  langchain 0.3.x defines ``Chain.dict()`` and similar
methods, so ``dict[str, Any]`` annotations resolve to the *method*
rather than the builtin — raising ``TypeError: 'function' object is
not subscriptable``.

The fix landed in langchain 1.x (``langchain-classic``) via
langchain-ai/langchain#33575, which explicitly uses ``builtins.dict``
in annotations.  This module provides a workaround for environments
still running langchain 0.3.x on Python >= 3.14.
"""

import builtins
import logging
import sys

log = logging.getLogger(__name__)

_NEEDS_PATCH = sys.version_info >= (3, 14)


def _patch_langchain_dict_shadow():
    """Inject ``builtins.dict`` into langchain module namespaces.

    On Python >= 3.14, PEP 649 evaluates annotations lazily using the
    module-level globals.  By injecting the real ``dict`` builtin into
    the module's namespace *before* pydantic triggers annotation
    evaluation, we prevent the ``Chain.dict`` method from shadowing it.

    This must be called **before** any langchain class that defines a
    ``dict()`` method is used by pydantic.
    """
    if not _NEEDS_PATCH:
        return

    # Modules whose classes define a dict() method that shadows the builtin.
    # These are the known offenders in langchain 0.3.x.
    _MODULES_TO_PATCH = [
        "langchain.chains.base",
        "langchain.chains",
        "langchain.schema",
        "langchain.schema.runnable.base",
    ]

    patched = []
    for mod_name in _MODULES_TO_PATCH:
        mod = sys.modules.get(mod_name)
        if mod is not None and not hasattr(mod, "__dict_patched__"):
            mod.dict = builtins.dict  # type: ignore[attr-defined]
            mod.__dict_patched__ = True  # type: ignore[attr-defined]
            patched.append(mod_name)

    if patched:
        log.debug("Patched builtins.dict into langchain modules: %s", patched)


def safe_import_langchain():
    """Import the ``langchain`` package with PEP 649 safety.

    Returns the imported ``langchain`` module, or raises a clear error
    if langchain 0.3.x cannot be loaded on Python >= 3.14.
    """
    try:
        import langchain

        _patch_langchain_dict_shadow()
        return langchain
    except TypeError as exc:
        if _NEEDS_PATCH and "not subscriptable" in str(exc):
            raise ImportError(
                "langchain 0.3.x is not compatible with Python 3.14+ due to "
                "PEP 649 (deferred annotation evaluation).  The Chain.dict() "
                "method shadows the builtin dict type during annotation "
                "resolution.  Please upgrade to langchain >= 1.0.0 "
                "(langchain-classic) which includes the fix "
                "(langchain-ai/langchain#33575)."
            ) from exc
        raise


def import_init_chat_model():
    """Import ``init_chat_model`` with fallback for langchain 1.x.

    In langchain 1.x the import path may differ from 0.3.x.
    """
    try:
        from langchain.chat_models import init_chat_model

        return init_chat_model
    except (ImportError, TypeError):
        pass

    # langchain 1.x / langchain-classic may expose it differently
    try:
        from langchain_classic.chat_models import init_chat_model  # type: ignore[import-not-found]

        return init_chat_model
    except ImportError:
        pass

    raise ImportError(
        "Could not import init_chat_model from langchain.  "
        "On Python >= 3.14, langchain >= 1.0.0 is required."
    )


def import_chat_models_base():
    """Import ``langchain.chat_models.base`` with fallback.

    Used by providers.py to discover supported chat providers.
    """
    try:
        import langchain.chat_models.base as _base

        return _base
    except (ImportError, TypeError):
        pass

    try:
        import langchain_classic.chat_models.base as _base  # type: ignore[import-not-found]

        return _base
    except ImportError:
        pass

    return None
