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

"""Utilities for handling optional dependencies."""

import importlib
import warnings
from typing import Any, Optional


def optional_import(
    module_name: str, package_name: Optional[str] = None, error: str = "raise", extra: Optional[str] = None
) -> Any:
    """Import an optional dependency.

    Args:
        module_name: The module name to import.
        package_name: The package name for installation messages (defaults to module_name).
        error: What to do when dependency is not found. One of "raise", "warn", "ignore".
        extra: The name of the extra dependency group.

    Returns:
        The imported module, or None if not available and error="ignore".

    Raises:
        ImportError: If the module is not available and error="raise".
    """
    package_name = package_name or module_name

    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        if error == "raise":
            extra_msg = f" Install with: poetry install -E {extra}" if extra else ""
            msg = (
                f"Missing optional dependency '{package_name}'. "
                f"Use pip install {package_name} or poetry add {package_name}.{extra_msg}"
            )
            raise ImportError(msg) from e
        elif error == "warn":
            extra_msg = f" Install with: poetry install -E {extra}" if extra else ""
            msg = (
                f"Missing optional dependency '{package_name}'. "
                f"Use pip install {package_name} or poetry add {package_name}.{extra_msg}"
            )
            warnings.warn(msg, ImportWarning, stacklevel=2)
        return None


def check_optional_dependency(
    module_name: str, package_name: Optional[str] = None, extra: Optional[str] = None
) -> bool:
    """Check if an optional dependency is available.

    Args:
        module_name: The module name to check.
        package_name: The package name for installation messages (defaults to module_name).
        extra: The name of the extra dependency group.

    Returns:
        True if the module is available, False otherwise.
    """
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def import_optional_dependency(
    name: str,
    extra: Optional[str] = None,
    errors: str = "raise",
    min_version: Optional[str] = None,
) -> Any:
    """Import an optional dependency, inspired by pandas implementation.

    Args:
        name: The module name.
        extra: The name of the extra dependency group.
        errors: What to do when a dependency is not found or its version is too old.
            One of 'raise', 'warn', 'ignore'.
        min_version: Specify a minimum version that is different from the global version.

    Returns:
        The imported module or None.
    """
    assert errors in {"warn", "raise", "ignore"}

    package_name = name
    install_name = name

    try:
        module = importlib.import_module(name)
    except ImportError:
        if errors == "raise":
            extra_msg = f" Install it via poetry install -E {extra}" if extra else ""
            raise ImportError(f"Missing optional dependency '{install_name}'.{extra_msg}")
        elif errors == "warn":
            extra_msg = f" Install it via poetry install -E {extra}" if extra else ""
            warnings.warn(
                f"Missing optional dependency '{install_name}'.{extra_msg} Functionality will be limited.",
                ImportWarning,
                stacklevel=2,
            )
        return None

    # Version checking logic can be added here if needed
    if min_version:
        version = getattr(module, "__version__", None)
        if version:
            try:
                from packaging import version as version_mod
            except ImportError:
                pass
            else:
                if version_mod.parse(version) < version_mod.parse(min_version):
                    if errors == "raise":
                        raise ImportError(
                            f"NeMo Guardrails requires version '{min_version}' or newer of '{package_name}' "
                            f"(version '{version}' currently installed)."
                        )
                    elif errors == "warn":
                        warnings.warn(
                            f"NeMo Guardrails requires version '{min_version}' or newer of '{package_name}' "
                            f"(version '{version}' currently installed). Some functionality may be limited.",
                            ImportWarning,
                            stacklevel=2,
                        )

    return module


# Commonly used optional dependencies with their extra groups
OPTIONAL_DEPENDENCIES = {
    "openai": "openai",
    "langchain": None,  # Not in extras
    "langchain_openai": "openai",
    "langchain_community": None,
    "langchain_nvidia_ai_endpoints": "nvidia",
    "torch": None,
    "transformers": None,
    "presidio_analyzer": None,
    "presidio_anonymizer": None,
    "spacy": None,
}


def get_optional_dependency(name: str, errors: str = "raise") -> Any:
    """Get an optional dependency using predefined settings.

    Args:
        name: The module name (should be in OPTIONAL_DEPENDENCIES).
        errors: What to do when a dependency is not found. One of 'raise', 'warn', 'ignore'.

    Returns:
        The imported module or None.
    """
    extra = OPTIONAL_DEPENDENCIES.get(name)
    return import_optional_dependency(name, extra=extra, errors=errors)
