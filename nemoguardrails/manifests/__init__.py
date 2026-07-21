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

"""Public API for rail manifests, catalog access, and configured surface references."""

from nemoguardrails.manifests.catalog import RailCatalog, RailManifestRecord
from nemoguardrails.manifests.installation import (
    InstallationPackage,
    InstallationPlan,
    InstallationResource,
    build_installation_plan,
)
from nemoguardrails.manifests.manifest import (
    ActionRef,
    Binding,
    ConfigSpecRef,
    EnvVar,
    ImportRef,
    ModelRequirement,
    PythonPackage,
    RailActions,
    RailCapability,
    RailCategory,
    RailConfigSchema,
    RailDirection,
    RailFlows,
    RailLifecycle,
    RailManifest,
    RailMetadata,
    RailPrivacy,
    RailRequirements,
    RailSpec,
    RailSurface,
    ServiceRequirement,
    TransformTarget,
    import_ref_target,
    iter_manifest_import_refs,
    resolve_import_ref,
)
from nemoguardrails.manifests.registry import (
    all_rail_manifests,
    default_rail_catalog,
)
from nemoguardrails.manifests.surface_reference import (
    normalize_configured_surface_name,
    parse_configured_surface,
)
from nemoguardrails.manifests.validation import (
    RailDependencyError,
    RailValidationReport,
    RailValidationResult,
    RequirementCheck,
    RequirementStatus,
    configured_rail_manifests,
    raise_for_missing_package,
    require_python_package,
    validate_rail_requirements,
)

__all__ = [
    "ActionRef",
    "Binding",
    "ConfigSpecRef",
    "EnvVar",
    "ImportRef",
    "InstallationPackage",
    "InstallationPlan",
    "InstallationResource",
    "ModelRequirement",
    "PythonPackage",
    "RailActions",
    "RailCapability",
    "RailCatalog",
    "RailCategory",
    "RailConfigSchema",
    "RailDirection",
    "RailDependencyError",
    "RailFlows",
    "RailLifecycle",
    "RailManifest",
    "RailManifestRecord",
    "RailValidationReport",
    "RailValidationResult",
    "RailMetadata",
    "RailPrivacy",
    "RailRequirements",
    "RailSpec",
    "RailSurface",
    "ServiceRequirement",
    "RequirementCheck",
    "RequirementStatus",
    "TransformTarget",
    "all_rail_manifests",
    "build_installation_plan",
    "configured_rail_manifests",
    "default_rail_catalog",
    "import_ref_target",
    "iter_manifest_import_refs",
    "normalize_configured_surface_name",
    "parse_configured_surface",
    "raise_for_missing_package",
    "require_python_package",
    "resolve_import_ref",
    "validate_rail_requirements",
]
