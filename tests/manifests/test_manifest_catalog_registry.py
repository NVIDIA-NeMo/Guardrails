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

import nemoguardrails.manifests as manifests_pkg
from nemoguardrails.manifests import RailCatalog, registry


def test_package_accessors_with_no_built_in_rails():
    registry._reset_rail_manifest_cache()
    try:
        catalog = manifests_pkg.default_rail_catalog()
        assert isinstance(catalog, RailCatalog)
        # a second call returns the cached instance rather than rediscovering
        assert manifests_pkg.default_rail_catalog() is catalog

        assert manifests_pkg.all_rail_manifests() == {}
        # no enabled plugins short-circuits back to the default catalog
        assert manifests_pkg.rail_catalog() is catalog
    finally:
        registry._reset_rail_manifest_cache()
