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

from nemoguardrails.manifests import config_schema


def test_rail_config_spec_carries_annotation_field_and_key():
    field_info = config_schema.rail_field(default="threshold", description="score cutoff")
    spec = config_schema.RailConfigSpec(annotation=float, field_info=field_info, key="threshold")

    assert spec.annotation is float
    assert spec.key == "threshold"
    assert spec.exports == {}
    assert spec.field_info is field_info


def test_rail_config_base_model_ignores_unknown_keys():
    class SampleConfig(config_schema.RailConfigBaseModel):
        threshold: int = 1

    config = SampleConfig.model_validate({"threshold": 2, "unknown": 9})

    assert config.threshold == 2
    assert not hasattr(config, "unknown")
