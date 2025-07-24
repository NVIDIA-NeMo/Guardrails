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

from nemoguardrails.actions import action
from nemoguardrails.library.xgb.inference import xgb_inference
from nemoguardrails.rails.llm.config import RailsConfig


@action()
async def xgb_detect(
    source: str,
    text: str,
    config: RailsConfig,
    **kwargs,
):
    xgb_config = getattr(config.rails.config, "xgb")
    source_config = getattr(xgb_config, source)

    enabled_detectors = getattr(source_config, "detectors", None)
    if enabled_detectors is None:
        raise ValueError(
            f"Could not find 'detectors' in source_config: {source_config}"
        )
    valid_detectors = ["SPAM"]
    for detector in enabled_detectors:
        if detector not in valid_detectors:
            raise ValueError(
                f"XGB detectors can only be defined in the following detectors: {valid_detectors}. "
                f"The current detector, '{detector}' is not allowed."
            )

    valid_sources = ["input", "output"]
    if source not in valid_sources:
        raise ValueError(
            f"XGB detectors can only be defined in the following flows: {valid_sources}. "
            f"The current flow, '{source} is not allowed."
        )

    xgb_response = xgb_inference(
        text,
        enabled_detectors,
    )

    return xgb_response
