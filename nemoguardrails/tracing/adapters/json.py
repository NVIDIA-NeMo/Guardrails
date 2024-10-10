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

import json

from nemoguardrails.tracing import InteractionLog
from nemoguardrails.tracing.adapters.base import InteractionLogAdapter


class JsonAdapter(InteractionLogAdapter):
    def transform(self, interaction_log: InteractionLog):
        """Transforms the InteractionLog into a JSON string."""
        spans = []

        print("we are here")
        print(interaction_log.trace)
        for span_data in interaction_log.trace:
            span_dict = {
                "name": span_data.name,
                "span_id": span_data.span_id,
                "parent_id": span_data.parent_id,
                "trace_id": interaction_log.id,
                "start_time": span_data.start_time,
                "end_time": span_data.end_time,
                "duration": span_data.duration,
                "metrics": span_data.metrics,
            }
            spans.append(span_dict)

        log_dict = {
            "trace_id": interaction_log.id,
            "spans": spans,
        }

        with open("log.json", "w") as f:
            f.write(json.dumps(log_dict, indent=2))
