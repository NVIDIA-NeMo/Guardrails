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

import asyncio
from typing import Any, Dict, List, Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.llms.base import LLM


def query_tabular_data(usr_query: str, gpt: any, raw_data_frame: any):
    """Answer a question based on some tabular data."""

    cut_idx = usr_query.find("based on")
    usr_query = usr_query[:cut_idx] + "?"

    # TODO: check if there's a way to do this grouping dynamically
    grouped_by_cols = []

    if any(
        word in usr_query for word in ["first class", "second class", "third class"]
    ):
        grouped_by_cols.append("Class")
    elif any(
        word in usr_query for word in ["port", "Queenstown", "Southampton", "Cherbourg"]
    ):
        grouped_by_cols.append("port")
    elif any(
        word in usr_query for word in ["female", "male", "man", "woman", "men", "women"]
    ):
        grouped_by_cols.append("Sex")
    else:
        pass

    d = raw_data_frame.groupby(grouped_by_cols, as_index=False)["Lived"].value_counts()

    # flatten the grouped by pandas series to flatten dictionary
    d2 = d.reset_index(inplace=False)
    gpt.set_dataframe(d2)

    out = gpt.ask(usr_query)

    return out, d2.to_string()


class TabularLLM(LLM):
    """LLM wrapping for GPT4Pandas."""

    model: str = ""
    temperature: float = 0.7
    tokens_to_generate: int = 256
    stop: Optional[List[str]] = None

    # This is the GPT4Pandas instance
    gpt: Any

    # The path to the raw data
    raw_data_path: str

    # This is the raw data frame associated with the tabular LLM
    raw_data_frame: Any

    @property
    def _default_params(self) -> Dict[str, Any]:
        return {}

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {**{"model": self.model}, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "tabular_llm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> str:
        raise Exception("Sync mode not supported.")

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> str:
        result, processed_data = query_tabular_data(
            usr_query=prompt, gpt=self.gpt, raw_data_frame=self.raw_data_frame
        )

        return "###".join([result, self.raw_data_path, processed_data])
