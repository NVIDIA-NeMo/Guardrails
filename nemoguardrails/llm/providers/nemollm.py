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
import logging
import os
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

import httpx
from httpx import HTTPStatusError
from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.schema import Generation
from langchain.schema.output import GenerationChunk, LLMResult
from langchain_core.language_models.llms import BaseLLM
from pydantic.v1 import root_validator

log = logging.getLogger(__name__)


class NeMoLLM(BaseLLM):
    """Wrapper around NeMo LLM large language models.

    If NGC_API_HOST, NGC_API_KEY and NGC_ORGANIZATION_ID environment variables are set,
    they will be used for the requests.
    """

    model: str = ""
    temperature: float = 0.7
    tokens_to_generate: int = 256
    stop: Optional[List[str]] = ["<extra_id_1>"]
    api_host: Optional[str] = os.environ.get(
        "NGC_API_HOST", "https://api.llm.ngc.nvidia.com"
    )
    api_key: Optional[str] = os.environ.get("NGC_API_KEY")
    organization_id: Optional[str] = os.environ.get("NGC_ORGANIZATION_ID")
    customization_id: Optional[str] = None
    streaming: bool = False
    check_api_host_version: bool = True

    @root_validator(pre=True, allow_reuse=True)
    def check_env_variables(cls, values):
        for field in ["api_host", "api_key", "organization_id"]:
            # If it's an explicit environment variable, we use that
            if values.get(field, "").startswith("$"):
                env_var_name = values[field][1:]
                values[field] = os.environ.get(env_var_name)
                if not values[field]:
                    raise Exception(f"The env var ${env_var_name} is not set!")

        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling NeMoLLM API."""
        return {
            "temperature": self.temperature,
            "tokens_to_generate": self.tokens_to_generate,
        }

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {**{"model": self.model}, **self._default_params}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "nemollm"

    def _get_request_url(self) -> str:
        if self.check_api_host_version and not self.api_host.endswith("/v1"):
            self.api_host = self.api_host + "/v1"

        if self.customization_id is None:
            url = f"{self.api_host}/models/{self.model}/completions"
        else:
            url = f"{self.api_host}/models/{self.model}/customizations/{self.customization_id}/completions"
        return url

    def _get_request_headers(self) -> Dict[str, str]:
        # nemo_llm_org_id is only needed for users in more than one LLM org
        # or if more than one LLM team within an LLM org
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        if self.organization_id is not None:
            headers["Organization-ID"] = self.organization_id
        if self.streaming:
            headers["x-stream"] = "true"
        return headers

    def _get_request_json(self, prompt: str, stop: Optional[List[str]] = None) -> Dict:
        if stop is None:
            stop = []

        return {
            "prompt": prompt,
            "stop": stop,
            **self._default_params,
        }

    def _get_timeout(self) -> httpx.Timeout:
        return httpx.Timeout(60.0, connect=10.0)

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        stop = self.stop if stop is None else stop
        with httpx.Client(timeout=self._get_timeout()) as client:
            with client.stream(
                "POST",
                url=self._get_request_url(),
                headers=self._get_request_headers(),
                json=self._get_request_json(prompt, stop),
            ) as r:
                for json_line in r.iter_lines():
                    if not json_line:
                        break
                    text = json.loads(json_line)["text"]
                    chunk = GenerationChunk(text=text)
                    yield chunk
                    if run_manager:
                        run_manager.on_llm_new_token(chunk.text, chunk=chunk)

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> str:
        stop = self.stop if stop is None else stop
        if self.streaming:
            completion = ""
            for chunk in self._stream(
                prompt=prompt, stop=stop, run_manager=run_manager, **kwargs
            ):
                completion += chunk.text
            return completion

        with httpx.Client(timeout=self._get_timeout()) as client:
            response = client.post(
                url=self._get_request_url(),
                headers=self._get_request_headers(),
                json=self._get_request_json(prompt, stop),
            )

        if response.status_code == 401:
            # Gives a more helpful error message for the forbidden status code 401
            # All other status codes except 200 and 401 are handled by response.raise_for_status()
            message = "Unauthorized request to the LLM API. Please set a valid API key using the NGC_API_KEY environment variable."
            raise HTTPStatusError(
                message=message, request=response._request, response=response
            )

        response.raise_for_status()

        return response.json()["text"]

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        stop = self.stop if stop is None else stop
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
            generations.append(
                [Generation(text=text, generation_info={"prompt": prompt})]
            )
        return LLMResult(
            generations=generations,
            llm_output={
                "url": self._get_request_url(),
                "headers": {
                    k: v
                    for k, v in self._get_request_headers().items()
                    # We make sure the Authorization header is not returned as this
                    # can lean the authorization key.
                    if k != "Authorization"
                },
                "model_name": self.model,
            },
        )

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        stop = self.stop if stop is None else stop
        async with httpx.AsyncClient(timeout=self._get_timeout()) as client:
            async with client.stream(
                "POST",
                url=self._get_request_url(),
                headers=self._get_request_headers(),
                json=self._get_request_json(prompt, stop),
            ) as r:
                async for json_line in r.aiter_lines():
                    if not json_line:
                        break
                    text = json.loads(json_line)["text"]
                    chunk = GenerationChunk(text=text)

                    # We make sure we don't sent chunks of length 0 as this will end the streaming
                    if len(chunk.text) == 0:
                        continue

                    yield chunk
                    if run_manager:
                        await run_manager.on_llm_new_token(chunk.text, chunk=chunk)

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> str:
        """Call out to NeMoLLM completion endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.
        """
        stop = self.stop if stop is None else stop
        if self.streaming:
            completion = ""
            async for chunk in self._astream(
                prompt=prompt, stop=stop, run_manager=run_manager, **kwargs
            ):
                completion += chunk.text
            return completion

        async with httpx.AsyncClient(timeout=self._get_timeout()) as client:
            response = await client.post(
                url=self._get_request_url(),
                headers=self._get_request_headers(),
                json=self._get_request_json(prompt, stop),
            )

        if response.status_code == 401:
            # Gives a more helpful error message for the forbidden status code 401
            # All other status codes except 200 and 401 are handled by response.raise_for_status()
            message = "Unauthorized request to the LLM API. Please set a valid API key using the NGC_API_KEY environment variable."
            raise HTTPStatusError(
                message=message, request=response._request, response=response
            )

        response.raise_for_status()

        return response.json()["text"]

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        stop = self.stop if stop is None else stop
        generations = []
        for prompt in prompts:
            text = await self._acall(
                prompt, stop=stop, run_manager=run_manager, **kwargs
            )
            generations.append(
                [Generation(text=text, generation_info={"prompt": prompt})]
            )
        return LLMResult(
            generations=generations,
            llm_output={
                "url": self._get_request_url(),
                "headers": {
                    k: v
                    for k, v in self._get_request_headers().items()
                    # We make sure the Authorization header is not returned as this
                    # can lean the authorization key.
                    if k != "Authorization"
                },
                "model_name": self.model,
            },
        )
