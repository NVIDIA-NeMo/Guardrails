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

"""Module for handling Clavata requests."""

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Type, TypeVar

import aiohttp
from pydantic import BaseModel, Field, ValidationError

from nemoguardrails.library.clavata.utils import exponential_backoff

from .errs import (
    ClavataPluginAPIError,
    ClavataPluginAPIRateLimitError,
    ClavataPluginConfigurationError,
    ClavataPluginValueError,
)

log = logging.getLogger(__name__)


_CLAVATA_API_KEY = os.environ.get("CLAVATA_API_KEY")


@dataclass
class AuthHeader:
    """Represents the authorization header structure for Clavata API requests."""

    api_key: Optional[str] = None

    def to_headers(self) -> Dict[str, str]:
        """
        Converts the auth token into a dictionary that can be used with aiohttp
        to supply headers for the request.
        """
        api_key = self.api_key or _CLAVATA_API_KEY
        if api_key is None:
            raise ClavataPluginConfigurationError(
                "CLAVATA_API_KEY environment variable is not set. "
                "An API_KEY is required to make a request to the Clavata API."
            )

        return {"Authorization": f"Bearer {api_key}"}


class ContentData(BaseModel):
    """Represents the content data structure for Clavata API requests."""

    text: str


class JobRequest(BaseModel):
    """Represents the job request structure for Clavata API requests."""

    content_data: List[ContentData]
    policy_id: str
    wait_for_completion: bool = Field(default=True)


JobStatus = Literal[
    "JOB_STATUS_UNSPECIFIED",
    "JOB_STATUS_PENDING",
    "JOB_STATUS_RUNNING",
    "JOB_STATUS_COMPLETED",
    "JOB_STATUS_FAILED",
    "JOB_STATUS_CANCELED",
]

Outcome = Literal[
    "OUTCOME_UNSPECIFIED", "OUTCOME_TRUE", "OUTCOME_FALSE", "OUTCOME_FAILED"
]


class SectionReport(BaseModel):
    """A section report for a job result"""

    name: str = Field(description="The name of the section")
    message: str = Field(
        description="""An arbitrary message attached to the section in the policy.
        Often used to match an internal identifier or to provide an action to take."""
    )
    result: Outcome = Field(description="The result of the section")


class Report(BaseModel):
    """A report for a job result"""

    result: Outcome = Field(description="The result of the report")
    sectionEvaluationReports: List[SectionReport] = Field(
        description="The section evaluation reports for the job result"
    )


class Result(BaseModel):
    """A JobResult. One will be created for each content item submitted."""

    report: Report = Field(description="The report for the job result")


class Job(BaseModel):
    """A job in Clavata"""

    status: JobStatus = Field(description="The status of the job")
    results: List[Result] = Field(description="The results of the job")


class CreateJobResponse(BaseModel):
    """Response from the Clavata Create Job API"""

    job: Job = Field(description="The job that was created")


ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)


class ClavataClient:
    """A client for the Clavata API."""

    base_endpoint: str
    api_key: Optional[str]

    def __init__(self, base_endpoint: str, api_key: Optional[str] = None):
        self.base_endpoint = base_endpoint

        # API key can be passed or set in the environment variable CLAVATA_API_KEY
        self.api_key = api_key or os.environ.get("CLAVATA_API_KEY")
        if self.api_key is None:
            raise ClavataPluginConfigurationError(
                "CLAVATA_API_KEY environment variable is not set. "
                "An API_KEY is required to make a request to the Clavata API."
            )

    def _get_full_endpoint(self, endpoint: str) -> str:
        return f"{self.base_endpoint}{endpoint}"

    def _get_headers(self) -> Dict[str, str]:
        return AuthHeader(api_key=self.api_key).to_headers()

    @exponential_backoff(
        initial_delay=0.1, retry_exceptions=(ClavataPluginAPIRateLimitError,)
    )
    async def _make_request(
        self,
        endpoint: str,
        payload: BaseModel,
        response_model: Type[ResponseModelT],
    ) -> ResponseModelT:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._get_full_endpoint(endpoint),
                    json=payload.model_dump(),
                    headers=self._get_headers(),
                ) as resp:
                    if resp.status == 429:
                        # Trigger exponential backoff on rate limit errors
                        raise ClavataPluginAPIRateLimitError(
                            f"Clavata API rate limit exceeded. Status code: {resp.status}"
                        )

                    if resp.status != 200:
                        raise ClavataPluginAPIError(
                            f"Clavata call failed with status code {resp.status}.\n"
                            f"Details: {await resp.text()}"
                        )

                    try:
                        parsed_response = await resp.json()
                    except aiohttp.ContentTypeError as e:
                        raise ClavataPluginValueError(
                            f"Failed to parse Clavata response as JSON. Status: {resp.status}, "
                            f"Content: {await resp.text()}"
                        ) from e

                    # Now we actually parse the JSON into a meaningful object
                    try:
                        return response_model.model_validate(parsed_response)
                    except ValidationError as e:
                        raise ClavataPluginValueError(
                            f"Invalid response format from Clavata API. Details: {e}"
                        ) from e

        except Exception as e:
            raise ClavataPluginAPIError(
                f"Failed to make Clavata API request. Error: {e}"
            ) from e

    async def create_job(self, text: str, policy_id: str) -> Job:
        """
        Create a job in Clavata.

        Args:
            text: The text to send to the Clavata API.
            policy_id: The policy ID to use for the request.

        Returns:
            Job: The job that was created.
        """
        payload = JobRequest(
            content_data=[ContentData(text=text)],
            policy_id=policy_id,
            wait_for_completion=True,
        )

        rv = await self._make_request("/v1/jobs", payload, CreateJobResponse)
        return rv.job
