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

import logging
from typing import Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from nemoguardrails.actions.action_dispatcher import ActionDispatcher

log = logging.getLogger(__name__)

api_description = """Guardrails Action Sever API."""

app = FastAPI(
    title="Guardrails Action Server API",
    description=api_description,
    version="0.1.0",
    license_info={"name": "Apache License, Version 2.0"},
)


# Create action dispatcher object to communicate with actions
app.action_dispatcher = ActionDispatcher(load_all_actions=True)


class RequestBody(BaseModel):
    """Request body for executing an action."""

    action_name: str = ""
    action_parameters: Dict = Field(default={}, description="The list of action parameters.")


class ResponseBody(BaseModel):
    """Response body for action execution."""

    status: str = "success"  # success / failed
    result: Optional[str]


@app.post(
    "/v1/actions/run",
    summary="Execute action",
    response_model=ResponseBody,
)
async def run_action(body: RequestBody):
    """Execute the specified action and return the result.

    Args:
        body (RequestBody): The request body containing action_name and action_parameters.

    Returns:
        dict: The response containing the execution status and result.
    """

    log.info(f"Request body: {body}")
    result, status = await app.action_dispatcher.execute_action(body.action_name, body.action_parameters)
    resp = {"status": status, "result": result}
    log.info(f"Response: {resp}")
    return resp


@app.get(
    "/v1/actions/list",
    summary="List available actions",
)
async def get_actions_list():
    """Returns the list of available actions."""

    return app.action_dispatcher.get_registered_actions()
