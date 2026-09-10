#!/usr/bin/env python3
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

"""Startup script for the Mock Tool LLM Server, launched as a subprocess by
tests/guardrails/test_per_tool_regex_rails_server.py. Behavior is configured entirely
via the MOCK_TOOL_LLM_* environment variables read in api.py -- no config file, unlike
benchmark/mock_llm_server.
"""

import argparse

import uvicorn


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the Mock Tool LLM Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the server to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8010, help="Port to bind the server to (default: 8010)")
    parser.add_argument(
        "--log-level",
        default="warning",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Log level (default: warning)",
    )
    return parser.parse_args()


def main():  # pragma: no cover
    args = parse_arguments()
    uvicorn.run("tests.mock_tool_llm_server.api:app", host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":  # pragma: no cover
    main()
