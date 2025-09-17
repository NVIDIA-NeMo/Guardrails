#!/usr/bin/env python3
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

"""
Startup script for the Mock LLM Server.

This script starts the FastAPI server with configurable host and port settings.
"""

import argparse
import sys

import uvicorn

from nemoguardrails.benchmark.mock_llm_server.config import get_config, load_config


def main():
    parser = argparse.ArgumentParser(description="Run the Mock LLM Server")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to (default: 8000)",
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Log level (default: info)",
    )

    parser.add_argument(
        "--config-file", help="YAML file to configure model", required=True
    )

    args = parser.parse_args()

    # Load model configuration
    load_config(args.config_file)
    model_config = get_config()

    # Import the app after configuration is loaded. This caches the values in the app Dependencies
    from nemoguardrails.benchmark.mock_llm_server.api import app

    print(f"Starting Mock LLM Server on {args.host}:{args.port}")
    print(f"OpenAPI docs available at: http://{args.host}:{args.port}/docs")
    print(f"Health check at: http://{args.host}:{args.port}/health")
    print(f"Model configuration: {model_config}")
    print("Press Ctrl+C to stop the server")

    try:
        uvicorn.run(
            app=app,
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level,
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:  # pylint: disable=broad-except
        print(f"Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
