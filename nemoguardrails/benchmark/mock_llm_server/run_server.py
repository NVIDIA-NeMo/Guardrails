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
import logging
import sys

import uvicorn
from uvicorn.logging import AccessFormatter

from nemoguardrails.benchmark.mock_llm_server.config import get_config, load_config

# 1. Get a logger instance
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)  # Set the lowest level to capture all messages

# Set up formatter and direct it to the console
formatter = logging.Formatter(
    "%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # DEBUG and higher will go to the console
console_handler.setFormatter(formatter)

# Add the console handler for logging
log.addHandler(console_handler)


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

    log.info(f"Starting Mock LLM Server on {args.host}:{args.port}")
    log.info(f"OpenAPI docs available at: http://{args.host}:{args.port}/docs")
    log.info(f"Health check at: http://{args.host}:{args.port}/health")
    log.info(f"Model configuration: {model_config}")
    log.info("Press Ctrl+C to stop the server")

    try:
        uvicorn.run(
            app=app,
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level,
        )
    except KeyboardInterrupt:
        log.info("\nServer stopped by user")
    except Exception as e:  # pylint: disable=broad-except
        log.error(f"Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
