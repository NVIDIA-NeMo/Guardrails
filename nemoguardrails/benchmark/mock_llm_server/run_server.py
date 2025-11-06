#!/usr/bin/env python3
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

"""
Startup script for the Mock LLM Server.

This script starts the FastAPI server with configurable host and port settings.
"""

import argparse
import logging
import os
import sys

import uvicorn

from nemoguardrails.benchmark.mock_llm_server.config import CONFIG_FILE_ENV_VAR

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


def parse_arguments():
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
        "--config-file", help=".env file to configure model", required=True
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of uvicorn worker processes (default: 1)",
    )
    return parser.parse_args()


def validate_config_file(config_file):
    if not config_file:
        raise RuntimeError(
            "No CONFIG_FILE environment variable set, or --config-file CLI argument"
        )

    if not (os.path.exists(config_file) and os.path.isfile(config_file)):
        raise RuntimeError(f"Can't open {config_file}")

    return config_file


def main():  # pragma: no cover
    args = parse_arguments()

    config_file = os.environ.get("CONFIG_FILE", args.config_file)
    config_file = validate_config_file(config_file)

    log.info("Using config file: %s", config_file)
    os.environ[CONFIG_FILE_ENV_VAR] = config_file

    log.info(f"Starting Mock LLM Server on {args.host}:{args.port}")
    log.info(f"OpenAPI docs available at: http://{args.host}:{args.port}/docs")
    log.info(f"Health check at: http://{args.host}:{args.port}/health")
    log.info(f"Serving model with config {config_file}")
    log.info("Press Ctrl+C to stop the server")

    try:
        uvicorn.run(
            "nemoguardrails.benchmark.mock_llm_server.api:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level,
            env_file=config_file,
            workers=args.workers,
        )
    except KeyboardInterrupt:
        log.info("\nServer stopped by user")
    except Exception as e:  # pylint: disable=broad-except
        log.error(f"Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
