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

import logging

DEFAULT_FORMAT = "%(asctime)s %(levelname)s: %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(
    level: int = logging.INFO,
    formatter: logging.Formatter | None = None,
    handler: logging.Handler | None = None,
) -> logging.Logger:
    """Configure logging for the ``nemoguardrails.guardrails`` package.

    Attaches a handler with a formatter to the ``nemoguardrails.guardrails``
    logger so that all modules under this package (model_engine, api_engine,
    rails_manager, etc.) inherit the same settings.

    """
    logger = logging.getLogger("nemoguardrails.guardrails")

    # If the logger already has handlers, update logger and all handler levels
    if logger.handlers:
        logger.setLevel(level)
        for log_handler in logger.handlers:
            log_handler.setLevel(level)
        return logger

    # If there are no handlers, create them and add them to the logger
    logger.setLevel(level)

    if formatter is None:
        formatter = logging.Formatter(DEFAULT_FORMAT, datefmt=DEFAULT_DATEFMT)

    if handler is None:
        handler = logging.StreamHandler()

    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    return logger
