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

    Attaches a handler if none exist, or updates existing handlers if they do.
    **If a handler is provided on repeat calls, it is ignored to avoid accumulating handlers.**
    Sets level and formatter of all handlers so that all modules under this package
    (model_engine, api_engine, rails_manager, etc.) inherit the same settings.

    """
    logger = logging.getLogger("nemoguardrails.guardrails")
    logger.setLevel(level)

    if formatter is None:
        formatter = logging.Formatter(DEFAULT_FORMAT, datefmt=DEFAULT_DATEFMT)

    # If the logger already has handlers, update logger level and handler level and formatters
    if logger.handlers:
        for log_handler in logger.handlers:
            log_handler.setLevel(level)
            log_handler.setFormatter(formatter)
        return logger

    # There are no handlers. So create one (if needed) and set level and formatter
    if handler is None:
        handler = logging.StreamHandler()

    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    return logger
