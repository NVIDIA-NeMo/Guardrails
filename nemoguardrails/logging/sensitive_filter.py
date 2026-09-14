# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Logging filter for redacting sensitive data from log records.

Integrates with Python's standard logging to automatically redact
sensitive information from all log messages.
"""

import logging
from typing import Optional

from nemoguardrails.logging.redactor import SensitiveDataRedactor, get_redactor


class SensitiveDataFilter(logging.Filter):
    """Logging filter that redacts sensitive data from log records."""

    def __init__(self, redactor: Optional[SensitiveDataRedactor] = None):
        """Initialize the filter.

        Args:
            redactor: Optional custom redactor instance
        """
        super().__init__()
        self.redactor = redactor or get_redactor()

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log record by redacting sensitive data.

        Args:
            record: The log record to filter

        Returns:
            True (always allow the record to be logged)
        """
        # Pre-format %-style records before redacting so that a sensitive keyword
        # in the template (e.g. "password: %s") cannot corrupt the format spec,
        # which would cause TypeError in getMessage() called by the log handler.
        if isinstance(record.msg, str) and record.args:
            try:
                record.msg = record.getMessage()
                record.args = None
            except Exception:
                pass

        # Redact the main message
        if record.msg:
            if isinstance(record.msg, str):
                record.msg = self.redactor.redact(record.msg)
            elif isinstance(record.msg, dict):
                record.msg = self.redactor.redact_dict(record.msg)

        # Redact message arguments (fallback when pre-formatting was skipped or failed)
        if record.args:
            if isinstance(record.args, dict):
                record.args = self.redactor.redact_dict(record.args)
            elif isinstance(record.args, (tuple, list)):
                new_args = []
                for arg in record.args:
                    if isinstance(arg, str):
                        new_args.append(self.redactor.redact(arg))
                    elif isinstance(arg, dict):
                        new_args.append(self.redactor.redact_dict(arg))
                    else:
                        new_args.append(arg)
                record.args = tuple(new_args)

        # Redact exception information if present
        if record.exc_info:
            exc_type, exc_value, exc_tb = record.exc_info
            if exc_value:
                exc_str = str(exc_value)
                exc_str = self.redactor.redact(exc_str)
                # Update the exception with redacted message
                try:
                    exc_value.args = (exc_str,)
                except (AttributeError, TypeError):
                    pass

        return True


def setup_sensitive_data_filter(
    logger: Optional[logging.Logger] = None,
    redactor: Optional[SensitiveDataRedactor] = None,
) -> SensitiveDataFilter:
    """Add sensitive data filter to a logger's handlers.

    Args:
        logger: Logger whose handlers to protect (root logger if None)
        redactor: Optional custom redactor instance

    Returns:
        The created filter instance
    """
    if logger is None:
        logger = logging.getLogger()

    # Attach to handlers, not the logger itself. Logger.filter() is never
    # invoked for records propagated from child loggers, so a logger-level
    # filter silently bypasses every named logger in the codebase.
    _fallback = logging.lastResort
    handlers: list[logging.Handler] = logger.handlers or ([_fallback] if _fallback is not None else [])

    # Reuse an existing instance so multiple setup calls share the same
    # redactor state, but still add to every handler that lacks it.
    existing: Optional[SensitiveDataFilter] = None
    for handler in handlers:
        for f in handler.filters:
            if isinstance(f, SensitiveDataFilter):
                existing = f
                break
        if existing:
            break

    filter_instance = existing or SensitiveDataFilter(redactor=redactor)
    for handler in handlers:
        if not any(isinstance(f, SensitiveDataFilter) for f in handler.filters):
            handler.addFilter(filter_instance)
    return filter_instance


def setup_all_loggers(redactor: Optional[SensitiveDataRedactor] = None) -> None:
    """Add sensitive data filter to the root logger's handlers.

    Args:
        redactor: Optional custom redactor instance
    """
    root_logger = logging.getLogger()
    setup_sensitive_data_filter(root_logger, redactor=redactor)

    # For loggers that own their own handlers (propagate=False or extra
    # handlers attached), also protect those directly.
    for logger_name in ["nemoguardrails", "langchain", "llama_index", "openai"]:
        logger = logging.getLogger(logger_name)
        if logger.handlers:
            setup_sensitive_data_filter(logger, redactor=redactor)
