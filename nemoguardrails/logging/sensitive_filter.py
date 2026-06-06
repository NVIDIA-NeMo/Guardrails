# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Logging filter for redacting sensitive data from log records.

Integrates with Python's standard logging to automatically redact
sensitive information from all log messages.
"""

import logging
from typing import Any, Dict, Optional

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
        # Redact the main message
        if record.msg:
            if isinstance(record.msg, str):
                record.msg = self.redactor.redact(record.msg)
            elif isinstance(record.msg, dict):
                record.msg = self.redactor.redact_dict(record.msg)

        # Redact message arguments
        if record.args:
            if isinstance(record.args, dict):
                record.args = self.redactor.redact_dict(record.args)
            elif isinstance(record.args, (tuple, list)):
                record.args = tuple(
                    self.redactor.redact(arg) if isinstance(arg, str) else arg
                    for arg in record.args
                )

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
    """Add sensitive data filter to a logger.

    Args:
        logger: Logger to add filter to (root logger if None)
        redactor: Optional custom redactor instance

    Returns:
        The created filter instance
    """
    if logger is None:
        logger = logging.getLogger()

    filter_instance = SensitiveDataFilter(redactor=redactor)
    logger.addFilter(filter_instance)
    return filter_instance


def setup_all_loggers(redactor: Optional[SensitiveDataRedactor] = None) -> None:
    """Add sensitive data filter to all active loggers.

    Args:
        redactor: Optional custom redactor instance
    """
    root_logger = logging.getLogger()
    setup_sensitive_data_filter(root_logger, redactor=redactor)

    # Also add to commonly used loggers
    for logger_name in ['nemoguardrails', 'langchain', 'llama_index', 'openai']:
        logger = logging.getLogger(logger_name)
        setup_sensitive_data_filter(logger, redactor=redactor)
