"""
Sphinx extension to generate JSON output for every page alongside HTML output.

This extension creates parallel JSON files for each document containing metadata,
content, and other structured data that can be consumed by search engines, APIs,
or other applications.

See README.md for detailed configuration options and usage examples.
"""

from typing import Any

from sphinx.application import Sphinx

from .config import get_default_settings, validate_config
from .processing import on_build_finished


def setup(app: Sphinx) -> dict[str, Any]:
    """Setup function for Sphinx extension."""
    # Add configuration with default settings
    default_settings = get_default_settings()
    app.add_config_value("json_output_settings", default_settings, "html")

    # Connect to build events
    app.connect("config-inited", validate_config)
    app.connect("build-finished", on_build_finished)

    return {
        "version": "1.0.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
