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

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.utils import get_railsignore_patterns, is_ignored_by_railsignore

CONFIGS_FOLDER = os.path.join(os.path.dirname(__file__), ".", "test_configs")


def cleanup_railsignore(railsignore_path):
    """Helper for clearing a railsignore file."""
    try:
        with open(railsignore_path, "w") as f:
            pass
    except OSError as e:
        print(f"Error: Unable to create {railsignore_path}. {e}")
    else:
        print(f"Successfully cleaned up .railsignore: {railsignore_path}")


@pytest.fixture(scope="function")
def cleanup():
    # Get the system's temporary directory
    temp_dir = Path(tempfile.gettempdir())
    # Create a path for the .railsignore file in the temp directory
    railsignore_path = temp_dir / ".railsignore"

    # Mock the path to the .railsignore file
    with patch("nemoguardrails.utils.get_railsignore_path") as mock_get_railsignore_path:
        mock_get_railsignore_path.return_value = railsignore_path

        # Ensure the mock file exists
        railsignore_path.touch()

        # Clean railsignore file before
        cleanup_railsignore(railsignore_path)

        # Yield control to test
        yield railsignore_path

        # Clean railsignore file after
        cleanup_railsignore(railsignore_path)

        # Remove the mock file
        if railsignore_path.exists():
            railsignore_path.unlink()


def test_railsignore_config_loading(cleanup):
    railsignore_path = cleanup
    # Setup railsignore
    append_railsignore(railsignore_path, "ignored_config.co")

    # Load config
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "railsignore_config"))

    config_string = str(config)
    # Assert .railsignore successfully ignores
    assert "ignored_config.co" not in config_string

    # Other files should load successfully
    assert "config_to_load.co" in config_string


def test_get_railsignore_patterns(cleanup):
    railsignore_path = cleanup
    # Empty railsignore
    ignored_files = get_railsignore_patterns(railsignore_path)

    assert "ignored_module.py" not in ignored_files
    assert "ignored_colang.co" not in ignored_files

    # Append files to railsignore
    append_railsignore(railsignore_path, "ignored_module.py")
    append_railsignore(railsignore_path, "ignored_colang.co")

    # Grab ignored files
    ignored_files = get_railsignore_patterns(railsignore_path)

    # Check files exist
    assert "ignored_module.py" in ignored_files
    assert "ignored_colang.co" in ignored_files

    # Append comment and whitespace
    append_railsignore(railsignore_path, "# This_is_a_comment.py")
    append_railsignore(railsignore_path, "  ")
    append_railsignore(railsignore_path, "")

    # Grab ignored files
    ignored_files = get_railsignore_patterns(railsignore_path)

    # Comments and whitespace not retrieved
    assert "# This_is_a_comment.py" not in ignored_files
    assert "  " not in ignored_files
    assert "" not in ignored_files

    # Assert files still exist
    assert "ignored_module.py" in ignored_files
    assert "ignored_colang.co" in ignored_files


def test_is_ignored_by_railsignore(cleanup):
    railsignore_path = cleanup
    # Append files to railsignore
    append_railsignore(railsignore_path, "ignored_module.py")
    append_railsignore(railsignore_path, "ignored_colang.co")

    # Grab ignored files
    ignored_files = get_railsignore_patterns(railsignore_path)

    # Check if files are ignored
    assert is_ignored_by_railsignore("ignored_module.py", ignored_files)
    assert is_ignored_by_railsignore("ignored_colang.co", ignored_files)
    assert not is_ignored_by_railsignore("not_ignored.py", ignored_files)


def append_railsignore(railsignore_path: str, file_name: str) -> None:
    """Helper for appending to a railsignore file."""
    try:
        with open(railsignore_path, "a") as f:
            f.write(file_name + "\n")
    except FileNotFoundError:
        print(f"No {railsignore_path} found in the current directory.")
    except OSError as e:
        print(f"Error: Failed to write to {railsignore_path}. {e}")
