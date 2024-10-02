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

import os
import shutil

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.utils import get_railsignore_path, get_railsignore_patterns

CONFIGS_FOLDER = os.path.join(os.path.dirname(__file__), ".", "test_configs")


@pytest.fixture(scope="function")
def cleanup():
    # Copy current rails ignore and prepare for tests
    railsignore_path = get_railsignore_path()

    temp_file_path = str(railsignore_path) + "-copy"

    # Copy the original .railsignore to a temporary file
    shutil.copy(railsignore_path, temp_file_path)
    print(f"Copied {railsignore_path} to {temp_file_path}")

    # Clean railsignore file before
    cleanup_railsignore()

    # Yield control to test
    yield

    # Clean railsignore file before
    cleanup_railsignore()

    # Restore the original .railsignore from the temporary copy
    shutil.copy(temp_file_path, railsignore_path)
    print(f"Restored {railsignore_path} from {temp_file_path}")

    # Delete the temporary file
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)
        print(f"Deleted temporary file {temp_file_path}")


def test_railsignore_config_loading(cleanup):
    # Setup railsignore
    append_railsignore("ignored_config.co")

    # Load config
    config = RailsConfig.from_path(os.path.join(CONFIGS_FOLDER, "railsignore_config"))

    config_string = str(config)
    # Assert .railsignore successfully ignores
    assert "ignored_config.co" not in config_string

    # Other files should load successfully
    assert "config_to_load.co" in config_string


def test_get_railsignore_files(cleanup):
    # Empty railsignore
    ignored_files = get_railsignore_patterns()

    assert "ignored_module.py" not in ignored_files
    assert "ignored_colang.co" not in ignored_files

    # Append files to railsignore
    append_railsignore("ignored_module.py")
    append_railsignore("ignored_colang.co")

    # Grab ignored files
    ignored_files = get_railsignore_patterns()

    # Check files exist
    assert "ignored_module.py" in ignored_files
    assert "ignored_colang.co" in ignored_files

    # Append comment and whitespace
    append_railsignore("# This_is_a_comment.py")
    append_railsignore("  ")
    append_railsignore("")

    # Grab ignored files
    ignored_files = get_railsignore_patterns()

    # Comments and whitespace not retrieved
    assert "# This_is_a_comment.py" not in ignored_files
    assert "  " not in ignored_files
    assert "" not in ignored_files

    # Assert files still exist
    assert "ignored_module.py" in ignored_files
    assert "ignored_colang.co" in ignored_files


def cleanup_railsignore():
    """
    Helper for clearing a railsignore file.
    """
    railsignore_path = get_railsignore_path()

    try:
        with open(railsignore_path, "w") as f:
            pass
    except OSError as e:
        print(f"Error: Unable to create {railsignore_path}. {e}")
    else:
        print(f"Successfully cleaned up .railsignore: {railsignore_path}")


def append_railsignore(file_name: str) -> None:
    """
    Helper for appending to a railsignore file.
    """
    railsignore_path = get_railsignore_path()

    try:
        with open(railsignore_path, "a") as f:
            f.write(file_name + "\n")
    except FileNotFoundError:
        print(f"No {railsignore_path} found in the current directory.")
    except OSError as e:
        print(f"Error: Failed to write to {railsignore_path}. {e}")
