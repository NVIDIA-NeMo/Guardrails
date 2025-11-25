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
import re
import subprocess
from pathlib import Path

import typer
import yaml

app = typer.Typer()
app.pretty_exceptions_enable = False


def _remove_code_blocks_with_text(md_file_path, text_to_remove):
    # Define a regular expression pattern to match code blocks
    code_block_pattern = re.compile(r"```.*?```", re.DOTALL)

    # Read the content of the Markdown file
    with open(md_file_path, "r", encoding="utf-8") as file:
        content = file.read()

    # Find all code blocks
    code_blocks = code_block_pattern.findall(content)

    # Filter out code blocks containing the specific text
    blocks_to_remove = [block for block in code_blocks if text_to_remove in block]

    # Remove the blocks from content
    for block in blocks_to_remove:
        content = content.replace(block, "")

    # Write the modified content back to the file
    with open(md_file_path, "w", encoding="utf-8") as file:
        file.write(content)


def _fix_prefix_and_type_in_code_blocks(md_file_path):
    # Define a regular expression pattern to match code blocks
    code_block_pattern = re.compile(r"```.*?```", re.DOTALL)

    # Read the content of the Markdown file
    with open(md_file_path, "r", encoding="utf-8") as file:
        content = file.read()

    # Find all code blocks
    code_blocks = code_block_pattern.findall(content)

    for block in code_blocks:
        lines = block.split("\n")
        start_with_four_spaces = True
        for i in range(1, len(lines) - 1):
            if not lines[i].startswith("    "):
                start_with_four_spaces = False
                break

        if start_with_four_spaces:
            for i in range(1, len(lines) - 1):
                lines[i] = lines[i][4:]

            # print(f"Need to remove prefix for block:\n{block}")
            updated_block = "\n".join(lines)

            content = content.replace(block, updated_block)
            block = updated_block

        # Next, we also try to fix the type of the block using some heuristics
        if lines[0] == "```python" or lines[0] == "```":
            # If it parses correctly as a yaml file, we mark it as yaml
            try:
                data = yaml.safe_load("\n".join(lines[1:-1]))
                if isinstance(data, dict) and " " not in list(data.keys())[0]:
                    lines[0] = "```yaml"
                    updated_block = "\n".join(lines)
                    content = content.replace(block, updated_block)
                    block = updated_block
            except Exception:
                pass

        if lines[0] == "```" and "from nemoguardrails" in block:
            lines[0] = "```python"
            updated_block = "\n".join(lines)
            content = content.replace(block, updated_block)
            block = updated_block

        if lines[0].startswith("```py "):
            lines[0] = "```python"
            updated_block = "\n".join(lines)
            content = content.replace(block, updated_block)
            block = updated_block

        if lines[0].startswith("```co "):
            lines[0] = "```colang"
            updated_block = "\n".join(lines)
            content = content.replace(block, updated_block)
            block = updated_block

        if lines[0].startswith("```yml"):
            lines[0] = "```yaml"
            updated_block = "\n".join(lines)
            content = content.replace(block, updated_block)
            block = updated_block

    # Write the modified content back to the file
    with open(md_file_path, "w", encoding="utf-8") as file:
        file.write(content)


def _remove_specific_text(md_file_path, text_to_remove):
    # Read the content of the Markdown file
    with open(md_file_path, "r", encoding="utf-8") as file:
        content = file.read()

    # Replace the specified text with an empty string
    content = content.replace(text_to_remove, "")

    # Write the modified content back to the file
    with open(md_file_path, "w", encoding="utf-8") as file:
        file.write(content)


def _post_process(md_file_path):
    # Read the content of the Markdown file
    with open(md_file_path, "r", encoding="utf-8") as file:
        content = file.read()

    content = re.sub(r"\n<!-- WARNING.*\n", "\n", content)
    content = re.sub(r"\n<.?CodeOutputBlock.*\n", "\n", content)
    content = re.sub(r"\n\n\n+", "\n\n", content)
    content = re.sub(r"\n +\n", "\n\n", content)
    content = re.sub(r"\n    \n", "\n\n", content)
    content = re.sub(r"\n\n+$", "\n", content)

    # Write the modified content back to the file
    with open(md_file_path, "w", encoding="utf-8") as file:
        file.write(content)


# Function to run the nbdoc_build command
def run_nbdoc_build(srcdir, force_all):
    try:
        # Run the nbdoc_build command with specified arguments
        subprocess.run(
            ["nbdoc_build", "--srcdir", srcdir, "--force_all", str(force_all)],
            check=True,
        )

    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running nbdoc_build: {e}")
        return False
    return True


# Function to recursively rename .md files to README.md
def rename_md_to_readme(start_dir):
    generated = set()

    for path in Path(start_dir).rglob("*.md"):
        if path.name == "README.md":
            # if path.exists() and not path.absolute() in generated:
            #     path.unlink()
            continue

        # Skip processing the root directory
        if path.parent.name == "getting-started":
            continue

        # Generate the new file name, assuming the path as a directory with README.md
        readme_path = path.parent / "README.md"

        # # Skip if README.md already exists
        if readme_path.exists():
            print(f"{readme_path} already exists, deleting.")
            readme_path.unlink()

        # Rename the file
        path.rename(readme_path)
        print(f"Renamed {path} to {readme_path}")
        generated.add(readme_path.absolute())
        print(f"Adding {readme_path.absolute()}")

        # We do some additional post-processing
        _remove_code_blocks_with_text(readme_path.absolute(), "# Init:")
        _remove_code_blocks_with_text(readme_path.absolute(), "# Hide from documentation page.")

        _remove_code_blocks_with_text(
            readme_path.absolute(),
            "huggingface/tokenizers: The current process just got forked",
        )
        _remove_code_blocks_with_text(readme_path.absolute(), "Writing config/")
        _remove_code_blocks_with_text(readme_path.absolute(), "Appending to config/")
        _remove_specific_text(
            readme_path.absolute(),
            '<CodeOutputBlock lang="bash">\n\n\n\n</CodeOutputBlock>',
        )

        _fix_prefix_and_type_in_code_blocks(readme_path.absolute())

        _post_process(readme_path.absolute())


@app.command()
def convert(folder: str):
    """Convert a Jupyter notebook in the provided folder to .md.

    It creates a README.md file next to the Jupyter notebook.
    """
    print(f"Processing {folder}...")

    notebooks = [f for f in os.listdir(folder) if f.endswith(".ipynb")]

    if len(notebooks) == 0:
        raise RuntimeError(f"No .ipynb file found in {folder}.")
    elif len(notebooks) > 1:
        raise RuntimeError(f"Found {len(notebooks)} in {folder}: {notebooks}.")

    print(f"Found notebook: {notebooks[0]}")

    if run_nbdoc_build(folder, True):
        # Rename .md files if nbdev_build was successful
        rename_md_to_readme(folder)
        subprocess.run(["git", "add", "."])
        subprocess.run(["pre-commit", "run", "--all-files"])
    else:
        print("nbdoc_build command failed. Exiting without renaming .md files.")


if __name__ == "__main__":
    app()
