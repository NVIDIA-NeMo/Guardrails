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
Live documentation server with auto-rebuild on file changes.

This script runs sphinx-autobuild to serve the documentation locally
and automatically rebuilds it when source files change.

Usage:
    python serve.py [--port PORT] [--host HOST] [--open]

Options:
    --port PORT    Port to serve on (default: 8000)
    --host HOST    Host to bind to (default: 0.0.0.0)
    --open         Automatically open browser
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Live documentation server with auto-rebuild")
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to serve on (default: 8000)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Automatically open browser",
    )
    args = parser.parse_args()

    # Set up paths
    docs_dir = Path(__file__).parent
    source_dir = docs_dir
    build_dir = docs_dir / "_build" / "html"

    print("=" * 50)
    print("NeMo Guardrails Documentation Server")
    print("=" * 50)
    print()
    print(f"Starting live documentation server on port {args.port}...")
    print("Documentation will auto-rebuild on file changes.")
    print()
    print(f"Open your browser to: http://127.0.0.1:{args.port}")
    print()
    print("Press Ctrl+C to stop the server.")
    print("=" * 50)
    print()

    # Build command
    cmd = [
        "sphinx-autobuild",
        str(source_dir),
        str(build_dir),
        "--port",
        str(args.port),
        "--host",
        args.host,
        # Ignore patterns
        "--ignore",
        "*.swp",
        "--ignore",
        "*.swo",
        "--ignore",
        "*~",
        "--ignore",
        ".DS_Store",
        "--ignore",
        "_build/*",
        "--ignore",
        "*.pyc",
        "--ignore",
        "__pycache__/*",
        "--ignore",
        ".git/*",
        # Additional options
        "--delay",
        "1",
        "--watch",
        str(docs_dir.parent / "nemoguardrails"),
        "--re-ignore",
        r"_build/.*",
        "--re-ignore",
        r".*\.egg-info.*",
    ]

    if args.open:
        cmd.append("--open-browser")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\nServer stopped.")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f"\n\nError: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(
            "\n\nError: sphinx-autobuild not found. "
            "Please install it with:\n"
            "  poetry install --with docs\n"
            "or:\n"
            "  pip install sphinx-autobuild",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
