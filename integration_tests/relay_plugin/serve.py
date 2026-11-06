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

"""Serves one config with the hooks server, on a listening socket the test process passes by file descriptor."""

# The test process runs this with the repository's .venv Python, since its own environment has no Guardrails.
# The CLI's `hooks-server` binds its own port, and a port picked for it could be taken before it binds; a socket the
# test process already listens on can't be, and connections to it wait in its backlog until the server starts.

import argparse
import logging
import socket

import uvicorn

from nemoguardrails import RailsConfig
from nemoguardrails.server.hooks.app import create_hooks_app


def main() -> None:
    """Serves the config at --config on the socket at --fd until the process is stopped."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="The configuration directory to serve.")
    parser.add_argument("--fd", type=int, required=True, help="The file descriptor of a listening TCP socket.")
    args = parser.parse_args()

    # As the CLI does, so the log holds the line the server writes for each check.
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    app = create_hooks_app(RailsConfig.from_path(args.config))
    server = uvicorn.Server(uvicorn.Config(app, log_level="info"))
    server.run(sockets=[socket.socket(fileno=args.fd)])


if __name__ == "__main__":
    main()
