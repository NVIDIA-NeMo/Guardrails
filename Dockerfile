
# syntax=docker/dockerfile:1

# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.26@sha256:3d868e555f8f1dbc324afa005066cd11e1053fc4743b9808ca8025283e65efa5 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1
# Use copy mode so the BuildKit cache mount below works across the mount boundary
ENV UV_LINK_MODE=copy

WORKDIR /nemoguardrails
# Install deps first (cached layer — only invalidated when pyproject.toml/uv.lock change).
# The cache mount persists uv's download/build cache across image builds.
COPY pyproject.toml uv.lock /nemoguardrails/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --all-extras --no-dev --locked --no-install-project
# Copy source and install the project itself
COPY . /nemoguardrails
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --all-extras --no-dev --locked
ENV PATH="/nemoguardrails/.venv/bin:$PATH"


# Make port 8000 available to the world outside this container
EXPOSE 8000

# We copy the example bot configurations
WORKDIR /config
COPY ./examples/bots /config

# Run app.py when the container launches
WORKDIR /nemoguardrails

# Download the `all-MiniLM-L6-v2` model
RUN python -c "from fastembed.embedding import FlagEmbedding; FlagEmbedding('sentence-transformers/all-MiniLM-L6-v2');"

RUN nemoguardrails --help

ENV NEMO_GUARDRAILS_HEALTHCHECK_URL=http://localhost:8000/v1/health
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request, sys; sys.exit(0 if urllib.request.urlopen(os.environ['NEMO_GUARDRAILS_HEALTHCHECK_URL']).status == 200 else 1)"

ENTRYPOINT ["nemoguardrails"]
CMD ["server", "--verbose", "--config=/config"]
