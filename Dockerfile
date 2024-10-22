# syntax=docker/dockerfile:experimental

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

FROM python:3.10

# Install git
RUN apt-get update && apt-get install -y git

# Install gcc/g++ for annoy
RUN apt-get install -y gcc g++

# Copy and install NeMo Guardrails
WORKDIR /nemoguardrails
COPY . /nemoguardrails
# pip cannot resolve dependencies if eval is part of all
RUN pip install --no-cache-dir -e .[all]
RUN pip install --no-cache-dir -e .[eval]

# Make port 8000 available to the world outside this container
EXPOSE 8000

# We copy the example bot configurations
WORKDIR /config
COPY ./examples/bots /config

# Run app.py when the container launches
WORKDIR /nemoguardrails

# Download the `all-MiniLM-L6-v2` model
RUN python -c "from fastembed.embedding import FlagEmbedding; FlagEmbedding('sentence-transformers/all-MiniLM-L6-v2');"

# Run this so that everything is initialized
RUN nemoguardrails --help

ENTRYPOINT ["/usr/local/bin/nemoguardrails"]
CMD ["server", "--verbose", "--config=/config"]
