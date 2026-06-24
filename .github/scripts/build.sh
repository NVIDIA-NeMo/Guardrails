#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define variables for paths
PACKAGE_DIR="nemoguardrails"
EXAMPLES_SRC="examples/bots"
EXAMPLES_DST="$PACKAGE_DIR/examples/bots"

# Copy the directories into the package directory
mkdir -p "$PACKAGE_DIR/examples"
cp -r "$EXAMPLES_SRC" "$EXAMPLES_DST"

# Build the wheel using Poetry
poetry build

# Remove the copied directories after building
rm -rf "$PACKAGE_DIR/examples"
