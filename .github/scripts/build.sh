#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define variables for paths
PACKAGE_DIR="nemoguardrails"
CHAT_UI_SRC="chat-ui"
EXAMPLES_SRC="examples"
CHAT_UI_DST="$PACKAGE_DIR/chat-ui"
EXAMPLES_DST="$PACKAGE_DIR/examples"

# Copy the directories into the package directory
cp -r "$CHAT_UI_SRC" "$CHAT_UI_DST"
cp -r "$EXAMPLES_SRC" "$EXAMPLES_DST"

# Build the wheel using Poetry
poetry build

# Remove the copied directories after building
rm -rf "$CHAT_UI_DST"
rm -rf "$EXAMPLES_DST"
