#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly VERSION="0.0.106"
readonly RELEASE="v${VERSION}"
readonly REPOSITORY="NVIDIA/OpenShell"
readonly TARGET_DIR="${XDG_BIN_HOME:-${HOME}/.local/bin}"

if [ "$(uname -s)" != "Linux" ] || [ "$(uname -m)" != "x86_64" ]; then
  echo "The documentation agent workflow supports only the GitHub ubuntu x86_64 runner." >&2
  exit 1
fi

temporary_directory="$(mktemp -d)"
trap 'rm -rf "$temporary_directory"' EXIT

download_and_verify() {
  local asset="$1"
  local expected="$2"
  curl --fail --location --silent --show-error \
    "https://github.com/${REPOSITORY}/releases/download/${RELEASE}/${asset}" \
    --output "${temporary_directory}/${asset}"
  printf '%s  %s\n' "$expected" "${temporary_directory}/${asset}" | sha256sum --check --status
}

download_and_verify \
  "openshell-x86_64-unknown-linux-musl.tar.gz" \
  "d1a885a91b3e5aaa006c36aca95dc78bed0638c1ba1a79b55f1da93211b8a0a0"
download_and_verify \
  "openshell-gateway-x86_64-unknown-linux-gnu.tar.gz" \
  "b7760cb752a4363c2f21d32298dd0c683dc438f6edfd16c2e4242bc0baefbb7c"
download_and_verify \
  "openshell-sandbox-x86_64-unknown-linux-gnu.tar.gz" \
  "559b8aaad3a8eeab45c511e7de531d9baa98a311282dcb0c2c5f38cc2d4ca355"

for mapping in \
  "openshell-x86_64-unknown-linux-musl.tar.gz:openshell" \
  "openshell-gateway-x86_64-unknown-linux-gnu.tar.gz:openshell-gateway" \
  "openshell-sandbox-x86_64-unknown-linux-gnu.tar.gz:openshell-sandbox"; do
  archive="${mapping%%:*}"
  member="${mapping##*:}"
  if [ "$(tar -tzf "${temporary_directory}/${archive}")" != "$member" ]; then
    echo "Unexpected archive contents in ${archive}." >&2
    exit 1
  fi
  tar -xzf "${temporary_directory}/${archive}" -C "$temporary_directory"
done

mkdir -p "$TARGET_DIR"
install -m 0755 "$temporary_directory/openshell" "$TARGET_DIR/openshell"
install -m 0755 "$temporary_directory/openshell-gateway" "$TARGET_DIR/openshell-gateway"
install -m 0755 "$temporary_directory/openshell-sandbox" "$TARGET_DIR/openshell-sandbox"

if [ -n "${GITHUB_PATH:-}" ]; then
  printf '%s\n' "$TARGET_DIR" >> "$GITHUB_PATH"
fi

"$TARGET_DIR/openshell" --version
