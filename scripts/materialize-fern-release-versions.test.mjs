/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import assert from "node:assert/strict";
import test from "node:test";

import {
  parseReleaseTag,
  rewriteNavigation,
} from "./materialize-fern-release-versions.mjs";

test("accepts stable release tags starting at v0.24.0", () => {
  const release = parseReleaseTag("v0.24.0");
  assert.equal(release?.slug, "v0.24.0");
  assert.equal(release?.apiNamespace, "docs_v0_24_0");
  assert.equal(parseReleaseTag("v1.0.0")?.displayName, "v1.0.0");
});

test("rejects older, prerelease, and documentation-only tags", () => {
  assert.equal(parseReleaseTag("v0.23.0"), undefined);
  assert.equal(parseReleaseTag("v0.24.0-rc1"), undefined);
  assert.equal(parseReleaseTag("docs-v0.24.0"), undefined);
  assert.equal(parseReleaseTag("0.24.0"), undefined);
});

test("rewrites authored and generated references into the release snapshot", () => {
  const navigation = `
landing-page:
  page: Home
  path: index.mdx
navigation:
  - page: Overview
    path: about/overview.mdx
  - folder: _static/python-sdk-reference/guardrails-python-sdk
    title: Python SDK Reference
  - api: Guardrails API Server
    layout:
      - chatCompletions:
          - endpoint: POST /v1/chat/completions
`;

  const release = parseReleaseTag("v0.24.0");
  release.endpointIds = new Map([
    ["POST /v1/chat/completions", "createGuardrailsChatCompletion"],
  ]);
  const rewritten = rewriteNavigation(navigation, release);
  assert.match(
    rewritten,
    /path: \.\.\/generated\/release-versions\/v0\.24\.0\/docs\/index\.mdx/,
  );
  assert.match(
    rewritten,
    /path: \.\.\/generated\/release-versions\/v0\.24\.0\/docs\/about\/overview\.mdx/,
  );
  assert.match(
    rewritten,
    /folder: \.\.\/generated\/release-versions\/v0\.24\.0\/docs\/_static\/python-sdk-reference\/guardrails-python-sdk/,
  );
  assert.match(rewritten, /api: Guardrails API Server/);
  assert.match(rewritten, /docs_v0_24_0\.chatCompletions:/);
  assert.match(
    rewritten,
    /endpoint: docs_v0_24_0\.chatCompletions\.createGuardrailsChatCompletion/,
  );
});
