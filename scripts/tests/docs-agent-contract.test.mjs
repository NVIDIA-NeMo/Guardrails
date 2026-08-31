// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  isAuthorizedAssociation,
  isDocumentationRelatedPath,
  isPublicDocumentationPath,
  isReviewCommand,
  managedBranch,
  parseMaintainerLogins,
  requestedReviewers,
  validateQualityScore,
  validateReviewResult,
} from "../../tools/docs-agent/contract.mjs";
import { selectPostMerge } from "../../tools/docs-agent/select-post-merge.mjs";
import { manualReviewRequest } from "../../tools/docs-agent/select-review.mjs";
import { withCleanup } from "../../tools/docs-agent/agent.mjs";

const SHA_A = "a".repeat(40);
const SHA_B = "b".repeat(40);
const SHA_C = "c".repeat(40);

function quality(overrides = {}) {
  return {
    confidence: 0.9,
    decision: "pass",
    hard_gates: {
      broken_required_examples: 0,
      critical_factual_errors: 0,
      invalid_required_artifacts: 0,
      missing_mandatory_content: 0,
      security_compliance_violations: 0,
      unsupported_critical_claims: 0,
    },
    rationale: "The changed task is accurate and complete.",
    rubric: "task-guide-v0.1",
    scores: {
      clarity: 9,
      completeness: 18,
      source_grounding: 14,
      style_compliance: 5,
      task_success: 18,
      technical_accuracy: 28,
    },
    total: 92,
    ...overrides,
  };
}

test("separates writable public documentation from related review surfaces", () => {
  assert.equal(isPublicDocumentationPath("docs/getting-started/example.mdx"), true);
  assert.equal(isPublicDocumentationPath("fern/docs.yml"), true);
  assert.equal(isPublicDocumentationPath("fern/assets/logo.svg"), true);
  assert.equal(isPublicDocumentationPath("fern/fern.config.json"), false);
  assert.equal(isPublicDocumentationPath("docs/_build/result.html"), false);
  assert.equal(isDocumentationRelatedPath("scripts/run-fern-with-ref-sdk.mjs"), true);
  assert.equal(isDocumentationRelatedPath("scripts/fern-ref-sdk-hooks/post-checkout"), true);
  assert.equal(isDocumentationRelatedPath("nemoguardrails/rails/llm/config.py"), false);
});

test("accepts only an exact authorized review command", () => {
  assert.equal(isReviewCommand(" /review-doc\n"), true);
  assert.equal(isReviewCommand("/review-doc please"), false);
  assert.equal(isAuthorizedAssociation("MEMBER"), true);
  assert.equal(isAuthorizedAssociation("CONTRIBUTOR"), false);
  assert.equal(
    manualReviewRequest({
      comment: { author_association: "OWNER", body: "/review-doc" },
      issue: { pull_request: { url: "https://api.github.com/example" } },
    }),
    true,
  );
});

test("builds one managed branch per source pull request and merge", () => {
  assert.equal(managedBranch(2350, SHA_C), `automation/post-merge-docs-pr-2350-${SHA_C.slice(0, 12)}`);
});

test("requests configured maintainers and the source author without duplicates", () => {
  const maintainers = parseMaintainerLogins("maintainer-one, maintainer-two,maintainer-one");
  assert.deepEqual(requestedReviewers(maintainers, "contributor"), [
    "maintainer-one",
    "maintainer-two",
    "contributor",
  ]);
  assert.deepEqual(requestedReviewers(maintainers, "dependabot[bot]"), maintainers);
});

test("validates 100-point score arithmetic and hard gates", () => {
  assert.equal(validateQualityScore(quality()).total, 92);
  assert.throws(() => validateQualityScore(quality({ total: 91 })), /score sum/u);
  assert.throws(() => validateQualityScore(quality({ rubric: "__proto__" })), /unsupported rubric/u);
  const gated = quality({ decision: "pass" });
  gated.hard_gates.critical_factual_errors = 1;
  assert.throws(() => validateQualityScore(gated), /requires decision fail/u);
});

test("binds a review result to the exact pull request revision", () => {
  const result = {
    base_sha: SHA_A,
    findings: [],
    head_sha: SHA_B,
    pull_request: 25,
    quality: quality(),
    repository: "NVIDIA-NeMo/Guardrails",
    summary: "No concrete findings.",
    version: 1,
  };
  assert.equal(
    validateReviewResult(result, {
      baseSha: SHA_A,
      headSha: SHA_B,
      pullRequest: 25,
      repository: "NVIDIA-NeMo/Guardrails",
    }),
    result,
  );
  assert.throws(
    () =>
      validateReviewResult(result, {
        baseSha: SHA_A,
        headSha: SHA_C,
        pullRequest: 25,
        repository: "NVIDIA-NeMo/Guardrails",
      }),
    /does not match/u,
  );
});

test("selects only merged non-documentation development pull requests", () => {
  const event = {
    action: "closed",
    pull_request: {
      base: { ref: "develop", repo: { full_name: "NVIDIA-NeMo/Guardrails" }, sha: SHA_A },
      head: { ref: "feature", repo: { full_name: "contributor/Guardrails" }, sha: SHA_B },
      merge_commit_sha: SHA_C,
      merged: true,
      number: 50,
      state: "closed",
      user: { login: "contributor" },
    },
  };
  assert.equal(selectPostMerge(event, "NVIDIA-NeMo/Guardrails", ["nemoguardrails/config.py"]).automate, true);
  assert.equal(selectPostMerge(event, "NVIDIA-NeMo/Guardrails", ["docs/example.mdx"]).automate, false);
  const deletedFork = structuredClone(event);
  deletedFork.pull_request.head.repo = null;
  assert.equal(selectPostMerge(deletedFork, "NVIDIA-NeMo/Guardrails", ["nemoguardrails/config.py"]).automate, false);
});

test("runs cleanup after a failed short-lived agent operation", async () => {
  let cleaned = false;
  await assert.rejects(
    withCleanup(
      async () => {
        throw new Error("analysis failed");
      },
      async () => {
        cleaned = true;
      },
    ),
    /analysis failed/u,
  );
  assert.equal(cleaned, true);
});
