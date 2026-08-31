// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createHash } from "node:crypto";
import fs from "node:fs";

const SHA = /^[0-9a-f]{40}$/u;
const LOGIN = /^(?!-)[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$/u;
const SAFE_PATH = /^[A-Za-z0-9._/-]+$/u;

export const RUBRICS = {
  "api-reference-v0.1": {
    allowConditionalPass: false,
    maxima: {
      clarity: 5,
      completeness: 25,
      source_grounding: 20,
      style_compliance: 5,
      task_success: 10,
      technical_accuracy: 35,
    },
    threshold: 92,
  },
  "concept-guide-v0.1": {
    allowConditionalPass: false,
    maxima: {
      clarity: 15,
      completeness: 20,
      source_grounding: 20,
      style_compliance: 5,
      task_success: 10,
      technical_accuracy: 30,
    },
    threshold: 90,
  },
  "task-guide-v0.1": {
    allowConditionalPass: false,
    maxima: {
      clarity: 10,
      completeness: 20,
      source_grounding: 15,
      style_compliance: 5,
      task_success: 20,
      technical_accuracy: 30,
    },
    threshold: 90,
  },
};

export const HARD_GATES = [
  "broken_required_examples",
  "critical_factual_errors",
  "invalid_required_artifacts",
  "missing_mandatory_content",
  "security_compliance_violations",
  "unsupported_critical_claims",
];

export function fail(message) {
  throw new Error(message);
}

export function exactSha(value, name = "SHA") {
  return SHA.test(value ?? "") ? value : fail(`${name} must be a lowercase 40-character Git SHA`);
}

function safePath(file) {
  return (
    typeof file === "string" &&
    file.length > 0 &&
    Buffer.byteLength(file) <= 512 &&
    SAFE_PATH.test(file) &&
    !file.includes("//") &&
    !file.endsWith("/") &&
    !/(?:^|\/)(?:\.{1,2}|\.git|\.gitattributes|\.gitmodules|node_modules)(?:\/|$)/u.test(file)
  );
}

export function isPublicDocumentationPath(file) {
  return (
    safePath(file) &&
    file !== "docs/_build" &&
    !file.startsWith("docs/_build/") &&
    (/^docs\//u.test(file) || file === "fern/docs.yml" || /^fern\/assets\//u.test(file))
  );
}

export function isDocumentationRelatedPath(file) {
  return (
    isPublicDocumentationPath(file) ||
    file === "AGENTS.md" ||
    file === "AI_POLICY.md" ||
    file === "CONTRIBUTING.md" ||
    file === "Makefile" ||
    file === "package.json" ||
    file === "package-lock.json" ||
    file === ".github/workflows/docs-build.yaml" ||
    /^fern\//u.test(file) ||
    /^scripts\/(?:[^/]*fern[^/]*|watch-fern-preview[.]mjs|tests\/)/u.test(file)
  );
}

export function isReviewCommand(value) {
  return typeof value === "string" && value.trim() === "/review-doc";
}

export function isAuthorizedAssociation(value) {
  return ["COLLABORATOR", "MEMBER", "OWNER"].includes(value ?? "");
}

export function parseMaintainerLogins(value) {
  const logins = [...new Set((value ?? "").split(",").map((item) => item.trim()).filter(Boolean))];
  if (logins.length === 0) fail("DOCS_MAINTAINERS must name at least one GitHub maintainer");
  if (logins.length > 14) fail("DOCS_MAINTAINERS cannot contain more than 14 maintainers");
  for (const login of logins) {
    if (!LOGIN.test(login) || login.endsWith("[bot]")) fail(`Invalid maintainer login: ${login}`);
  }
  return logins;
}

export function requestedReviewers(maintainers, sourceAuthor) {
  const author = sourceAuthor?.trim() ?? "";
  if (author.endsWith("[bot]")) return maintainers;
  if (author && !LOGIN.test(author)) fail("Source pull request author is invalid");
  return [...new Set([...maintainers, ...(author ? [author] : [])])];
}

export function managedBranch(sourcePullRequest, mergeSha) {
  const number = Number(sourcePullRequest);
  if (!Number.isSafeInteger(number) || number < 1) fail("Source pull request number is invalid");
  return `automation/post-merge-docs-pr-${number}-${exactSha(mergeSha, "merge SHA").slice(0, 12)}`;
}

export function readBoundedFile(file, maximum, allowEmpty = false) {
  const descriptor = fs.openSync(file, fs.constants.O_RDONLY | fs.constants.O_NOFOLLOW);
  try {
    const stat = fs.fstatSync(descriptor);
    if (!stat.isFile() || stat.size > maximum || (!allowEmpty && stat.size === 0)) {
      fail(`${file} must be a bounded regular file`);
    }
    const content = fs.readFileSync(descriptor);
    if (content.length !== stat.size) fail(`${file} changed while read`);
    return content;
  } finally {
    fs.closeSync(descriptor);
  }
}

export function readBoundedJson(file, maximum) {
  let value;
  try {
    value = JSON.parse(readBoundedFile(file, maximum).toString("utf8"));
  } catch (error) {
    fail(`${file} must contain valid JSON: ${error instanceof Error ? error.message : String(error)}`);
  }
  return value;
}

function isNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

export function validateQualityScore(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) fail("quality score must be an object");
  if (typeof value.rubric !== "string" || !Object.hasOwn(RUBRICS, value.rubric)) {
    fail(`unsupported rubric: ${String(value.rubric)}`);
  }
  const rubric = RUBRICS[value.rubric];
  const scoreKeys = Object.keys(rubric.maxima).sort();
  if (!value.scores || typeof value.scores !== "object" || Array.isArray(value.scores)) fail("scores must be an object");
  if (Object.keys(value.scores).sort().join() !== scoreKeys.join()) fail("scores contain missing or unsupported dimensions");
  let total = 0;
  for (const [dimension, maximum] of Object.entries(rubric.maxima)) {
    const score = value.scores[dimension];
    if (!isNumber(score) || score < 0 || score > maximum) fail(`score ${dimension} must be between 0 and ${maximum}`);
    total += score;
  }
  if (!isNumber(value.total) || value.total < 0 || value.total > 100 || Math.abs(value.total - total) > 1e-9) {
    fail("total must be the score sum between 0 and 100");
  }
  if (!value.hard_gates || typeof value.hard_gates !== "object" || Array.isArray(value.hard_gates)) fail("hard_gates must be an object");
  if (Object.keys(value.hard_gates).sort().join() !== [...HARD_GATES].sort().join()) fail("hard_gates contain missing or unsupported fields");
  const gateFailed = HARD_GATES.some((name) => {
    const count = value.hard_gates[name];
    if (!Number.isSafeInteger(count) || count < 0) fail(`hard gate ${name} must be a nonnegative integer`);
    return count > 0;
  });
  if (!["fail", "human-review", "pass"].includes(value.decision)) fail("decision must be fail, human-review, or pass");
  if (gateFailed && value.decision !== "fail") fail("a nonzero hard gate requires decision fail");
  if (!gateFailed && value.decision === "pass" && value.total < rubric.threshold) fail(`pass requires a total of at least ${rubric.threshold}`);
  if (!isNumber(value.confidence) || value.confidence < 0 || value.confidence > 1) fail("confidence must be between 0 and 1");
  if (typeof value.rationale !== "string" || !value.rationale.trim()) fail("rationale must be non-empty");
  return value;
}

export function validateReviewResult(value, expected) {
  if (!value || typeof value !== "object" || Array.isArray(value)) fail("review result must be an object");
  if (value.version !== 1 || value.repository !== expected.repository || value.pull_request !== expected.pullRequest || value.base_sha !== expected.baseSha || value.head_sha !== expected.headSha) {
    fail("review result does not match the requested pull request revision");
  }
  validateQualityScore(value.quality);
  if (typeof value.summary !== "string" || !value.summary.trim() || value.summary.length > 4000) fail("review summary is invalid");
  if (!Array.isArray(value.findings) || value.findings.length > 100) fail("findings must be an array with at most 100 items");
  for (const finding of value.findings) {
    if (!finding || typeof finding !== "object" || Array.isArray(finding)) fail("finding must be an object");
    if (!["blocker", "warning", "suggestion"].includes(finding.severity)) fail("finding severity is invalid");
    if (!safePath(finding.file) || !Number.isSafeInteger(finding.line) || finding.line < 1) fail("finding location is invalid");
    for (const name of ["title", "impact", "recommendation", "evidence"]) {
      if (typeof finding[name] !== "string" || !finding[name].trim() || finding[name].length > 4000) fail(`finding ${name} is invalid`);
    }
  }
  return value;
}

export function patchSha256(patch) {
  return createHash("sha256").update(patch).digest("hex");
}
