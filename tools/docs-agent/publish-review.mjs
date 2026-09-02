#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import path from "node:path";

import { exactSha, fail, readBoundedJson, validateReviewResult } from "./contract.mjs";
import { githubRequest } from "./github.mjs";

const MARKER = "<!-- guardrails-documentation-review -->";

function required(value, name) {
  return value || fail(`${name} is required`);
}

function safeText(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll("@", "&#64;");
}

function render(result, runUrl) {
  const quality = result.quality;
  const maximums = {
    "api-reference-v0.1": [35, 25, 10, 20, 5, 5],
    "concept-guide-v0.1": [30, 20, 10, 20, 15, 5],
    "task-guide-v0.1": [30, 20, 20, 15, 10, 5],
  }[quality.rubric];
  const dimensions = [
    "technical_accuracy",
    "completeness",
    "task_success",
    "source_grounding",
    "clarity",
    "style_compliance",
  ];
  const rows = dimensions.map(
    (name, index) => `| ${name.replaceAll("_", " ")} | ${quality.scores[name]} | ${maximums[index]} |`,
  );
  const findings = [];
  let length = 0;
  for (const finding of result.findings) {
    const block = `### ${safeText(finding.severity)}: ${safeText(finding.title)}\n\n` +
      `\`${safeText(finding.file)}:${finding.line}\` — ${safeText(finding.impact)}\n\n` +
      `Evidence: ${safeText(finding.evidence)}\n\n` +
      `Recommended action: ${safeText(finding.recommendation)}`;
    if (length + block.length > 45_000) break;
    findings.push(block);
    length += block.length;
  }
  return [
    MARKER,
    "## Documentation review",
    "",
    `**Score: ${quality.total}/100 — ${safeText(quality.decision)}**`,
    "",
    `Rubric: \`${safeText(quality.rubric)}\` · Confidence: ${quality.confidence}`,
    "",
    safeText(result.summary),
    "",
    "| Dimension | Score | Maximum |",
    "| --- | ---: | ---: |",
    ...rows,
    "",
    `Rationale: ${safeText(quality.rationale)}`,
    "",
    findings.length ? findings.join("\n\n") : "No concrete documentation findings were recorded.",
    "",
    `_Advisory review of head \`${result.head_sha}\`. [Workflow run](${runUrl})_`,
  ].join("\n");
}

async function main() {
  const repository = required(process.env.GITHUB_REPOSITORY, "GITHUB_REPOSITORY");
  const pullRequest = Number(required(process.env.PR_NUMBER, "PR_NUMBER"));
  if (!Number.isSafeInteger(pullRequest) || pullRequest < 1) fail("PR_NUMBER is invalid");
  const headSha = exactSha(process.env.PR_HEAD_SHA, "PR_HEAD_SHA");
  const artifact = required(process.env.DOCS_AGENT_ARTIFACT_DIR, "DOCS_AGENT_ARTIFACT_DIR");
  const result = readBoundedJson(path.join(artifact, "review.json"), 262_144);
  const pull = await githubRequest("GET", `/repos/${repository}/pulls/${pullRequest}`);
  if (pull.state !== "open" || pull.head?.sha !== headSha) fail("Pull request head changed before documentation review publication");
  validateReviewResult(result, {
    baseSha: exactSha(pull.base.sha, "current PR base SHA"),
    headSha,
    pullRequest,
    repository,
  });
  const runUrl = `https://github.com/${repository}/actions/runs/${required(process.env.GITHUB_RUN_ID, "GITHUB_RUN_ID")}`;
  const body = render(result, runUrl);
  const comments = [];
  for (let page = 1; page <= 20; page += 1) {
    const values = await githubRequest("GET", `/repos/${repository}/issues/${pullRequest}/comments?per_page=100&page=${page}`);
    if (!Array.isArray(values)) fail("GitHub returned an invalid comment list");
    comments.push(...values);
    if (values.length < 100) break;
  }
  const existing = comments.find(
    (comment) =>
      comment.user?.login === "github-actions[bot]" &&
      comment.user?.type === "Bot" &&
      comment.body?.startsWith(MARKER),
  );
  if (existing) await githubRequest("PATCH", `/repos/${repository}/issues/comments/${existing.id}`, { body });
  else await githubRequest("POST", `/repos/${repository}/issues/${pullRequest}/comments`, { body });
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
