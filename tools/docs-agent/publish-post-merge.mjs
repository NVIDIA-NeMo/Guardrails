#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

import {
  exactSha,
  fail,
  isPublicDocumentationPath,
  managedBranch,
  parseMaintainerLogins,
  patchSha256,
  readBoundedFile,
  readBoundedJson,
  requestedReviewers,
} from "./contract.mjs";
import { githubRequest } from "./github.mjs";

const SIGN_OFF = "Signed-off-by: github-actions[bot] <41898282+github-actions[bot]@users.noreply.github.com>";
const GIT_ENV = {
  ...process.env,
  GIT_CONFIG_GLOBAL: "/dev/null",
  GIT_CONFIG_NOSYSTEM: "1",
  GIT_LFS_SKIP_SMUDGE: "1",
  GIT_TERMINAL_PROMPT: "0",
};
for (const name of ["DOCS_AGENT_API_KEY", "GH_TOKEN", "GITHUB_TOKEN", "NVIDIA_API_KEY", "OPENAI_API_KEY"]) {
  delete GIT_ENV[name];
}

function required(value, name) {
  return value || fail(`${name} is required`);
}

function git(repository, args, capture = true) {
  return String(
    execFileSync("git", ["-c", "core.hooksPath=/dev/null", "-C", repository, ...args], {
      encoding: "utf8",
      env: GIT_ENV,
      stdio: capture ? ["ignore", "pipe", "inherit"] : "inherit",
    }) ?? "",
  ).trim();
}

function validateMetadata(value, repository, patch) {
  if (!value || typeof value !== "object" || Array.isArray(value)) fail("metadata must be an object");
  if (
    value.version !== 1 ||
    value.kind !== "post-merge" ||
    value.repository !== repository ||
    value.patch_sha256 !== patchSha256(patch) ||
    !Number.isSafeInteger(value.source_pull_request) ||
    value.source_pull_request < 1 ||
    typeof value.source_author !== "string"
  ) {
    fail("Approved documentation metadata is invalid");
  }
  exactSha(value.base_sha, "metadata base SHA");
  exactSha(value.head_sha, "metadata head SHA");
  exactSha(value.merge_sha, "metadata merge SHA");
  return value;
}

function validateStagedPaths(repository) {
  const fields = git(repository, ["diff", "--cached", "--name-status", "--no-renames", "-z", "HEAD"])
    .split("\0")
    .filter(Boolean);
  if (fields.length % 2 !== 0 || fields.length > 400) fail("Patch has an invalid changed-path list");
  for (let index = 0; index < fields.length; index += 2) {
    if (!/^[AMD]$/u.test(fields[index]) || !isPublicDocumentationPath(fields[index + 1])) {
      fail(`Patch changes unsupported path: ${fields[index + 1] ?? ""}`);
    }
  }
  if (fields.length === 0) fail("Non-empty patch produced no staged documentation changes");
}

function pullBody(sourcePullRequest, mergeSha) {
  return `## Description

Fast-follow public documentation for development PR #${sourcePullRequest}. The short-lived documentation author and independent reviewer evaluated only that pull request's exact delta at merge commit \`${mergeSha}\`.

## Related Issue(s)

- Follow-up to development PR #${sourcePullRequest} and its linked issue.

## Verification

- The generated patch is restricted to public documentation paths.
- An independent short-lived documentation reviewer approved the exact patch.
- Required documentation checks run on this pull request.

## AI Assistance

- [ ] No AI tools were used.
- [x] AI tools were used; a human must review and be able to explain every change (tool: isolated Pi documentation author and reviewer).

## Checklist

- [x] The pull request is limited to documentation for one merged development PR.
- [x] The pull request title follows the project commit convention.
- [x] Generated changelog files were not updated.
- [ ] Maintainers and the development PR author reviewed the documentation.

${SIGN_OFF}`;
}

async function main() {
  const repositoryName = required(process.env.GITHUB_REPOSITORY, "GITHUB_REPOSITORY");
  const checkout = required(process.env.PUBLISH_CHECKOUT, "PUBLISH_CHECKOUT");
  const artifact = required(process.env.DOCS_AGENT_ARTIFACT_DIR, "DOCS_AGENT_ARTIFACT_DIR");
  const patch = readBoundedFile(path.join(artifact, "docs.patch"), 5_242_880, true);
  const metadata = validateMetadata(
    readBoundedJson(path.join(artifact, "metadata.json"), 16_384),
    repositoryName,
    patch,
  );
  const sourcePull = await githubRequest("GET", `/repos/${repositoryName}/pulls/${metadata.source_pull_request}`);
  if (
    !sourcePull.merged ||
    sourcePull.base?.ref !== "develop" ||
    sourcePull.merge_commit_sha !== metadata.merge_sha ||
    sourcePull.user?.login !== metadata.source_author
  ) {
    fail("Source pull request no longer matches the approved documentation artifact");
  }
  if (patch.length === 0) {
    console.log(`Development PR #${metadata.source_pull_request} requires no documentation pull request.`);
    return;
  }
  const branch = managedBranch(metadata.source_pull_request, metadata.merge_sha);
  const owner = repositoryName.split("/")[0];
  const existing = await githubRequest(
    "GET",
    `/repos/${repositoryName}/pulls?state=open&base=develop&head=${encodeURIComponent(`${owner}:${branch}`)}&per_page=10`,
  );
  if (Array.isArray(existing) && existing.length === 1) {
    const pull = existing[0];
    if (
      pull.user?.login !== "github-actions[bot]" ||
      pull.draft !== true ||
      pull.base?.ref !== "develop" ||
      pull.head?.ref !== branch ||
      pull.head?.repo?.full_name !== repositoryName
    ) {
      fail("An open pull request uses the managed branch but is not owned by this automation");
    }
    console.log(pull.html_url);
    return;
  }
  if (!Array.isArray(existing) || existing.length > 1) fail("Managed documentation pull request lookup is invalid");
  const maintainers = parseMaintainerLogins(process.env.DOCS_MAINTAINERS);
  const reviewers = requestedReviewers(maintainers, metadata.source_author);
  if (git(checkout, ["status", "--porcelain=v1", "--untracked-files=all"])) fail("Publisher checkout must be clean");
  const liveRef = await githubRequest("GET", `/repos/${repositoryName}/git/ref/heads/develop`);
  const liveSha = exactSha(liveRef.object?.sha, "live develop SHA");
  if (git(checkout, ["rev-parse", "HEAD"]) !== liveSha) fail("develop changed before documentation publication; rerun the workflow");
  const remoteBranch = git(checkout, ["ls-remote", "--heads", "origin", `refs/heads/${branch}`]);
  if (remoteBranch) fail(`Managed branch already exists without an open pull request: ${branch}`);
  git(checkout, ["switch", "--create", branch], false);
  execFileSync("git", ["-c", "core.hooksPath=/dev/null", "-C", checkout, "apply", "--binary", "--index", "--whitespace=nowarn", "-"], {
    env: GIT_ENV,
    input: patch,
    stdio: ["pipe", "inherit", "inherit"],
  });
  validateStagedPaths(checkout);
  git(checkout, ["config", "user.name", "github-actions[bot]"], false);
  git(checkout, ["config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com"], false);
  git(checkout, ["commit", "--signoff", "--message", `docs: follow up PR #${metadata.source_pull_request}`], false);
  git(checkout, ["push", "origin", `HEAD:refs/heads/${branch}`], false);
  const pull = await githubRequest("POST", `/repos/${repositoryName}/pulls`, {
    base: "develop",
    body: pullBody(metadata.source_pull_request, metadata.merge_sha),
    draft: true,
    head: branch,
    title: `docs: follow up PR #${metadata.source_pull_request}`,
  });
  await githubRequest("POST", `/repos/${repositoryName}/issues/${pull.number}/labels`, {
    labels: ["documentation"],
  });
  await githubRequest("POST", `/repos/${repositoryName}/pulls/${pull.number}/requested_reviewers`, {
    reviewers,
  });
  console.log(pull.html_url);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
