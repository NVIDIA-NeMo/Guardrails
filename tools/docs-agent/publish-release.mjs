#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { execFileSync, spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

import {
  exactSha,
  exactTimestamp,
  fail,
  isReleaseDocumentationPath,
  isReleaseSnapshotPath,
  managedReleaseBranch,
  parseMaintainerLogins,
  patchSha256,
  readBoundedFile,
  readBoundedJson,
  releaseSnapshotTag,
  releaseVersionFromPull,
  requestedReviewers,
  RELEASE_DOCUMENTATION_PATHS,
  RELEASE_SNAPSHOT_PATHS,
  stableVersion,
} from "./contract.mjs";
import { githubRequest } from "./github.mjs";

const BOT_NAME = "github-actions[bot]";
const BOT_EMAIL = "41898282+github-actions[bot]@users.noreply.github.com";
const SIGN_OFF = `Signed-off-by: ${BOT_NAME} <${BOT_EMAIL}>`;
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

function git(repository, args, capture = true, environment = GIT_ENV) {
  return String(
    execFileSync("git", ["-c", "core.hooksPath=/dev/null", "-C", repository, ...args], {
      encoding: "utf8",
      env: environment,
      stdio: capture ? ["ignore", "pipe", "inherit"] : "inherit",
    }) ?? "",
  ).trim();
}

function gitSucceeds(repository, args) {
  return spawnSync("git", ["-c", "core.hooksPath=/dev/null", "-C", repository, ...args], {
    env: GIT_ENV,
    stdio: "ignore",
  }).status === 0;
}

function validateMetadata(value, repository, patch, snapshotPatch) {
  if (!value || typeof value !== "object" || Array.isArray(value)) fail("metadata must be an object");
  if (
    value.version !== 1 ||
    value.kind !== "release" ||
    value.repository !== repository ||
    value.patch_sha256 !== patchSha256(patch) ||
    value.snapshot_patch_sha256 !== patchSha256(snapshotPatch) ||
    !Number.isSafeInteger(value.source_pull_request) ||
    value.source_pull_request < 1 ||
    value.source_author !== BOT_NAME
  ) {
    fail("Approved release documentation metadata is invalid");
  }
  exactSha(value.base_sha, "metadata base SHA");
  exactSha(value.head_sha, "metadata head SHA");
  exactSha(value.merge_sha, "metadata merge SHA");
  exactTimestamp(value.merged_at, "metadata merge time");
  stableVersion(value.release_version, "metadata release version");
  return value;
}

function stagedPaths(repository, allowedPath, requiredPaths) {
  const fields = git(repository, ["diff", "--cached", "--name-status", "--no-renames", "-z", "HEAD"])
    .split("\0")
    .filter(Boolean);
  if (fields.length !== requiredPaths.length * 2) fail("Patch does not have the exact required changed-path count");
  const changed = [];
  for (let index = 0; index < fields.length; index += 2) {
    if (!/^[AM]$/u.test(fields[index]) || !allowedPath(fields[index + 1])) {
      fail(`Patch changes unsupported path: ${fields[index + 1] ?? ""}`);
    }
    changed.push(fields[index + 1]);
  }
  if (!requiredPaths.every((file) => changed.includes(file))) fail("Patch is missing a required release documentation path");
}

function applyPatch(repository, patch) {
  execFileSync("git", ["-c", "core.hooksPath=/dev/null", "-C", repository, "apply", "--binary", "--index", "--whitespace=nowarn", "-"], {
    env: GIT_ENV,
    input: patch,
    stdio: ["pipe", "inherit", "inherit"],
  });
}

function remoteTagCommit(repository, tag) {
  const output = git(repository, ["ls-remote", "--tags", "origin", `refs/tags/${tag}`, `refs/tags/${tag}^{}`]);
  if (!output) return null;
  const refs = new Map(output.split(/\r?\n/u).filter(Boolean).map((line) => line.split(/\s+/u, 2).reverse()));
  return refs.get(`refs/tags/${tag}^{}`) ?? refs.get(`refs/tags/${tag}`) ?? fail("Remote snapshot tag is invalid");
}

function pullBody(metadata, snapshotTag) {
  return `## Description

- Publish v${metadata.release_version} release notes derived from the generated changelog and merged release tree.
- Register the immutable \`${snapshotTag}\` source in the Fern version switcher.
- Update the documentation versioning example for v${metadata.release_version}.

Areas for careful review:

- Confirm that every breaking change is represented and that feature and fix summaries are at the right level of detail.
- Confirm that documentation links and the versioned Previous Releases route resolve after publication.

## Related Issue(s)

- Release-documentation follow-up for automated release PR #${metadata.source_pull_request}.

## Verification

- The generated patch is restricted to the three release-documentation paths.
- An independent short-lived reviewer approved the exact release candidate.
- The immutable snapshot commit is based on release merge \`${metadata.merge_sha}\`.
- Required documentation checks run on this pull request.

## AI Assistance

- [ ] No AI tools were used.
- [x] AI tools were used; a human must review and be able to explain every change (tool: isolated Pi release-documentation author and reviewer).

## Checklist

- [x] The pull request title follows the project commit convention.
- [x] Generated changelog files were not updated.
- [ ] Maintainers reviewed the release notes and immutable snapshot.

${SIGN_OFF}`;
}

function validateExistingPull(pull, branch, repository) {
  if (
    pull.user?.login !== BOT_NAME ||
    pull.draft !== true ||
    pull.base?.ref !== "develop" ||
    pull.head?.ref !== branch ||
    pull.head?.repo?.full_name !== repository
  ) {
    fail("An open pull request uses the managed release-documentation branch but is not owned by this automation");
  }
}

async function main() {
  const repositoryName = required(process.env.GITHUB_REPOSITORY, "GITHUB_REPOSITORY");
  const checkout = required(process.env.PUBLISH_CHECKOUT, "PUBLISH_CHECKOUT");
  const artifact = required(process.env.DOCS_AGENT_ARTIFACT_DIR, "DOCS_AGENT_ARTIFACT_DIR");
  const patch = readBoundedFile(path.join(artifact, "docs.patch"), 5_242_880);
  const snapshotPatch = readBoundedFile(path.join(artifact, "snapshot.patch"), 5_242_880);
  const metadata = validateMetadata(
    readBoundedJson(path.join(artifact, "metadata.json"), 16_384),
    repositoryName,
    patch,
    snapshotPatch,
  );
  const sourcePull = await githubRequest("GET", `/repos/${repositoryName}/pulls/${metadata.source_pull_request}`);
  if (
    !sourcePull.merged ||
    sourcePull.merge_commit_sha !== metadata.merge_sha ||
    sourcePull.head?.sha !== metadata.head_sha ||
    sourcePull.merged_at !== metadata.merged_at ||
    releaseVersionFromPull(sourcePull, repositoryName) !== metadata.release_version
  ) {
    fail("Source release pull request no longer matches the approved documentation artifact");
  }

  const branch = managedReleaseBranch(metadata.release_version, metadata.merge_sha);
  const tag = releaseSnapshotTag(metadata.release_version);
  const owner = repositoryName.split("/")[0];
  const existing = await githubRequest(
    "GET",
    `/repos/${repositoryName}/pulls?state=open&base=develop&head=${encodeURIComponent(`${owner}:${branch}`)}&per_page=10`,
  );
  if (!Array.isArray(existing) || existing.length > 1) fail("Managed release-documentation pull request lookup is invalid");
  if (existing.length === 1) validateExistingPull(existing[0], branch, repositoryName);

  const maintainers = parseMaintainerLogins(process.env.DOCS_MAINTAINERS);
  const reviewers = requestedReviewers(maintainers, metadata.source_author);
  if (git(checkout, ["status", "--porcelain=v1", "--untracked-files=all"])) fail("Publisher checkout must be clean");
  const liveRef = await githubRequest("GET", `/repos/${repositoryName}/git/ref/heads/develop`);
  const liveSha = exactSha(liveRef.object?.sha, "live develop SHA");
  if (git(checkout, ["rev-parse", "HEAD"]) !== liveSha) fail("develop changed before release documentation publication; rerun the workflow");
  if (!gitSucceeds(checkout, ["merge-base", "--is-ancestor", metadata.merge_sha, "HEAD"])) {
    fail("The release merge is not an ancestor of current develop");
  }

  const remoteBranch = git(checkout, ["ls-remote", "--heads", "origin", `refs/heads/${branch}`]);
  if (existing.length === 0 && remoteBranch) fail(`Managed branch already exists without an open pull request: ${branch}`);
  if (existing.length === 0) {
    git(checkout, ["switch", "--create", branch], false);
    applyPatch(checkout, patch);
    stagedPaths(checkout, isReleaseDocumentationPath, RELEASE_DOCUMENTATION_PATHS);
    git(checkout, ["config", "user.name", BOT_NAME], false);
    git(checkout, ["config", "user.email", BOT_EMAIL], false);
    git(checkout, ["commit", "--signoff", "--message", `docs: publish v${metadata.release_version} release notes and snapshot`], false);
  }

  const temporaryRoot = fs.mkdtempSync(path.join(required(process.env.RUNNER_TEMP, "RUNNER_TEMP"), "release-snapshot-"));
  const snapshotCheckout = path.join(temporaryRoot, "repository");
  const datedEnvironment = {
    ...GIT_ENV,
    GIT_AUTHOR_DATE: metadata.merged_at,
    GIT_COMMITTER_DATE: metadata.merged_at,
  };
  let snapshotCommit;
  try {
    git(checkout, ["worktree", "add", "--detach", snapshotCheckout, metadata.merge_sha], false);
    applyPatch(snapshotCheckout, snapshotPatch);
    stagedPaths(snapshotCheckout, isReleaseSnapshotPath, RELEASE_SNAPSHOT_PATHS);
    git(snapshotCheckout, ["config", "user.name", BOT_NAME], false);
    git(snapshotCheckout, ["config", "user.email", BOT_EMAIL], false);
    git(snapshotCheckout, ["commit", "--signoff", "--message", `docs: create v${metadata.release_version} Fern snapshot`], false, datedEnvironment);
    snapshotCommit = exactSha(git(snapshotCheckout, ["rev-parse", "HEAD"]), "snapshot commit");
    const existingTagCommit = remoteTagCommit(checkout, tag);
    if (existingTagCommit && existingTagCommit !== snapshotCommit) {
      fail(`Immutable snapshot tag ${tag} already references a different commit`);
    }
    if (!existingTagCommit && existing.length === 1) fail("Managed pull request exists without its immutable snapshot tag");
    if (!existingTagCommit) {
      git(snapshotCheckout, ["tag", "--annotate", tag, "--message", `Fern docs snapshot for v${metadata.release_version}`], false, datedEnvironment);
    }
    if (existing.length === 0) {
      const refs = [`HEAD:refs/heads/${branch}`];
      if (!existingTagCommit) refs.push(`refs/tags/${tag}:refs/tags/${tag}`);
      git(checkout, ["push", ...(existingTagCommit ? [] : ["--atomic"]), "origin", ...refs], false);
    }
    if (!existingTagCommit) {
      if (remoteTagCommit(checkout, tag) !== snapshotCommit) fail("Published snapshot tag does not match the approved commit");
    }
  } finally {
    if (fs.existsSync(snapshotCheckout)) git(checkout, ["worktree", "remove", "--force", snapshotCheckout], false);
    fs.rmSync(temporaryRoot, { force: true, recursive: true });
  }

  if (existing.length === 1) {
    console.log(existing[0].html_url);
    return;
  }
  const pull = await githubRequest("POST", `/repos/${repositoryName}/pulls`, {
    base: "develop",
    body: pullBody(metadata, tag),
    draft: true,
    head: branch,
    title: `docs: publish v${metadata.release_version} release notes and Fern snapshot`,
  });
  await githubRequest("POST", `/repos/${repositoryName}/issues/${pull.number}/labels`, {
    labels: ["documentation", "release", "automated"],
  });
  await githubRequest("POST", `/repos/${repositoryName}/pulls/${pull.number}/requested_reviewers`, {
    reviewers,
  });
  console.log(pull.html_url);
}

main().catch((_error) => {
  console.error("Release documentation publication failed. Review preceding trusted command output.");
  process.exit(1);
});
