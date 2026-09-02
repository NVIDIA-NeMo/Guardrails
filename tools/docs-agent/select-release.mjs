#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import fs from "node:fs";
import { pathToFileURL } from "node:url";

import { exactSha, exactTimestamp, fail, releaseVersionFromPull } from "./contract.mjs";
import { pullRequestFiles, workflowOutput } from "./github.mjs";

const RELEASE_PATHS = ["CHANGELOG.md", "README.md", "pyproject.toml", "uv.lock"];

export function selectRelease(event, repository, files) {
  const pull = event.pull_request;
  const version = releaseVersionFromPull(pull, repository);
  if (
    event.action !== "closed" ||
    !pull?.merged ||
    pull.state !== "closed" ||
    !Number.isSafeInteger(pull.number) ||
    pull.number < 1 ||
    !version
  ) {
    return { automate: false };
  }
  const labels = new Set((pull.labels ?? []).map((label) => label?.name));
  if (!labels.has("release") || !labels.has("automated")) {
    fail("The merged release preparation pull request is missing required release automation labels");
  }
  if (files.length !== RELEASE_PATHS.length || !RELEASE_PATHS.every((file) => files.includes(file))) {
    fail("The merged release preparation pull request does not have the expected generated file set");
  }
  return {
    automate: true,
    baseRepository: pull.base.repo.full_name,
    baseSha: exactSha(pull.base.sha, "release base SHA"),
    headRepository: pull.head.repo.full_name,
    headSha: exactSha(pull.head.sha, "release head SHA"),
    mergeSha: exactSha(pull.merge_commit_sha, "release merge SHA"),
    mergedAt: exactTimestamp(pull.merged_at, "release merge time"),
    pullRequest: pull.number,
    sourceAuthor: pull.user.login,
    version,
  };
}

async function main() {
  const repository = process.env.GITHUB_REPOSITORY || fail("GITHUB_REPOSITORY is required");
  const event = JSON.parse(fs.readFileSync(process.env.GITHUB_EVENT_PATH || fail("GITHUB_EVENT_PATH is required"), "utf8"));
  const pullRequest = event.pull_request?.number;
  const files = Number.isSafeInteger(pullRequest) ? await pullRequestFiles(repository, pullRequest) : [];
  const selected = selectRelease(event, repository, files);
  workflowOutput("automate", selected.automate);
  if (!selected.automate) return;
  workflowOutput("base_repository", selected.baseRepository);
  workflowOutput("base_sha", selected.baseSha);
  workflowOutput("head_repository", selected.headRepository);
  workflowOutput("head_sha", selected.headSha);
  workflowOutput("merge_sha", selected.mergeSha);
  workflowOutput("merged_at", selected.mergedAt);
  workflowOutput("pull_request", selected.pullRequest);
  workflowOutput("source_author", selected.sourceAuthor);
  workflowOutput("version", selected.version);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(1);
  });
}
