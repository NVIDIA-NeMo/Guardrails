#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import fs from "node:fs";
import { pathToFileURL } from "node:url";

import { exactSha, fail, isDocumentationRelatedPath, releaseVersionFromPull } from "./contract.mjs";
import { pullRequestFiles, workflowOutput } from "./github.mjs";

export function selectPostMerge(event, repository, files) {
  const pull = event.pull_request;
  if (
    event.action !== "closed" ||
    !pull?.merged ||
    pull.state !== "closed" ||
    pull.base?.repo?.full_name !== repository ||
    pull.base?.ref !== "develop" ||
    typeof pull.head?.repo?.full_name !== "string" ||
    typeof pull.user?.login !== "string" ||
    !Number.isSafeInteger(pull.number) ||
    pull.number < 1 ||
    releaseVersionFromPull(pull, repository) !== null ||
    pull.head?.ref?.startsWith("automation/post-merge-docs-pr-")
  ) {
    return { automate: false };
  }
  if (files.length === 0 || files.every(isDocumentationRelatedPath)) return { automate: false };
  return {
    automate: true,
    baseRepository: pull.base.repo.full_name,
    baseSha: exactSha(pull.base.sha, "source base SHA"),
    headRepository: pull.head.repo.full_name,
    headSha: exactSha(pull.head.sha, "source head SHA"),
    mergeSha: exactSha(pull.merge_commit_sha, "source merge SHA"),
    pullRequest: pull.number,
    sourceAuthor: pull.user.login,
  };
}

async function main() {
  const repository = process.env.GITHUB_REPOSITORY || fail("GITHUB_REPOSITORY is required");
  const event = JSON.parse(fs.readFileSync(process.env.GITHUB_EVENT_PATH || fail("GITHUB_EVENT_PATH is required"), "utf8"));
  const pullRequest = event.pull_request?.number;
  const files = Number.isSafeInteger(pullRequest)
    ? await pullRequestFiles(repository, pullRequest)
    : [];
  const selected = selectPostMerge(event, repository, files);
  workflowOutput("automate", selected.automate);
  if (!selected.automate) return;
  workflowOutput("base_repository", selected.baseRepository);
  workflowOutput("base_sha", selected.baseSha);
  workflowOutput("head_repository", selected.headRepository);
  workflowOutput("head_sha", selected.headSha);
  workflowOutput("merge_sha", selected.mergeSha);
  workflowOutput("pull_request", selected.pullRequest);
  workflowOutput("source_author", selected.sourceAuthor);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(1);
  });
}
