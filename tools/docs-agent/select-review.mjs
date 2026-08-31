#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import fs from "node:fs";
import { pathToFileURL } from "node:url";

import {
  exactSha,
  fail,
  isAuthorizedAssociation,
  isDocumentationRelatedPath,
  isReviewCommand,
} from "./contract.mjs";
import { githubRequest, pullRequestFiles, workflowOutput } from "./github.mjs";

export function manualReviewRequest(event) {
  return Boolean(
    event.issue?.pull_request &&
      isReviewCommand(event.comment?.body) &&
      isAuthorizedAssociation(event.comment?.author_association),
  );
}

async function main() {
  const repository = process.env.GITHUB_REPOSITORY || fail("GITHUB_REPOSITORY is required");
  const eventName = process.env.GITHUB_EVENT_NAME || fail("GITHUB_EVENT_NAME is required");
  const event = JSON.parse(fs.readFileSync(process.env.GITHUB_EVENT_PATH || fail("GITHUB_EVENT_PATH is required"), "utf8"));
  let pull;
  if (eventName === "pull_request_target") {
    pull = event.pull_request;
  } else if (eventName === "issue_comment") {
    if (!manualReviewRequest(event)) {
      workflowOutput("eligible", false);
      return;
    }
    pull = await githubRequest("GET", `/repos/${repository}/pulls/${event.issue.number}`);
  } else if (eventName === "workflow_dispatch") {
    const number = Number(process.env.INPUT_PR_NUMBER);
    if (!Number.isSafeInteger(number) || number < 1) fail("pr_number is invalid");
    pull = await githubRequest("GET", `/repos/${repository}/pulls/${number}`);
  } else {
    fail(`Unsupported event: ${eventName}`);
  }
  if (
    pull?.state !== "open" ||
    pull.base?.repo?.full_name !== repository ||
    pull.base?.ref !== "develop" ||
    typeof pull.head?.repo?.full_name !== "string" ||
    !Number.isSafeInteger(pull.number)
  ) {
    workflowOutput("eligible", false);
    return;
  }
  const files = await pullRequestFiles(repository, pull.number);
  if (!files.some(isDocumentationRelatedPath)) {
    workflowOutput("eligible", false);
    return;
  }
  workflowOutput("eligible", true);
  workflowOutput("base_repository", pull.base.repo.full_name);
  workflowOutput("base_sha", exactSha(pull.base.sha, "PR base SHA"));
  workflowOutput("head_repository", pull.head.repo.full_name);
  workflowOutput("head_sha", exactSha(pull.head.sha, "PR head SHA"));
  workflowOutput("pull_request", pull.number);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(1);
  });
}
