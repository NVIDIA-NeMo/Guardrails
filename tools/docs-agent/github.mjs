// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import fs from "node:fs";

import { fail } from "./contract.mjs";

export async function githubRequest(method, apiPath, body, env = process.env) {
  const token = env.GITHUB_TOKEN ?? env.GH_TOKEN;
  if (!token) fail("GITHUB_TOKEN or GH_TOKEN is required");
  if (!apiPath.startsWith("/")) fail("GitHub API path must be absolute");
  const response = await fetch(`https://api.github.com${apiPath}`, {
    body: body === undefined ? undefined : JSON.stringify(body),
    headers: {
      Accept: "application/vnd.github+json",
      Authorization: `Bearer ${token}`,
      "User-Agent": "guardrails-docs-agent",
      "X-GitHub-Api-Version": "2022-11-28",
    },
    method,
  });
  const text = await response.text();
  let value;
  try {
    value = text ? JSON.parse(text) : undefined;
  } catch {
    fail(`GitHub returned non-JSON content for ${method} ${apiPath}`);
  }
  if (!response.ok) fail(`GitHub ${method} ${apiPath} failed with ${response.status}: ${text.slice(0, 1000)}`);
  return value;
}

export async function pullRequestFiles(repository, pullRequest, env = process.env) {
  const files = [];
  for (let page = 1; page <= 30; page += 1) {
    const values = await githubRequest(
      "GET",
      `/repos/${repository}/pulls/${pullRequest}/files?per_page=100&page=${page}`,
      undefined,
      env,
    );
    if (!Array.isArray(values)) fail("GitHub returned an invalid pull request file list");
    files.push(...values.map((value) => value.filename));
    if (values.length < 100) return files;
  }
  return fail("Pull request file pagination exceeded 30 pages");
}

export function workflowOutput(name, value, env = process.env) {
  const file = env.GITHUB_OUTPUT;
  if (!file) fail("GITHUB_OUTPUT is required");
  const text = String(value);
  if (!/^[a-z_][a-z0-9_]*$/u.test(name) || /[\r\n]/u.test(text)) fail("Unsafe workflow output");
  fs.appendFileSync(file, `${name}=${text}\n`);
}
