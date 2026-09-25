// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import fs from "node:fs";
import path from "node:path";

import { fail } from "./contract.mjs";

const MAXIMUM_BODY_BYTES = 65_536;
const MAXIMUM_RESPONSE_BYTES = 8_388_608;
const API_PATHS = {
  GET: [
    /^\/repos\/NVIDIA-NeMo\/Guardrails\/git\/ref\/heads\/develop$/u,
    /^\/repos\/NVIDIA-NeMo\/Guardrails\/issues\/[1-9][0-9]*\/comments\?per_page=100&page=(?:[1-9]|1[0-9]|20)$/u,
    /^\/repos\/NVIDIA-NeMo\/Guardrails\/pulls\/[1-9][0-9]*$/u,
    /^\/repos\/NVIDIA-NeMo\/Guardrails\/pulls\/[1-9][0-9]*\/files\?per_page=100&page=(?:[1-9]|[12][0-9]|30)$/u,
    /^\/repos\/NVIDIA-NeMo\/Guardrails\/pulls\?state=open&base=develop&head=NVIDIA-NeMo%3Aautomation%2F(?:post-merge-docs-pr-[1-9][0-9]*|release-docs-v(?:0|[1-9][0-9]*)[.](?:0|[1-9][0-9]*)[.](?:0|[1-9][0-9]*))-[0-9a-f]{12}&per_page=10$/u,
  ],
  PATCH: [/^\/repos\/NVIDIA-NeMo\/Guardrails\/issues\/comments\/[1-9][0-9]*$/u],
  POST: [
    /^\/repos\/NVIDIA-NeMo\/Guardrails\/issues\/[1-9][0-9]*\/labels$/u,
    /^\/repos\/NVIDIA-NeMo\/Guardrails\/pulls$/u,
    /^\/repos\/NVIDIA-NeMo\/Guardrails\/pulls\/[1-9][0-9]*\/requested_reviewers$/u,
  ],
};

export function isAllowedGithubApiPath(method, apiPath) {
  return Object.hasOwn(API_PATHS, method) && API_PATHS[method].some((pattern) => pattern.test(apiPath));
}

export async function githubRequest(method, apiPath, body, env = process.env) {
  const token = env.GITHUB_TOKEN ?? env.GH_TOKEN;
  if (!token) fail("GITHUB_TOKEN or GH_TOKEN is required");
  if (!isAllowedGithubApiPath(method, apiPath)) fail(`GitHub API path is not allowed for ${method}`);
  const payload = body === undefined ? undefined : JSON.stringify(body);
  if (payload !== undefined && Buffer.byteLength(payload) > MAXIMUM_BODY_BYTES) fail("GitHub request body is too large");
  // lgtm[js/file-access-to-http] The endpoint is allowlisted above; bounded event and artifact fields are the intended request data.
  const response = await fetch(`https://api.github.com${apiPath}`, {
    body: payload,
    headers: {
      Accept: "application/vnd.github+json",
      Authorization: `Bearer ${token}`,
      "User-Agent": "guardrails-docs-agent",
      "X-GitHub-Api-Version": "2022-11-28",
    },
    method,
  });
  const declaredLength = Number(response.headers.get("content-length"));
  if (Number.isFinite(declaredLength) && declaredLength > MAXIMUM_RESPONSE_BYTES) fail("GitHub response is too large");
  const text = await response.text();
  if (Buffer.byteLength(text) > MAXIMUM_RESPONSE_BYTES) fail("GitHub response is too large");
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
  if (!/^[a-z_][a-z0-9_]*$/u.test(name) || Buffer.byteLength(text) > 1024 || /[\0\r\n]/u.test(text)) {
    fail("Unsafe workflow output");
  }
  const runnerTemp = path.resolve(env.RUNNER_TEMP || fail("RUNNER_TEMP is required"));
  const output = path.resolve(file);
  if (
    !output.startsWith(`${runnerTemp}${path.sep}`) ||
    !/^set_output_[A-Za-z0-9-]+$/u.test(path.basename(output))
  ) {
    fail("GITHUB_OUTPUT is outside the runner command directory");
  }
  const descriptor = fs.openSync(output, fs.constants.O_APPEND | fs.constants.O_NOFOLLOW | fs.constants.O_WRONLY);
  try {
    if (!fs.fstatSync(descriptor).isFile()) fail("GITHUB_OUTPUT is not a regular file");
    // lgtm[js/http-to-file-access] One-line bounded PR metadata is intentionally passed through GitHub's runner output file.
    fs.writeFileSync(descriptor, `${name}=${text}\n`);
  } finally {
    fs.closeSync(descriptor);
  }
}
