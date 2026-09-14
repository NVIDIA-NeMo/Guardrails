#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

import { exactSha, fail } from "./contract.mjs";

const REPOSITORY = /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/u;
const GIT_ENV = {
  GIT_CONFIG_GLOBAL: "/dev/null",
  GIT_CONFIG_NOSYSTEM: "1",
  GIT_LFS_SKIP_SMUDGE: "1",
  GIT_TERMINAL_PROMPT: "0",
  LANG: "C",
  LC_ALL: "C",
  PATH: process.env.PATH,
};

function required(value, name) {
  return value || fail(`${name} is required`);
}

function repository(value, name) {
  return REPOSITORY.test(value ?? "") ? value : fail(`${name} is invalid`);
}

function git(directory, args) {
  execFileSync("git", ["-c", "core.hooksPath=/dev/null", "-C", directory, ...args], {
    env: GIT_ENV,
    stdio: "inherit",
  });
}

const destination = path.resolve(required(process.env.TARGET_CHECKOUT, "TARGET_CHECKOUT"));
const baseRepository = repository(process.env.BASE_REPOSITORY, "BASE_REPOSITORY");
const headRepository = repository(process.env.HEAD_REPOSITORY, "HEAD_REPOSITORY");
const baseSha = exactSha(process.env.BASE_SHA, "BASE_SHA");
const headSha = exactSha(process.env.HEAD_SHA, "HEAD_SHA");
const checkoutSha = exactSha(process.env.CHECKOUT_SHA, "CHECKOUT_SHA");

fs.rmSync(destination, { force: true, recursive: true });
fs.mkdirSync(destination, { recursive: true });
git(destination, ["init"]);
git(destination, ["remote", "add", "base", `https://github.com/${baseRepository}.git`]);
git(destination, ["remote", "add", "head", `https://github.com/${headRepository}.git`]);
git(destination, ["fetch", "--no-tags", "--depth=1", "base", baseSha]);
git(destination, ["fetch", "--no-tags", "--depth=1", "head", headSha]);
if (checkoutSha !== baseSha && checkoutSha !== headSha) {
  git(destination, ["fetch", "--no-tags", "--depth=1", "base", checkoutSha]);
}
git(destination, ["update-ref", "refs/heads/docs-agent-base", baseSha]);
git(destination, ["update-ref", "refs/heads/docs-agent-head", headSha]);
git(destination, ["update-ref", "refs/heads/docs-agent-checkout", checkoutSha]);
git(destination, ["checkout", "--detach", checkoutSha]);
const observed = execFileSync("git", ["-C", destination, "rev-parse", "HEAD"], {
  encoding: "utf8",
  env: GIT_ENV,
}).trim();
if (observed !== checkoutSha) fail("Prepared checkout has the wrong revision");
