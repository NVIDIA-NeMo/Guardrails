#!/usr/bin/env node

import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { createFernRefSdkEnvironment } from "./fern-ref-sdk-environment.mjs";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const fernRoot = path.join(repoRoot, "fern");
const fernVersion = JSON.parse(readFileSync(path.join(fernRoot, "fern.config.json"), "utf8")).version;
const fernArguments = process.argv.slice(2);

if (fernArguments.length === 0) {
  console.error("Usage: run-fern-with-ref-sdk.mjs <fern arguments...>");
  process.exit(2);
}

const result = spawnSync("npx", ["--yes", `fern-api@${fernVersion}`, ...fernArguments], {
  cwd: fernRoot,
  env: createFernRefSdkEnvironment(repoRoot),
  stdio: "inherit",
});

if (result.error) {
  throw result.error;
}

process.exit(result.status ?? 1);
