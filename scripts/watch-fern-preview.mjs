// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { spawn, spawnSync } from "node:child_process";
import { existsSync, readdirSync, readFileSync, statSync, watch } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { createFernRefSdkEnvironment } from "./fern-ref-sdk-environment.mjs";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const fernRoot = path.join(repoRoot, "fern");
const watchRoots = ["docs", "fern", "nemoguardrails", "scripts"];
const watchFiles = ["pyproject.toml", "uv.lock"];
const ignoredDirectoryNames = new Set([".fern-cache", ".git", "_build", "node_modules", "__pycache__"]);
const debounceMs = 500;
const fernDocsInstance = "nvidia-nemo-guardrails.docs.buildwithfern.com/nemo/guardrails";

const branchName = currentBranchName();
let running = false;
let pending = false;
let referenceRevision = 0;
let generatedReferenceRevision = -1;
let debounceTimer;
let currentChild;
const watchers = new Map();

console.log(`Using Fern preview id: ${branchName}`);
console.log(`Watching: ${[...watchRoots, ...watchFiles].join(", ")}`);

for (const root of watchRoots) {
  watchDirectoryTree(path.join(repoRoot, root));
}

watchers.set(repoRoot, watch(repoRoot, (_eventType, filename) => {
  if (filename && watchFiles.includes(filename.toString())) {
    scheduleRun(path.join(repoRoot, filename.toString()));
  }
}));

runFernGenerate("initial run");

process.on("SIGINT", () => {
  closeWatchers();
  currentChild?.kill("SIGINT");
  process.exit(130);
});

process.on("SIGTERM", () => {
  closeWatchers();
  currentChild?.kill("SIGTERM");
  process.exit(143);
});

function currentBranchName() {
  const result = spawnSync("git", ["branch", "--show-current"], {
    cwd: repoRoot,
    encoding: "utf8",
  });
  const branch = result.stdout.trim();

  if (result.status !== 0 || branch.length === 0) {
    console.error("Could not determine the current Git branch name.");
    process.exit(1);
  }

  return branch;
}

function readFernVersion() {
  const fernConfig = JSON.parse(readFileSync(path.join(fernRoot, "fern.config.json"), "utf8"));
  const trimmedFernVersion = typeof fernConfig.version === "string" ? fernConfig.version.trim() : "";
  if (!/^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$/.test(trimmedFernVersion)) {
    throw new Error("fern.config.json must contain an exact semver version");
  }
  return trimmedFernVersion;
}

function watchDirectoryTree(directory) {
  if (!existsSync(directory) || watchers.has(directory)) {
    return;
  }

  try {
    watchers.set(
      directory,
      watch(directory, { persistent: true }, (_eventType, filename) => {
        let changedPath;
        if (filename) {
          changedPath = path.join(directory, filename.toString());
          if (shouldIgnorePath(changedPath) || !shouldTriggerRun(changedPath)) {
            return;
          }
          addWatcherForNewDirectory(changedPath);
        }
        scheduleRun(changedPath);
      }),
    );
  } catch (error) {
    if (!isIgnorableWatchError(error)) {
      throw error;
    }
    return;
  }

  let entries;
  try {
    entries = readdirSync(directory, { withFileTypes: true });
  } catch (error) {
    if (!isIgnorableWatchError(error)) {
      throw error;
    }
    return;
  }

  for (const entry of entries) {
    if (!entry.isDirectory()) {
      continue;
    }

    const childDirectory = path.join(directory, entry.name);
    if (!shouldIgnorePath(childDirectory)) {
      watchDirectoryTree(childDirectory);
    }
  }
}

function addWatcherForNewDirectory(changedPath) {
  if (!existsSync(changedPath)) {
    return;
  }

  try {
    const stats = statSync(changedPath);
    if (stats.isDirectory()) {
      watchDirectoryTree(changedPath);
    }
  } catch (error) {
    if (!isIgnorableWatchError(error)) {
      throw error;
    }
  }
}

function isIgnorableWatchError(error) {
  return error instanceof Error && (error.code === "ENOENT" || error.code === "EPERM");
}

function shouldIgnorePath(candidatePath) {
  return candidatePath.split(path.sep).some((part) => ignoredDirectoryNames.has(part));
}

function shouldTriggerRun(candidatePath) {
  const relativePath = path.relative(repoRoot, candidatePath).split(path.sep).join("/");
  if (["fern/openapi.yml", "fern/proxy-openapi.yml"].includes(relativePath) || relativePath.startsWith("docs/_static/python-sdk-reference/")) {
    return false;
  }
  return true;
}

function scheduleRun(changedPath) {
  if (requiresReferenceRegeneration(changedPath)) {
    referenceRevision += 1;
  }
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => runFernGenerate("file change"), debounceMs);
}

function requiresReferenceRegeneration(candidatePath) {
  if (!candidatePath) {
    return true;
  }
  const relativePath = path.relative(repoRoot, candidatePath).split(path.sep).join("/");
  return relativePath === "fern/docs.yml" || relativePath === "fern/fern.config.json" ||
    relativePath.startsWith("nemoguardrails/") || relativePath.startsWith("scripts/") ||
    watchFiles.includes(relativePath);
}

function runFernGenerate(reason) {
  if (running) {
    pending = true;
    return;
  }

  running = true;
  pending = false;
  let fernVersion;
  try {
    fernVersion = readFernVersion();
  } catch (error) {
    running = false;
    console.error(error instanceof Error ? error.message : String(error));
    if (pending) {
      runFernGenerate("queued file change");
    }
    return;
  }

  const args = [
    "--yes",
    `fern-api@${fernVersion}`,
    "generate",
    "--docs",
    "--preview",
    "--id",
    branchName,
    "--instance",
    fernDocsInstance,
    "--force",
  ];

  console.log(`\n[${new Date().toLocaleTimeString()}] Running Fern (${reason})`);
  const targetReferenceRevision = referenceRevision;
  if (generatedReferenceRevision !== targetReferenceRevision) {
    if (!generateReferences()) {
      running = false;
      if (pending) {
        runFernGenerate("queued file change");
      }
      return;
    }
    generatedReferenceRevision = targetReferenceRevision;
  }
  console.log(`cd fern && npx ${args.join(" ")}`);

  let fernEnvironment;
  try {
    fernEnvironment = createFernRefSdkEnvironment(repoRoot);
  } catch {
    running = false;
    console.error("Failed to configure Fern preview generation. Check the Git configuration values.");
    if (pending) {
      runFernGenerate("queued file change");
    }
    return;
  }

  const child = spawn("npx", args, {
    cwd: fernRoot,
    env: fernEnvironment,
    stdio: "inherit",
  });
  currentChild = child;

  child.on("error", (error) => {
    running = false;
    currentChild = undefined;
    console.error(`Failed to start Fern preview generation: ${error.message}`);
  });

  child.on("exit", (code, signal) => {
    running = false;
    currentChild = undefined;

    if (signal) {
      console.log(`Fern stopped by signal ${signal}.`);
    } else if (code === 0) {
      console.log("Fern preview generation completed.");
    } else {
      console.error(`Fern preview generation failed with exit code ${code}.`);
    }

    if (pending) {
      runFernGenerate("queued file change");
    }
  });
}

function generateReferences() {
  const result = spawnSync("make", ["docs-fern-generate-references"], {
    cwd: repoRoot,
    stdio: "inherit",
  });
  if (result.status === 0) {
    return true;
  }
  console.error("Failed to generate OpenAPI and Python SDK references.");
  return false;
}

function closeWatchers() {
  for (const watcher of watchers.values()) {
    watcher.close();
  }
}
