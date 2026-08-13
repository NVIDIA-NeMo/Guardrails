#!/usr/bin/env node

import { execFileSync } from "node:child_process";
import {
  cpSync,
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  renameSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import path from "node:path";
import { parse } from "yaml";

const libraryName = "guardrails-python-sdk";
const cacheSchemaVersion = "v1";
const minimumPageCount = 10;
const [worktreeRootArgument, expectedCommit] = process.argv.slice(2);
const repoRoot = requiredEnvironment("FERN_REF_SDK_REPO_ROOT");
const cacheRoot = requiredEnvironment("FERN_REF_SDK_CACHE_ROOT");
const fernVersion = requiredEnvironment("FERN_REF_SDK_VERSION");

if (!worktreeRootArgument || !expectedCommit) {
  throw new Error("Usage: cache-fern-ref-sdk.mjs <worktree-root> <commit-sha>");
}
if (!/^[0-9a-f]{40}$/i.test(expectedCommit)) {
  throw new Error(`Invalid ref commit SHA: ${expectedCommit}`);
}

const worktreeRoot = path.resolve(worktreeRootArgument);
const actualCommit = execFileSync("git", ["rev-parse", "HEAD"], {
  cwd: worktreeRoot,
  encoding: "utf8",
}).trim();
if (actualCommit !== expectedCommit) {
  throw new Error(`Fern ref worktree is at ${actualCommit}, expected ${expectedCommit}`);
}

const fernRoot = path.join(worktreeRoot, "fern");
const docsConfigPath = path.join(fernRoot, "docs.yml");
const fernConfigPath = path.join(fernRoot, "fern.config.json");
const docsConfig = parse(readFileSync(docsConfigPath, "utf8"));
const libraryConfig = docsConfig?.libraries?.[libraryName];
if (!libraryConfig) {
  throw new Error(`Historical ref ${expectedCommit} does not configure library ${libraryName}`);
}
if (typeof libraryConfig.input?.ref !== "string" || libraryConfig.input.ref.length === 0) {
  throw new Error(`Historical ref ${expectedCommit} must pin libraries.${libraryName}.input.ref`);
}
if (typeof libraryConfig.output?.path !== "string" || libraryConfig.output.path.length === 0) {
  throw new Error(`Historical ref ${expectedCommit} does not configure an SDK output path`);
}

const outputRoot = path.resolve(fernRoot, libraryConfig.output.path);
assertInsideWorktree(outputRoot, worktreeRoot);
const sdkRoot = path.join(outputRoot, libraryName);
const cacheDirectory = path.join(
  cacheRoot,
  cacheSchemaVersion,
  fernVersion,
  expectedCommit,
  libraryName,
);

if (isCompleteReference(cacheDirectory)) {
  restoreFromCache(cacheDirectory, outputRoot);
  console.log(`Restored ${libraryName} for ${libraryConfig.input.ref} from cache.`);
  process.exit(0);
}

console.log(`Generating ${libraryName} for ${libraryConfig.input.ref}.`);
rmSync(outputRoot, { force: true, recursive: true });
const originalFernConfig = readFileSync(fernConfigPath, "utf8");
const fernConfig = JSON.parse(originalFernConfig);
fernConfig.version = fernVersion;
writeFileSync(fernConfigPath, `${JSON.stringify(fernConfig, null, 2)}\n`);

try {
  execFileSync(
    "npx",
    ["--yes", `fern-api@${fernVersion}`, "docs", "md", "generate", "--library", libraryName],
    {
      cwd: fernRoot,
      env: { ...process.env, FERN_REF_SDK_HOOK: "0" },
      stdio: "inherit",
    },
  );
} finally {
  writeFileSync(fernConfigPath, originalFernConfig);
}

execFileSync(process.execPath, [path.join(repoRoot, "scripts", "normalize-fern-sdk-reference.mjs")], {
  cwd: repoRoot,
  env: { ...process.env, FERN_SDK_REFERENCE_ROOT: sdkRoot },
  stdio: "inherit",
});

if (!isCompleteReference(outputRoot)) {
  throw new Error(`Generated SDK reference is incomplete for ${libraryConfig.input.ref}`);
}

writeCache(outputRoot, cacheDirectory);
console.log(`Cached ${countMdxFiles(outputRoot)} SDK pages for ${libraryConfig.input.ref}.`);

function requiredEnvironment(name) {
  const value = process.env[name];
  if (!value) {
    throw new Error(`Missing required environment variable ${name}`);
  }
  return value;
}

function assertInsideWorktree(candidatePath, rootPath) {
  const relativePath = path.relative(rootPath, candidatePath);
  if (relativePath === "" || relativePath === ".." || relativePath.startsWith(`..${path.sep}`)) {
    throw new Error(`SDK output path must stay inside the Fern ref worktree: ${candidatePath}`);
  }
}

function isCompleteReference(directory) {
  return (
    existsSync(path.join(directory, "_navigation.yml")) &&
    existsSync(path.join(directory, libraryName)) &&
    countMdxFiles(directory) >= minimumPageCount
  );
}

function countMdxFiles(directory) {
  if (!existsSync(directory)) {
    return 0;
  }
  let count = 0;
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const entryPath = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      count += countMdxFiles(entryPath);
    } else if (entry.isFile() && entry.name.endsWith(".mdx")) {
      count += 1;
    }
  }
  return count;
}

function restoreFromCache(sourceDirectory, destinationDirectory) {
  rmSync(destinationDirectory, { force: true, recursive: true });
  mkdirSync(path.dirname(destinationDirectory), { recursive: true });
  cpSync(sourceDirectory, destinationDirectory, { recursive: true });
}

function writeCache(sourceDirectory, destinationDirectory) {
  if (isCompleteReference(destinationDirectory)) {
    return;
  }
  mkdirSync(path.dirname(destinationDirectory), { recursive: true });
  const temporaryDirectory = `${destinationDirectory}.tmp-${process.pid}`;
  rmSync(temporaryDirectory, { force: true, recursive: true });
  cpSync(sourceDirectory, temporaryDirectory, { recursive: true });
  try {
    rmSync(destinationDirectory, { force: true, recursive: true });
    renameSync(temporaryDirectory, destinationDirectory);
  } finally {
    rmSync(temporaryDirectory, { force: true, recursive: true });
  }
}
