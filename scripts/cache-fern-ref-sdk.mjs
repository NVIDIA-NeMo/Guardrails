#!/usr/bin/env node

import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
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
import { fileURLToPath } from "node:url";
import { parse } from "yaml";

const libraryName = "guardrails-python-sdk";
const cacheSchemaVersion = "v2";
// CI restores archives by a broad prefix, so each entry also carries a content
// fingerprint and cannot be reused after the generation pipeline changes.
const generatorInputPaths = [
  "package.json",
  "package-lock.json",
  "scripts/cache-fern-ref-sdk.mjs",
  "scripts/fern-ref-sdk-environment.mjs",
  "scripts/fern-ref-sdk-hooks/post-checkout",
  "scripts/normalize-fern-sdk-reference.mjs",
];
const minimumPageCount = 10;

export function main(args = process.argv.slice(2), environment = process.env) {
  const [worktreeRootArgument, expectedCommit] = args;
  if (!worktreeRootArgument || !expectedCommit) {
    throw new Error("Usage: cache-fern-ref-sdk.mjs <worktree-root> <commit-sha>");
  }
  if (!/^[0-9a-f]{40}$/i.test(expectedCommit)) {
    throw new Error(`Invalid ref commit SHA: ${expectedCommit}`);
  }

  const repoRoot = requiredEnvironment("FERN_REF_SDK_REPO_ROOT", environment);
  const cacheRoot = requiredEnvironment("FERN_REF_SDK_CACHE_ROOT", environment);
  const fernVersion = requiredEnvironment("FERN_REF_SDK_VERSION", environment);
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
  if (typeof libraryConfig.input?.git !== "string" || libraryConfig.input.git.length === 0) {
    throw new Error(`Historical ref ${expectedCommit} must configure libraries.${libraryName}.input.git`);
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
  const sdkInputCommit = resolveSdkInputCommit(libraryConfig.input.git, libraryConfig.input.ref, {
    environment,
    expectedCommit,
  });
  const generatorFingerprint = computeGeneratorFingerprint(repoRoot);
  const cacheDirectory = path.join(
    cacheRoot,
    cacheSchemaVersion,
    generatorFingerprint,
    fernVersion,
    expectedCommit,
    sdkInputCommit,
    libraryName,
  );

  if (isCompleteReference(cacheDirectory)) {
    restoreFromCache(cacheDirectory, outputRoot);
    console.log(
      `Restored ${libraryName} for ${libraryConfig.input.ref} (${sdkInputCommit.slice(0, 12)}) from cache.`,
    );
    return;
  }

  console.log(`Generating ${libraryName} for ${libraryConfig.input.ref} (${sdkInputCommit.slice(0, 12)}).`);
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
        env: { ...environment, FERN_REF_SDK_HOOK: "0" },
        stdio: "inherit",
      },
    );
  } finally {
    writeFileSync(fernConfigPath, originalFernConfig);
  }

  execFileSync(process.execPath, [path.join(repoRoot, "scripts", "normalize-fern-sdk-reference.mjs")], {
    cwd: repoRoot,
    env: { ...environment, FERN_SDK_REFERENCE_ROOT: sdkRoot },
    stdio: "inherit",
  });

  if (!isCompleteReference(outputRoot)) {
    throw new Error(`Generated SDK reference is incomplete for ${libraryConfig.input.ref}`);
  }

  writeCache(outputRoot, cacheDirectory);
  console.log(`Cached ${countMdxFiles(outputRoot)} SDK pages for ${libraryConfig.input.ref}.`);
}

export function requiredEnvironment(name, environment = process.env) {
  const value = environment[name];
  if (!value) {
    throw new Error(`Missing required environment variable ${name}`);
  }
  return value;
}

export function resolveSdkInputCommit(
  gitUrl,
  inputRef,
  {
    environment = process.env,
    expectedCommit = "unknown",
    executor = execFileSync,
    referenceLibraryName = libraryName,
  } = {},
) {
  if (/^[0-9a-f]{40}$/i.test(inputRef)) {
    return inputRef.toLowerCase();
  }

  const tagRef = inputRef.startsWith("refs/tags/") ? inputRef : `refs/tags/${inputRef}`;
  let output;
  try {
    output = executor("git", ["ls-remote", "--tags", gitUrl, tagRef, `${tagRef}^{}`], {
      encoding: "utf8",
      env: { ...environment, GIT_TERMINAL_PROMPT: "0" },
      stdio: ["ignore", "pipe", "pipe"],
      timeout: 30_000,
    });
  } catch {
    throw new Error(`Could not resolve SDK input tag ${inputRef} from ${gitUrl}`);
  }

  const refs = new Map(
    output
      .trim()
      .split("\n")
      .filter(Boolean)
      .map((line) => line.split(/\s+/, 2).reverse()),
  );
  const resolvedCommit = refs.get(`${tagRef}^{}`) ?? refs.get(tagRef);
  if (typeof resolvedCommit !== "string" || !/^[0-9a-f]{40}$/i.test(resolvedCommit)) {
    throw new Error(
      `Historical ref ${expectedCommit} must pin libraries.${referenceLibraryName}.input.ref to a commit SHA or tag`,
    );
  }
  return resolvedCommit.toLowerCase();
}

export function computeGeneratorFingerprint(repoRoot, relativePaths = generatorInputPaths) {
  const hash = createHash("sha256");
  for (const relativePath of relativePaths) {
    const inputPath = path.join(repoRoot, relativePath);
    if (!existsSync(inputPath)) {
      throw new Error(`Missing SDK cache generator input: ${relativePath}`);
    }
    hash.update(relativePath);
    hash.update("\0");
    hash.update(readFileSync(inputPath));
    hash.update("\0");
  }
  return hash.digest("hex");
}

export function assertInsideWorktree(candidatePath, rootPath) {
  const relativePath = path.relative(rootPath, candidatePath);
  if (relativePath === "" || relativePath === ".." || relativePath.startsWith(`..${path.sep}`)) {
    throw new Error(`SDK output path must stay inside the Fern ref worktree: ${candidatePath}`);
  }
}

export function isCompleteReference(directory, referenceLibraryName = libraryName) {
  return (
    existsSync(path.join(directory, "_navigation.yml")) &&
    existsSync(path.join(directory, referenceLibraryName)) &&
    countMdxFiles(directory) >= minimumPageCount
  );
}

export function countMdxFiles(directory) {
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

export function restoreFromCache(sourceDirectory, destinationDirectory) {
  rmSync(destinationDirectory, { force: true, recursive: true });
  mkdirSync(path.dirname(destinationDirectory), { recursive: true });
  cpSync(sourceDirectory, destinationDirectory, { recursive: true });
}

export function writeCache(sourceDirectory, destinationDirectory, referenceLibraryName = libraryName) {
  if (isCompleteReference(destinationDirectory, referenceLibraryName)) {
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

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main();
}
