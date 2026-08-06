#!/usr/bin/env node
/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Generate frozen Python SDK references from materialized release sources.
 */

import { execFileSync } from "node:child_process";
import {
  existsSync,
  readFileSync,
  readdirSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { dirname, isAbsolute, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parse, stringify } from "yaml";

const scriptPath = fileURLToPath(import.meta.url);
const repoRoot = resolve(dirname(scriptPath), "..");
const fernDir = resolve(repoRoot, "fern");
const docsYmlPath = resolve(fernDir, "docs.yml");
const fernConfigPath = resolve(fernDir, "fern.config.json");
const generatedRoot = resolve(
  fernDir,
  "generated",
  "release-versions",
);
const manifestPath = resolve(generatedRoot, "manifest.json");
const normalizeScript = resolve(
  repoRoot,
  "scripts",
  "normalize-fern-sdk-reference.mjs",
);
const minimumGeneratedPages = 10;

function countMdxFiles(directory) {
  let count = 0;
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const path = resolve(directory, entry.name);
    if (entry.isDirectory()) {
      count += countMdxFiles(path);
    } else if (entry.isFile() && entry.name.endsWith(".mdx")) {
      count += 1;
    }
  }
  return count;
}

function assertString(value, field, release) {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`Missing ${field} for ${release ?? "release"}`);
  }
  return value;
}

function assertInside(directory, path, field, release) {
  const relativePath = relative(directory, path);
  if (
    relativePath === "" ||
    relativePath === ".." ||
    relativePath.startsWith("../") ||
    isAbsolute(relativePath)
  ) {
    throw new Error(`${field} for ${release} must stay inside ${directory}`);
  }
  return path;
}

function generateReleaseReference(release, originalDocsYml, fernVersion) {
  const displayName = assertString(release.displayName, "displayName");
  const inputPath = assertString(release.inputPath, "inputPath", displayName);
  const libraryName = assertString(
    release.libraryName,
    "libraryName",
    displayName,
  );
  const outputPath = assertString(
    release.outputPath,
    "outputPath",
    displayName,
  );
  const inputRoot = assertInside(
    generatedRoot,
    resolve(fernDir, inputPath),
    "inputPath",
    displayName,
  );
  const outputRoot = assertInside(
    generatedRoot,
    resolve(fernDir, outputPath),
    "outputPath",
    displayName,
  );
  if (!existsSync(inputRoot)) {
    throw new Error(`Materialized SDK source not found: ${inputRoot}`);
  }

  const docsConfig = parse(originalDocsYml);
  const library = docsConfig?.libraries?.[libraryName];
  if (!library) {
    throw new Error(`Library ${libraryName} is not configured in fern/docs.yml`);
  }
  library.input = { path: inputPath };
  library.output.path = outputPath;

  rmSync(outputRoot, { force: true, recursive: true });
  writeFileSync(docsYmlPath, stringify(docsConfig, { lineWidth: 0 }));
  console.log(`Generating frozen Python SDK reference for ${displayName}`);
  execFileSync(
    "npx",
    [
      "--yes",
      `fern-api@${fernVersion}`,
      "docs",
      "md",
      "generate",
      "--local",
      "--library",
      libraryName,
    ],
    { cwd: fernDir, stdio: "inherit" },
  );

  const generatedLibraryRoot = assertInside(
    outputRoot,
    resolve(outputRoot, libraryName),
    "libraryName",
    displayName,
  );
  const generatedPageCount = countMdxFiles(generatedLibraryRoot);
  if (generatedPageCount < minimumGeneratedPages) {
    throw new Error(
      `Frozen SDK generation for ${displayName} produced only ${generatedPageCount} pages`,
    );
  }
  execFileSync(process.execPath, [normalizeScript, generatedLibraryRoot], {
    cwd: repoRoot,
    stdio: "inherit",
  });
}

function hasGeneratedReleaseReference(release) {
  const displayName = assertString(release.displayName, "displayName");
  const libraryName = assertString(
    release.libraryName,
    "libraryName",
    displayName,
  );
  const outputPath = assertString(
    release.outputPath,
    "outputPath",
    displayName,
  );
  const outputRoot = assertInside(
    generatedRoot,
    resolve(fernDir, outputPath),
    "outputPath",
    displayName,
  );
  const generatedLibraryRoot = assertInside(
    outputRoot,
    resolve(outputRoot, libraryName),
    "libraryName",
    displayName,
  );
  return (
    existsSync(generatedLibraryRoot) &&
    countMdxFiles(generatedLibraryRoot) >= minimumGeneratedPages
  );
}

function main() {
  if (!existsSync(manifestPath)) {
    console.log("No materialized Fern release SDK references to generate.");
    return;
  }

  const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
  if (!Array.isArray(manifest.releases)) {
    throw new Error(`${manifestPath} must contain a releases array`);
  }

  if (process.argv.includes("--check")) {
    const missing = manifest.releases.filter(
      (release) => !hasGeneratedReleaseReference(release),
    );
    if (missing.length > 0) {
      console.error(
        `Missing generated SDK reference for: ${missing.map((release) => release.displayName).join(", ")}`,
      );
      process.exitCode = 1;
      return;
    }
    console.log("Reusing materialized release SDK references.");
    return;
  }

  const originalDocsYml = readFileSync(docsYmlPath, "utf8");
  const fernVersion = JSON.parse(readFileSync(fernConfigPath, "utf8")).version;
  assertString(fernVersion, "version", fernConfigPath);

  try {
    for (const release of manifest.releases) {
      generateReleaseReference(release, originalDocsYml, fernVersion);
      writeFileSync(docsYmlPath, originalDocsYml);
    }
  } finally {
    writeFileSync(docsYmlPath, originalDocsYml);
  }
}

main();
