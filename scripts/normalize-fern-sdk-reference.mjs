// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const sdkRoot = path.join(repoRoot, "docs", "_static", "python-sdk-reference", "guardrails-python-sdk");

if (!fs.existsSync(sdkRoot)) {
  throw new Error(`Generated SDK reference folder not found: ${path.relative(repoRoot, sdkRoot)}`);
}

let moved = 0;
let removed = 0;

function walk(directory) {
  for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    const fullPath = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      walk(fullPath);
      continue;
    }

    if (!entry.isFile() || !entry.name.endsWith(".mdx") || entry.name === "index.mdx") {
      continue;
    }

    const stem = entry.name.slice(0, -".mdx".length);
    const siblingDirectory = path.join(directory, stem);
    if (!fs.existsSync(siblingDirectory) || !fs.statSync(siblingDirectory).isDirectory()) {
      continue;
    }

    const target = path.join(siblingDirectory, "index.mdx");
    if (fs.existsSync(target)) {
      fs.rmSync(fullPath);
      removed += 1;
      continue;
    }

    fs.renameSync(fullPath, target);
    moved += 1;
  }
}

walk(sdkRoot);

console.log(`Moved ${moved} generated package overview pages to index.mdx.`);
console.log(`Removed ${removed} duplicate package overview pages with existing index.mdx.`);
