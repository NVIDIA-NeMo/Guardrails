// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import fs from "node:fs";
import path from "node:path";
import { parse, stringify } from "yaml";

const repoRoot = path.resolve(new URL("..", import.meta.url).pathname);
const docsRoot = path.join(repoRoot, "docs");
const outputPath = path.join(docsRoot, "index.yml");

const slugify = (value) =>
  String(value)
    .toLowerCase()
    .replace(/&/g, "and")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");

function toPosix(value) {
  return value.split(path.sep).join("/");
}

function splitFrontmatter(text) {
  const match = /^---\n([\s\S]*?)\n---\n?/.exec(text);
  if (!match) {
    return [{}, text];
  }
  return [parse(match[1]) ?? {}, text.slice(match[0].length)];
}

function readSource(relativePath) {
  const fullPath = path.join(docsRoot, relativePath);
  return fs.readFileSync(fullPath, "utf8");
}

function pageTitle(relativePath) {
  const [metadata, body] = splitFrontmatter(readSource(relativePath));
  const explicitTitle = metadata.title?.nav ?? metadata["sidebar-title"] ?? metadata.title?.page ?? metadata.title;
  if (explicitTitle) {
    return String(explicitTitle);
  }
  const heading = body.match(/^#\s+(.+)$/m);
  if (heading) {
    return heading[1].trim();
  }
  const basename = path.basename(relativePath, ".md");
  return basename === "README" || basename === "index"
    ? path.basename(path.dirname(relativePath))
    : basename;
}

function parseToctree(block) {
  const lines = block.split("\n");
  const entries = [];
  let caption = "";
  let name = "";

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) {
      continue;
    }
    if (line.startsWith(":caption:")) {
      caption = line.replace(":caption:", "").trim();
      continue;
    }
    if (line.startsWith(":name:")) {
      name = line.replace(":name:", "").trim();
      continue;
    }
    if (line.startsWith(":")) {
      continue;
    }

    const titled = line.match(/^(.*?)\s*<([^>]+)>\s*$/);
    if (titled) {
      entries.push({ title: titled[1].trim(), target: titled[2].trim() });
      continue;
    }
    entries.push({ target: line });
  }

  return { caption, name, entries };
}

function toctreesFor(relativePath) {
  const text = readSource(relativePath);
  const matches = [...text.matchAll(/```\{toctree\}([\s\S]*?)```/g)];
  return matches.map((match) => parseToctree(match[1]));
}

function resolveSource(baseRelativePath, target) {
  const cleanTarget = target.split("#", 1)[0].split("?", 1)[0];
  const baseDir = path.dirname(baseRelativePath);
  const resolved = path.normalize(path.join(baseDir, cleanTarget));
  const candidates = [];
  const skippedCandidates = [];

  const extension = path.extname(resolved);
  if (extension === ".md") {
    candidates.push(resolved);
  } else if (extension === ".rst") {
    skippedCandidates.push(resolved);
  } else {
    candidates.push(
      `${resolved}.md`,
      path.join(resolved, "index.md"),
      path.join(resolved, "README.md"),
    );
    skippedCandidates.push(
      `${resolved}.rst`,
      path.join(resolved, "index.rst"),
      path.join(resolved, "README.rst"),
    );
  }

  for (const candidate of candidates) {
    const relative = toPosix(candidate);
    if (fs.existsSync(path.join(docsRoot, relative))) {
      return relative;
    }
  }

  for (const candidate of skippedCandidates) {
    const relative = toPosix(candidate);
    if (fs.existsSync(path.join(docsRoot, relative))) {
      console.error(`Skipped non-MDX toctree target "${relative}".`);
      return null;
    }
  }

  throw new Error(`Could not resolve toctree target "${target}" from "${baseRelativePath}".`);
}

function mdxPathFor(sourceRelativePath) {
  const mdxPath = sourceRelativePath.replace(/\.md$/, ".mdx");
  if (!fs.existsSync(path.join(docsRoot, mdxPath))) {
    throw new Error(`Missing converted MDX file for "${sourceRelativePath}": expected "${mdxPath}".`);
  }
  return mdxPath;
}

function slugForMdxPath(mdxPath) {
  const parts = mdxPath.split("/");
  const basename = path.basename(mdxPath, ".mdx");
  if (basename === "index" || basename === "README") {
    return slugify(parts.length > 1 ? parts[parts.length - 2] : "home");
  }
  return slugify(basename);
}

function itemForEntry(entry, baseRelativePath, ancestors = new Set()) {
  const sourcePath = resolveSource(baseRelativePath, entry.target);
  if (sourcePath === null) {
    return null;
  }
  if (ancestors.has(sourcePath)) {
    throw new Error(`Detected recursive toctree reference to "${sourcePath}".`);
  }

  const mdxPath = mdxPathFor(sourcePath);
  const title = entry.title || pageTitle(sourcePath);
  const nestedAncestors = new Set([...ancestors, sourcePath]);
  const childItems = toctreesFor(sourcePath)
    .flatMap((toctree) =>
      toctree.entries.map((child) => itemForEntry(child, sourcePath, nestedAncestors)),
    )
    .filter((item) => item !== null);

  if (childItems.length > 0) {
    return {
      section: title,
      path: mdxPath,
      slug: slugForMdxPath(mdxPath),
      collapsed: "open-by-default",
      contents: childItems,
    };
  }

  return {
    page: title,
    path: mdxPath,
    slug: slugForMdxPath(mdxPath),
  };
}

function buildNavigation() {
  const layout = [
    {
      page: "Home",
      path: "index.mdx",
      slug: "home",
    },
  ];

  for (const toctree of toctreesFor("index.md")) {
    const title = toctree.caption || toctree.name;
    if (!title) {
      throw new Error("Root toctree is missing :caption: or :name:.");
    }
    layout.push({
      section: title,
      slug: slugify(title),
      collapsed: "open-by-default",
      contents: toctree.entries
        .map((entry) => itemForEntry(entry, "index.md"))
        .filter((item) => item !== null),
    });
  }

  return {
    tabs: {
      "user-guide": {
        "display-name": "User Guide",
        icon: "book-open",
      },
    },
    navigation: [
      {
        tab: "user-guide",
        layout,
      },
    ],
  };
}

const header = [
  "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
  "# SPDX-License-Identifier: Apache-2.0",
  "",
].join("\n");

const output = `${header}${stringify(buildNavigation())}`;
if (!fs.existsSync(outputPath) || fs.readFileSync(outputPath, "utf8") !== output) {
  fs.writeFileSync(outputPath, output);
  console.log(path.relative(repoRoot, outputPath));
}
