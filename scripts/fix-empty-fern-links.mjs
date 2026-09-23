// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import fs from "node:fs";
import path from "node:path";
import { parse } from "yaml";

const repoRoot = path.resolve(new URL("..", import.meta.url).pathname);
const docsRoot = path.join(repoRoot, "docs");
const indexPath = path.join(docsRoot, "index.yml");
const unresolved = [];
let replacements = 0;

function titleFromSlug(value) {
  return value
    .replace(/^#/, "")
    .split(/[-_]+/)
    .filter(Boolean)
    .map((word) => {
      if (/^(api|cli|llm|sdk|rag|pii|nim|yaml|json)$/i.test(word)) {
        return word.toUpperCase();
      }
      return word.charAt(0).toUpperCase() + word.slice(1);
    })
    .join(" ");
}

function normalizeRoute(route) {
  if (!route) {
    return "/";
  }
  let normalized = route.trim();
  if (!normalized.startsWith("/")) {
    normalized = `/${normalized}`;
  }
  normalized = normalized.replace(/\.mdx$/, "");
  normalized = normalized.replace(/\/index$/, "");
  normalized = normalized.replace(/\/+$/, "");
  return normalized || "/";
}

function pathRoute(filePath) {
  return normalizeRoute(filePath.replace(/\\/g, "/").replace(/\.mdx$/, ""));
}

function addRoute(routes, route, title) {
  if (!route || !title) {
    return;
  }
  routes.set(normalizeRoute(route), title);
}

function collectRoutes(items, routes, parentSlugs = []) {
  if (!Array.isArray(items)) {
    return;
  }

  for (const item of items) {
    const title = item.page || item.section || item.api || item.title;
    if (item.path && title) {
      addRoute(routes, pathRoute(item.path), title);
    }
    if (item.slug && title) {
      addRoute(routes, `/${[...parentSlugs, item.slug].join("/")}`, title);
    }
    if (item.api) {
      addRoute(routes, `/api-reference/${item.api}`, item.api);
    }
    const childParentSlugs = item.slug ? [...parentSlugs, item.slug] : parentSlugs;
    collectRoutes(item.contents, routes, childParentSlugs);
    collectRoutes(item.layout, routes, childParentSlugs);
  }
}

function buildRoutes() {
  const index = parse(fs.readFileSync(indexPath, "utf8"));
  const routes = new Map();
  const navigation = Array.isArray(index.navigation) ? index.navigation : index.navigation?.layout;
  collectRoutes(navigation, routes);
  collectFileRoutes(routes);
  return routes;
}

function frontmatterTitle(text) {
  const match = /^---\n([\s\S]*?)\n---\n?/.exec(text);
  if (!match) {
    return null;
  }
  const metadata = parse(match[1]) ?? {};
  return metadata["sidebar-title"] ?? metadata.title?.nav ?? metadata.title?.page ?? metadata.title ?? null;
}

function collectFileRoutes(routes) {
  for (const filePath of walk(docsRoot).filter((candidate) => candidate.endsWith(".mdx"))) {
    const relative = path.relative(docsRoot, filePath).replace(/\\/g, "/");
    const title = frontmatterTitle(fs.readFileSync(filePath, "utf8"));
    if (title) {
      addRoute(routes, pathRoute(relative), String(title));
    }
  }
}

function walk(dir) {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  return entries.flatMap((entry) => {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      return walk(fullPath);
    }
    return [fullPath];
  });
}

function resolveTitle(target, routes) {
  const [routeWithQuery, anchor = ""] = target.split("#", 2);
  const [route] = routeWithQuery.split("?", 1);

  if (target.startsWith("#")) {
    return { title: titleFromSlug(target), target };
  }

  if (!route.startsWith("/") && !route.includes("/") && !route.includes(".")) {
    return { title: titleFromSlug(route), target: `#${route}` };
  }

  const routeTitle = routes.get(normalizeRoute(route));
  if (routeTitle) {
    return { title: anchor ? `${routeTitle}: ${titleFromSlug(anchor)}` : routeTitle, target };
  }

  return null;
}

function fixFile(filePath, routes) {
  const original = fs.readFileSync(filePath, "utf8");
  const relative = path.relative(repoRoot, filePath);
  let fileReplacements = 0;

  const updated = original.replace(/(!?)\[\]\(([^)\s]+)(\s+["'][^)"']*["'])?\)/g, (match, bang, target, title) => {
    if (bang) {
      return match;
    }
    if (/^(?:https?:|mailto:|api:)/.test(target)) {
      unresolved.push(`${relative}: ${target}`);
      return match;
    }
    const resolved = resolveTitle(target, routes);
    if (!resolved) {
      unresolved.push(`${relative}: ${target}`);
      return match;
    }
    fileReplacements += 1;
    replacements += 1;
    return `[${resolved.title}](${resolved.target})${title ?? ""}`;
  });

  if (updated !== original) {
    fs.writeFileSync(filePath, updated);
    console.log(`${relative}: ${fileReplacements} replacements`);
  }
}

const routes = buildRoutes();
for (const file of walk(docsRoot).filter((filePath) => filePath.endsWith(".mdx"))) {
  fixFile(file, routes);
}

console.log(`Total replacements: ${replacements}`);
if (unresolved.length > 0) {
  console.error("Unresolved empty links:");
  for (const item of unresolved) {
    console.error(`- ${item}`);
  }
  process.exitCode = 1;
}
