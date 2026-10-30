import assert from "node:assert/strict";
import { execFileSync, spawn } from "node:child_process";
import { once } from "node:events";
import {
  chmodSync,
  existsSync,
  mkdtempSync,
  mkdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  assertInsideWorktree,
  computeGeneratorFingerprint,
  isCompleteReference,
  main,
  resolveSdkInputCommit,
  restoreFromCache,
  writeCache,
} from "../cache-fern-ref-sdk.mjs";
import { createFernRefSdkEnvironment } from "../fern-ref-sdk-environment.mjs";

const libraryName = "guardrails-python-sdk";
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const historicalImagePage = "docs/configure-rails/guardrail-catalog/self-check.mdx";
const historicalImages = '![Sequence (step 1)](../../_static/puml/dialog_rails_flow.png "Sequence (step 1)")\n![Other](../../_static/puml/other.png "Useful title")\n';
const normalizedHistoricalImages = '![Sequence (step 1)](../../_static/puml/dialog_rails_flow.png)\n![Other](../../_static/puml/other.png "Useful title")\n';
const generatorInputPaths = [
  "package.json",
  "package-lock.json",
  "scripts/cache-fern-ref-sdk.mjs",
  "scripts/fern-ref-sdk-environment.mjs",
  "scripts/fern-ref-sdk-hooks/post-checkout",
  "scripts/normalize-fern-sdk-reference.mjs",
];

test("generator fingerprint changes with its inputs", (t) => {
  const root = temporaryDirectory(t);
  writeFile(root, "first.txt", "first");
  writeFile(root, "nested/second.txt", "second");

  const inputs = ["first.txt", "nested/second.txt"];
  const original = computeGeneratorFingerprint(root, inputs);
  assert.equal(computeGeneratorFingerprint(root, inputs), original);

  writeFile(root, "nested/second.txt", "changed");
  assert.notEqual(computeGeneratorFingerprint(root, inputs), original);
  assert.throws(
    () => computeGeneratorFingerprint(root, ["missing.txt"]),
    /Missing SDK cache generator input/,
  );
});

test("SDK input tags resolve to their peeled commit", () => {
  const tagObject = "1".repeat(40);
  const tagCommit = "2".repeat(40);
  let invocation;
  const executor = (command, args, options) => {
    invocation = { args, command, options };
    return `${tagObject}\trefs/tags/v0.23.0\n${tagCommit}\trefs/tags/v0.23.0^{}\n`;
  };

  assert.equal(
    resolveSdkInputCommit("https://example.com/repo.git", "v0.23.0", { executor }),
    tagCommit,
  );
  assert.equal(invocation.command, "git");
  assert.deepEqual(invocation.args, [
    "ls-remote",
    "--tags",
    "https://example.com/repo.git",
    "refs/tags/v0.23.0",
    "refs/tags/v0.23.0^{}",
  ]);
  assert.equal(invocation.options.env.GIT_TERMINAL_PROMPT, "0");
  assert.equal(invocation.options.timeout, 30_000);
});

test("SDK input resolution accepts a commit and rejects an unresolved ref", () => {
  const commit = "a".repeat(40);
  assert.equal(
    resolveSdkInputCommit("https://example.com/repo.git", commit, {
      executor: () => {
        throw new Error("commit SHAs must not require remote resolution");
      },
    }),
    commit,
  );

  assert.throws(
    () =>
      resolveSdkInputCommit("https://example.com/repo.git", "main", {
        expectedCommit: "b".repeat(40),
        executor: () => "",
      }),
    /must pin libraries\.guardrails-python-sdk\.input\.ref to a commit SHA or tag/,
  );
});

test("cache writes and restores complete SDK references", (t) => {
  const root = temporaryDirectory(t);
  const source = path.join(root, "source");
  const cache = path.join(root, "cache", libraryName);
  const destination = path.join(root, "destination");
  createCompleteReference(source, "cached");

  writeCache(source, cache);
  assert.equal(isCompleteReference(cache), true);
  assert.equal(readFileSync(path.join(cache, libraryName, "page-0.mdx"), "utf8"), "cached-0");

  writeFile(destination, "stale.txt", "stale");
  restoreFromCache(cache, destination);
  assert.equal(existsSync(path.join(destination, "stale.txt")), false);
  assert.equal(isCompleteReference(destination), true);
});

test("SDK output must stay inside the historical worktree", (t) => {
  const root = temporaryDirectory(t);
  assert.doesNotThrow(() => assertInsideWorktree(path.join(root, "docs"), root));
  assert.throws(() => assertInsideWorktree(root, root), /must stay inside/);
  assert.throws(() => assertInsideWorktree(path.resolve(root, "..", "outside"), root), /must stay inside/);
});

for (const generatedOpenapi of [false, true]) {
  test(`cache helper restores SDK references with ${generatedOpenapi ? "build-time" : "legacy"} OpenAPI`, (t) => {
    const root = temporaryDirectory(t);
    const repoRoot = path.join(root, "repo-root");
    const worktreeRoot = path.join(root, "historical-worktree");
    const cacheRoot = path.join(root, "cache");
    const sdkInputCommit = "c".repeat(40);
    const fernVersion = "5.91.0";

    for (const inputPath of generatorInputPaths) {
      writeFile(repoRoot, inputPath, inputPath);
    }
    writeFile(
      worktreeRoot,
      "fern/docs.yml",
      `libraries:\n  ${libraryName}:\n    input:\n      git: https://example.com/repo.git\n      ref: ${sdkInputCommit}\n    output:\n      path: ../docs/_static/python-sdk-reference\n`,
    );
    writeFile(worktreeRoot, "fern/fern.config.json", `${JSON.stringify({ version: fernVersion })}\n`);
    writeFile(worktreeRoot, historicalImagePage, historicalImages);
    if (generatedOpenapi) {
      writeFile(worktreeRoot, "scripts/export_openapi.py", "# exporter supplied by this snapshot\n");
      writeFile(worktreeRoot, "scripts/export_hooks_openapi.py", "# exporter supplied by this snapshot\n");
    } else {
      writeFile(worktreeRoot, "fern/openapi.yml", "legacy\n");
    }
    initializeGitRepository(worktreeRoot);
    const snapshotCommit = execFileSync("git", ["rev-parse", "HEAD"], {
      cwd: worktreeRoot,
      encoding: "utf8",
    }).trim();
    const fingerprint = computeGeneratorFingerprint(repoRoot);
    const cacheDirectory = path.join(
      cacheRoot,
      "v2",
      fingerprint,
      fernVersion,
      snapshotCommit,
      sdkInputCommit,
      libraryName,
    );
    createCompleteReference(cacheDirectory, "restored");

    writeFile(root, "bin/uv", `#!${process.execPath}
  const assert = require("node:assert/strict");
  const fs = require("node:fs");
  const path = require("node:path");
  const args = process.argv.slice(2);
  assert.deepEqual(args.slice(0, 5), ["run", "--locked", "--extra", "server", "python"]);
  assert.ok(["scripts/export_openapi.py", "scripts/export_hooks_openapi.py"].includes(args[5]));
  assert.equal(process.env.FERN_TOKEN, undefined);
  assert.equal(process.env.DOCS_FERN_TOKEN, undefined);
  assert.equal(process.env.FERN_REF_SDK_HOOK, "0");
  assert.equal(fs.realpathSync(path.dirname(process.env.UV_PROJECT_ENVIRONMENT)), process.cwd());
  fs.writeFileSync(args[5].includes("hooks") ? "fern/hooks-openapi.yml" : "fern/openapi.yml", "generated from snapshot\\n");
  if (args[5].includes("hooks")) {
    fs.mkdirSync("docs/reference/hooks-server-api/_generated", { recursive: true });
    fs.writeFileSync("docs/reference/hooks-server-api/_generated/get-health.mdx", "generated from snapshot\\n");
  }
  `);
    chmodSync(path.join(root, "bin/uv"), 0o755);

    main([worktreeRoot, snapshotCommit], {
      FERN_REF_SDK_CACHE_ROOT: cacheRoot,
      FERN_REF_SDK_REPO_ROOT: repoRoot,
      FERN_REF_SDK_VERSION: fernVersion,
      PATH: `${path.join(root, "bin")}${path.delimiter}${process.env.PATH}`,
      FERN_TOKEN: "test-publishing-credential",
      DOCS_FERN_TOKEN: "test-publishing-credential",
    });

    const outputRoot = path.join(worktreeRoot, "docs/_static/python-sdk-reference");
    assert.equal(isCompleteReference(outputRoot), true);
    assert.equal(readFileSync(path.join(outputRoot, libraryName, "page-0.mdx"), "utf8"), "restored-0");
    assert.equal(readFileSync(path.join(worktreeRoot, "fern/openapi.yml"), "utf8"),
      generatedOpenapi ? "generated from snapshot\n" : "legacy\n");
    if (generatedOpenapi) {
      assert.equal(readFileSync(path.join(worktreeRoot, "fern/hooks-openapi.yml"), "utf8"), "generated from snapshot\n");
      assert.equal(readFileSync(path.join(worktreeRoot, "docs/reference/hooks-server-api/_generated/get-health.mdx"), "utf8"), "generated from snapshot\n");
    }
    assert.equal(readFileSync(path.join(worktreeRoot, historicalImagePage), "utf8"), normalizedHistoricalImages);
    assert.equal(execFileSync("git", ["show", `${snapshotCommit}:${historicalImagePage}`], {
      cwd: worktreeRoot,
      encoding: "utf8",
    }), historicalImages);
  });
}

test("invalid Git configuration is rejected before Fern starts", (t) => {
  const root = temporaryDirectory(t);
  assert.throws(
    () => createFernRefSdkEnvironment(root, { GIT_CONFIG_COUNT: "invalid" }),
    /Invalid GIT_CONFIG_COUNT: invalid/,
  );
});

test("cold historical generation resets the parent Fern working directory and redirect guard", (t) => {
  const root = temporaryDirectory(t);
  const repoRoot = path.join(root, "parent");
  const worktreeRoot = path.join(root, "snapshot");
  for (const input of generatorInputPaths) writeFile(repoRoot, input, input);
  writeFile(repoRoot, "scripts/normalize-fern-sdk-reference.mjs", `
import assert from "node:assert/strict";
import fs from "node:fs";
assert.ok(fs.existsSync(process.env.FERN_SDK_REFERENCE_ROOT));
`);
  writeFile(worktreeRoot, "fern/docs.yml", `libraries:
  ${libraryName}:
    input:
      git: https://example.com/repo.git
      ref: ${"c".repeat(40)}
    output:
      path: ../docs/_static/python-sdk-reference
`);
  const originalConfig = '{"version":"5.91.0"}\n';
  writeFile(worktreeRoot, "fern/fern.config.json", originalConfig);
  writeFile(worktreeRoot, historicalImagePage, historicalImages);
  initializeGitRepository(worktreeRoot);
  const commit = execFileSync("git", ["rev-parse", "HEAD"], { cwd: worktreeRoot, encoding: "utf8" }).trim();
  writeFile(root, "bin/npx", `#!${process.execPath}
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
assert.equal(fs.readFileSync(path.join("..", ${JSON.stringify(historicalImagePage)}), "utf8"), ${JSON.stringify(normalizedHistoricalImages)});
assert.equal(fs.realpathSync(process.env.FERN_CWD), process.cwd());
assert.equal(process.env.FERN_NO_VERSION_REDIRECTION, undefined);
assert.equal(process.env.FERN_REF_SDK_HOOK, "0");
const output = path.resolve("../docs/_static/python-sdk-reference");
fs.mkdirSync(path.join(output, "${libraryName}"), {recursive: true});
fs.writeFileSync(path.join(output, "_navigation.yml"), "navigation: []\\n");
for (let i = 0; i < 10; i++) fs.writeFileSync(path.join(output, "${libraryName}", i + ".mdx"), "page\\n");
`);
  chmodSync(path.join(root, "bin/npx"), 0o755);
  main([worktreeRoot, commit], {
    FERN_REF_SDK_REPO_ROOT: repoRoot,
    FERN_REF_SDK_CACHE_ROOT: path.join(root, "cache"),
    FERN_REF_SDK_VERSION: "5.123.0",
    FERN_CWD: path.join(repoRoot, "fern"),
    FERN_NO_VERSION_REDIRECTION: "true",
    PATH: `${path.join(root, "bin")}${path.delimiter}${process.env.PATH}`,
  });
  assert.ok(isCompleteReference(path.join(worktreeRoot, "docs/_static/python-sdk-reference")));
  assert.equal(readFileSync(path.join(worktreeRoot, "fern/fern.config.json"), "utf8"), originalConfig);
});

test("preview watcher reports invalid Git configuration without crashing", async (t) => {
  const root = temporaryDirectory(t);
  const binDirectory = path.join(root, "bin");
  const makePath = path.join(binDirectory, "make");
  const gitPath = path.join(binDirectory, "git");
  writeFile(root, "bin/make", "#!/bin/sh\nexit 0\n");
  writeFile(root, "bin/git", "#!/bin/sh\nprintf '%s\\n' 'test-preview-branch'\n");
  chmodSync(makePath, 0o755);
  chmodSync(gitPath, 0o755);

  const child = spawn(process.execPath, [path.join(repoRoot, "scripts/watch-fern-preview.mjs")], {
    cwd: repoRoot,
    env: {
      ...process.env,
      // Git accepts +0, but the helper rejects it because the value is not canonical.
      GIT_CONFIG_COUNT: "+0",
      PATH: `${binDirectory}${path.delimiter}${process.env.PATH}`,
    },
    stdio: ["ignore", "pipe", "pipe"],
  });
  t.after(() => {
    if (child.exitCode === null) {
      child.kill("SIGTERM");
    }
  });

  const errorOutput = await waitForOutput(child, "Failed to configure Fern preview generation");
  assert.doesNotMatch(errorOutput, /\+0/);
  assert.equal(child.exitCode, null);

  child.kill("SIGTERM");
  await once(child, "exit");
});

test("preview watcher regenerates on source changes and ignores generated OpenAPI", async (t) => {
  const root = temporaryDirectory(t);
  for (const script of ["watch-fern-preview.mjs", "fern-ref-sdk-environment.mjs"]) {
    writeFile(root, `scripts/${script}`, readFileSync(path.join(repoRoot, "scripts", script), "utf8"));
  }
  writeFile(root, "fern/fern.config.json", '{"version":"5.123.0"}\n');
  writeFile(root, "docs/index.yml", "navigation: []\n");
  writeFile(root, "docs/reference/hooks-server-api/_generated/get-health.mdx", "stale\n");
  writeFile(root, "nemoguardrails/api.py", "original\n");
  writeFile(root, "pyproject.toml", "");
  writeFile(root, "uv.lock", "");
  writeFile(root, "bin/git", "#!/bin/sh\nprintf '%s\\n' 'test-preview-branch'\n");
  writeFile(root, "bin/make", `#!${process.execPath}
const fs = require("node:fs");
require("node:assert/strict").deepEqual(process.argv.slice(2), ["docs-fern-generate-references"]);
fs.appendFileSync("events", "generate\\n");
fs.writeFileSync("fern/openapi.yml", "generated\\n");
fs.writeFileSync("fern/hooks-openapi.yml", "generated\\n");
fs.mkdirSync("docs/reference/hooks-server-api/_generated", { recursive: true });
fs.writeFileSync("docs/reference/hooks-server-api/_generated/get-health.mdx", "generated\\n");
`);
  writeFile(root, "bin/npx", `#!${process.execPath}
require("node:fs").appendFileSync("../events", "publish\\n");
`);
  for (const name of ["git", "make", "npx"]) {
    chmodSync(path.join(root, "bin", name), 0o755);
  }
  const child = spawn(process.execPath, [path.join(root, "scripts/watch-fern-preview.mjs")], {
    cwd: root,
    env: { ...process.env, PATH: `${path.join(root, "bin")}${path.delimiter}${process.env.PATH}` },
    stdio: ["ignore", "pipe", "pipe"],
  });
  t.after(() => {
    if (child.exitCode === null) child.kill("SIGTERM");
  });
  await waitForOutput(child, "Fern preview generation completed.");
  writeFile(root, "nemoguardrails/api.py", "changed\n");
  await waitForOutput(child, "Fern preview generation completed.");
  await new Promise((resolve) => setTimeout(resolve, 1_000));
  assert.equal(readFileSync(path.join(root, "events"), "utf8"), "generate\npublish\ngenerate\npublish\n");
  child.kill("SIGTERM");
  await once(child, "exit");
});

function temporaryDirectory(t) {
  const directory = mkdtempSync(path.join(os.tmpdir(), "fern-ref-sdk-test-"));
  t.after(() => rmSync(directory, { force: true, recursive: true }));
  return directory;
}

function writeFile(root, relativePath, contents) {
  const filePath = path.join(root, relativePath);
  mkdirSync(path.dirname(filePath), { recursive: true });
  writeFileSync(filePath, contents);
}

function createCompleteReference(directory, prefix) {
  writeFile(directory, "_navigation.yml", "navigation: []\n");
  for (let index = 0; index < 10; index += 1) {
    writeFile(directory, `${libraryName}/page-${index}.mdx`, `${prefix}-${index}`);
  }
}

function initializeGitRepository(directory) {
  execFileSync("git", ["init", "--quiet"], { cwd: directory });
  execFileSync("git", ["config", "user.name", "Fern Test"], { cwd: directory });
  execFileSync("git", ["config", "user.email", "fern-test@example.com"], { cwd: directory });
  execFileSync("git", ["config", "commit.gpgsign", "false"], { cwd: directory });
  execFileSync("git", ["add", "."], { cwd: directory });
  execFileSync("git", ["commit", "--quiet", "-m", "test snapshot"], { cwd: directory });
}

function waitForOutput(child, expectedText) {
  return new Promise((resolve, reject) => {
    let output = "";
    const timeout = setTimeout(() => reject(new Error(`Timed out waiting for: ${expectedText}`)), 5_000);
    const receiveOutput = (chunk) => {
      output += chunk.toString();
      if (output.includes(expectedText)) {
        clearTimeout(timeout);
        resolve(output);
      }
    };
    child.stdout.on("data", receiveOutput);
    child.stderr.on("data", receiveOutput);
    child.once("exit", (code) => {
      clearTimeout(timeout);
      reject(new Error(`Preview watcher exited with ${code} before reporting: ${expectedText}\n${output}`));
    });
  });
}
