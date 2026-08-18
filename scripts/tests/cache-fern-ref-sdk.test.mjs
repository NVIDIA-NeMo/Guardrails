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

test("cache helper restores a reference through its command entrypoint", (t) => {
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

  main([worktreeRoot, snapshotCommit], {
    FERN_REF_SDK_CACHE_ROOT: cacheRoot,
    FERN_REF_SDK_REPO_ROOT: repoRoot,
    FERN_REF_SDK_VERSION: fernVersion,
  });

  const outputRoot = path.join(worktreeRoot, "docs/_static/python-sdk-reference");
  assert.equal(isCompleteReference(outputRoot), true);
  assert.equal(readFileSync(path.join(outputRoot, libraryName, "page-0.mdx"), "utf8"), "restored-0");
});

test("invalid Git configuration is rejected before Fern starts", (t) => {
  const root = temporaryDirectory(t);
  assert.throws(
    () => createFernRefSdkEnvironment(root, { GIT_CONFIG_COUNT: "invalid" }),
    /Invalid GIT_CONFIG_COUNT: invalid/,
  );
});

test("preview watcher reports invalid Git configuration without crashing", async (t) => {
  const root = temporaryDirectory(t);
  const binDirectory = path.join(root, "bin");
  const makePath = path.join(binDirectory, "make");
  writeFile(root, "bin/make", "#!/bin/sh\nexit 0\n");
  chmodSync(makePath, 0o755);

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
  assert.match(errorOutput, /Invalid GIT_CONFIG_COUNT: \+0/);
  assert.equal(child.exitCode, null);

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
