// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { execFileSync, spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  exactTimestamp,
  isAuthorizedAssociation,
  isDocumentationRelatedPath,
  isPublicDocumentationPath,
  isReleaseDocumentationPath,
  isReleaseSnapshotPath,
  isReviewCommand,
  managedBranch,
  managedReleaseBranch,
  parseMaintainerLogins,
  releaseSnapshotTag,
  requestedReviewers,
  stableVersion,
  validateQualityScore,
  validateReviewResult,
} from "../../tools/docs-agent/contract.mjs";
import { selectPostMerge } from "../../tools/docs-agent/select-post-merge.mjs";
import { selectRelease } from "../../tools/docs-agent/select-release.mjs";
import { manualReviewRequest } from "../../tools/docs-agent/select-review.mjs";
import {
  buildReleaseAuthorPrompt,
  buildReleaseCoverageReviewPrompt,
  withCleanup,
} from "../../tools/docs-agent/agent.mjs";
import { isAllowedGithubApiPath, workflowOutput } from "../../tools/docs-agent/github.mjs";
import {
  cleanupInference,
  credentialFreeEnvironment,
} from "../../tools/docs-agent/openshell-runtime.mjs";
import {
  remoteBranchCommit,
  validateManagedBranch,
} from "../../tools/docs-agent/publish-release.mjs";

const SHA_A = "a".repeat(40);
const SHA_B = "b".repeat(40);
const SHA_C = "c".repeat(40);

function quality(overrides = {}) {
  return {
    confidence: 0.9,
    decision: "pass",
    hard_gates: {
      broken_required_examples: 0,
      critical_factual_errors: 0,
      invalid_required_artifacts: 0,
      missing_mandatory_content: 0,
      security_compliance_violations: 0,
      unsupported_critical_claims: 0,
    },
    rationale: "The changed task is accurate and complete.",
    rubric: "task-guide-v0.1",
    scores: {
      clarity: 9,
      completeness: 18,
      source_grounding: 14,
      style_compliance: 5,
      task_success: 18,
      technical_accuracy: 28,
    },
    total: 92,
    ...overrides,
  };
}

test("separates writable public documentation from related review surfaces", () => {
  assert.equal(isPublicDocumentationPath("docs/getting-started/example.mdx"), true);
  assert.equal(isPublicDocumentationPath("fern/docs.yml"), true);
  assert.equal(isPublicDocumentationPath("fern/assets/logo.svg"), true);
  assert.equal(isPublicDocumentationPath("fern/fern.config.json"), false);
  assert.equal(isPublicDocumentationPath("docs/_build/result.html"), false);
  assert.equal(isDocumentationRelatedPath("scripts/run-fern-with-ref-sdk.mjs"), true);
  assert.equal(isDocumentationRelatedPath("scripts/fern-ref-sdk-hooks/post-checkout"), true);
  assert.equal(isDocumentationRelatedPath(".github/workflows/release-documentation.yaml"), true);
  assert.equal(isDocumentationRelatedPath("tools/docs-agent/publish-release.mjs"), true);
  assert.equal(isDocumentationRelatedPath("nemoguardrails/rails/llm/config.py"), false);
  assert.equal(isReleaseDocumentationPath("docs/about/release-notes.mdx"), true);
  assert.equal(isReleaseDocumentationPath("fern/fern.config.json"), false);
  assert.equal(isReleaseSnapshotPath("fern/docs.yml"), true);
  assert.equal(isReleaseSnapshotPath("docs/README.mdx"), false);
});

test("accepts only an exact authorized review command", () => {
  assert.equal(isReviewCommand(" /review-doc\n"), true);
  assert.equal(isReviewCommand("/review-doc please"), false);
  assert.equal(isAuthorizedAssociation("MEMBER"), true);
  assert.equal(isAuthorizedAssociation("CONTRIBUTOR"), false);
  assert.equal(
    manualReviewRequest({
      comment: { author_association: "OWNER", body: "/review-doc" },
      issue: { pull_request: { url: "https://api.github.com/example" } },
    }),
    true,
  );
});

test("allows only the GitHub API endpoints owned by the documentation workflows", () => {
  assert.equal(isAllowedGithubApiPath("GET", "/repos/NVIDIA-NeMo/Guardrails/pulls/25"), true);
  assert.equal(
    isAllowedGithubApiPath("GET", "/repos/NVIDIA-NeMo/Guardrails/pulls/25/files?per_page=100&page=30"),
    true,
  );
  assert.equal(isAllowedGithubApiPath("DELETE", "/repos/NVIDIA-NeMo/Guardrails/git/refs/heads/develop"), false);
  assert.equal(isAllowedGithubApiPath("GET", "/repos/other/project/pulls/25"), false);
  assert.equal(
    isAllowedGithubApiPath(
      "GET",
      `/repos/NVIDIA-NeMo/Guardrails/pulls?state=open&base=develop&head=NVIDIA-NeMo%3Aautomation%2Frelease-docs-v0.25.0-${SHA_A.slice(0, 12)}&per_page=10`,
    ),
    true,
  );
});

test("writes bounded outputs only to the GitHub runner command directory", () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "docs-agent-output-"));
  const commands = path.join(directory, "_runner_file_commands");
  const output = path.join(commands, "set_output_1234");
  fs.mkdirSync(commands);
  fs.writeFileSync(output, "");
  try {
    workflowOutput("head_sha", SHA_A, { GITHUB_OUTPUT: output, RUNNER_TEMP: directory });
    assert.equal(fs.readFileSync(output, "utf8"), `head_sha=${SHA_A}\n`);
    assert.throws(
      () => workflowOutput("head_sha", SHA_A, { GITHUB_OUTPUT: path.join(directory, "outside"), RUNNER_TEMP: directory }),
      /outside the runner/u,
    );
  } finally {
    fs.rmSync(directory, { force: true, recursive: true });
  }
});

test("builds one managed branch per source pull request and merge", () => {
  assert.equal(managedBranch(2350, SHA_C), `automation/post-merge-docs-pr-2350-${SHA_C.slice(0, 12)}`);
  assert.equal(managedReleaseBranch("0.25.0", SHA_C), `automation/release-docs-v0.25.0-${SHA_C.slice(0, 12)}`);
  assert.equal(releaseSnapshotTag("0.25.0"), "fern-docs-snapshot-v0.25.0");
  assert.throws(() => stableVersion("0.25.0-rc1"), /stable X.Y.Z/u);
  assert.equal(exactTimestamp("2026-08-31T23:00:00Z"), "2026-08-31T23:00:00Z");
  assert.throws(() => exactTimestamp("2026-02-31T23:00:00Z"), /exact UTC timestamp/u);
});

test("requests configured maintainers and the source author without duplicates", () => {
  const maintainers = parseMaintainerLogins("maintainer-one, maintainer-two,maintainer-one");
  assert.deepEqual(requestedReviewers(maintainers, "contributor"), [
    "maintainer-one",
    "maintainer-two",
    "contributor",
  ]);
  assert.deepEqual(requestedReviewers(maintainers, "dependabot[bot]"), maintainers);
});

test("validates 100-point score arithmetic and hard gates", () => {
  assert.equal(validateQualityScore(quality()).total, 92);
  assert.throws(() => validateQualityScore(quality({ total: 91 })), /score sum/u);
  assert.throws(() => validateQualityScore(quality({ rubric: "__proto__" })), /unsupported rubric/u);
  const gated = quality({ decision: "pass" });
  gated.hard_gates.critical_factual_errors = 1;
  assert.throws(() => validateQualityScore(gated), /requires decision fail/u);
});

test("binds a review result to the exact pull request revision", () => {
  const result = {
    base_sha: SHA_A,
    findings: [],
    head_sha: SHA_B,
    pull_request: 25,
    quality: quality(),
    repository: "NVIDIA-NeMo/Guardrails",
    summary: "No concrete findings.",
    version: 1,
  };
  assert.equal(
    validateReviewResult(result, {
      baseSha: SHA_A,
      headSha: SHA_B,
      pullRequest: 25,
      repository: "NVIDIA-NeMo/Guardrails",
    }),
    result,
  );
  assert.throws(
    () =>
      validateReviewResult(result, {
        baseSha: SHA_A,
        headSha: SHA_C,
        pullRequest: 25,
        repository: "NVIDIA-NeMo/Guardrails",
      }),
    /does not match/u,
  );
});

test("selects only merged non-documentation development pull requests", () => {
  const event = {
    action: "closed",
    pull_request: {
      base: { ref: "develop", repo: { full_name: "NVIDIA-NeMo/Guardrails" }, sha: SHA_A },
      head: { ref: "feature", repo: { full_name: "contributor/Guardrails" }, sha: SHA_B },
      merge_commit_sha: SHA_C,
      merged: true,
      number: 50,
      state: "closed",
      user: { login: "contributor" },
    },
  };
  assert.equal(selectPostMerge(event, "NVIDIA-NeMo/Guardrails", ["nemoguardrails/config.py"]).automate, true);
  assert.equal(selectPostMerge(event, "NVIDIA-NeMo/Guardrails", ["docs/example.mdx"]).automate, false);
  const deletedFork = structuredClone(event);
  deletedFork.pull_request.head.repo = null;
  assert.equal(selectPostMerge(deletedFork, "NVIDIA-NeMo/Guardrails", ["nemoguardrails/config.py"]).automate, false);
});

test("routes a merged Prepare Release pull request only to release documentation", () => {
  const event = {
    action: "closed",
    pull_request: {
      base: { ref: "develop", repo: { full_name: "NVIDIA-NeMo/Guardrails" }, sha: SHA_A },
      head: { ref: "chore/release-0.25.0", repo: { full_name: "NVIDIA-NeMo/Guardrails" }, sha: SHA_B },
      labels: [{ name: "automated" }, { name: "release" }],
      merge_commit_sha: SHA_C,
      merged: true,
      merged_at: "2026-08-31T23:00:00Z",
      number: 70,
      state: "closed",
      title: "chore: prepare for release v0.25.0",
      user: { login: "github-actions[bot]" },
    },
  };
  const files = ["CHANGELOG.md", "README.md", "pyproject.toml", "uv.lock"];
  const selected = selectRelease(event, "NVIDIA-NeMo/Guardrails", files);
  assert.equal(selected.automate, true);
  assert.equal(selected.version, "0.25.0");
  assert.equal(selectPostMerge(event, "NVIDIA-NeMo/Guardrails", files).automate, false);
  const missingLabel = structuredClone(event);
  missingLabel.pull_request.labels = [{ name: "release" }];
  assert.throws(
    () => selectRelease(missingLabel, "NVIDIA-NeMo/Guardrails", files),
    /missing required release automation labels/u,
  );
});

test("release prompts enforce the release-note and immutable-version contract", () => {
  const context = {
    baseSha: SHA_A,
    headSha: SHA_B,
    mergeSha: SHA_C,
    pullRequest: 70,
    releaseVersion: "0.25.0",
  };
  const author = buildReleaseAuthorPrompt(context, "trusted guidance");
  const reviewer = buildReleaseCoverageReviewPrompt(context, "trusted guidance");
  for (const text of [author, reviewer]) {
    assert.match(text, /fern-docs-snapshot-v0[.]25[.]0/u);
    assert.match(text, /breaking change/iu);
  }
  assert.match(author, /Do not change the Fern CLI version/u);
  assert.match(reviewer, /Reject changes to any other path/u);
});

test("keeps the provider credential in one trusted configuration step", () => {
  const clean = credentialFreeEnvironment({
    DOCS_AGENT_API_KEY: "repository-secret",
    GH_TOKEN: "github-token",
    GITHUB_TOKEN: "github-token",
    HOME: "/tmp/runner-home",
    NVIDIA_API_KEY: "provider-secret",
    OPENAI_API_KEY: "provider-secret",
    PATH: "/usr/bin",
  });
  for (const name of ["DOCS_AGENT_API_KEY", "GH_TOKEN", "GITHUB_TOKEN", "NVIDIA_API_KEY", "OPENAI_API_KEY"]) {
    assert.equal(clean[name], undefined);
  }

  for (const file of [
    "post-merge-documentation.yaml",
    "release-documentation.yaml",
    "review-documentation.yaml",
  ]) {
    const source = fs.readFileSync(new URL(`../../.github/workflows/${file}`, import.meta.url), "utf8");
    assert.equal([...source.matchAll(/secrets[.]DOCS_AGENT_API_KEY/gu)].length, 1);
    const agentSteps = source.split(/\n(?=      - name:)/u).filter((step) => step.includes("tools/docs-agent/agent.mjs"));
    const configure = agentSteps.filter((step) => step.includes('agent.mjs" configure'));
    assert.equal(configure.length, 1);
    assert.match(configure[0], /OPENAI_API_KEY/u);
    for (const step of agentSteps.filter((step) => !step.includes('agent.mjs" configure'))) {
      assert.doesNotMatch(step, /OPENAI_API_KEY|DOCS_AGENT_API_KEY/u);
    }
  }

  for (const file of ["post-merge-documentation.yaml", "release-documentation.yaml"]) {
    const source = fs.readFileSync(new URL(`../../.github/workflows/${file}`, import.meta.url), "utf8");
    assert.doesNotMatch(source, /DOCS_FERN_TOKEN|FERN_TOKEN/u);
    const agent = source.indexOf(file.startsWith("post-") ? 'agent.mjs" post-merge' : 'agent.mjs" release-docs');
    const cleanup = source.indexOf('agent.mjs" cleanup');
    const validation = source.indexOf(file.startsWith("post-") ? "Validate generated documentation" : "Validate release documentation");
    assert.ok(agent > 0 && cleanup > agent && validation > cleanup);
  }
});

test("stops the inference gateway and removes its temporary state", async () => {
  const runnerTemp = fs.mkdtempSync(path.join(os.tmpdir(), "docs-agent-gateway-test-"));
  const gateway = path.join(runnerTemp, "docs-agent-gateway");
  fs.mkdirSync(gateway, { mode: 0o700 });
  const child = spawn(process.execPath, ["-e", "setInterval(() => undefined, 1000)"], {
    detached: true,
    stdio: "ignore",
  });
  child.unref();
  fs.writeFileSync(path.join(gateway, "gateway.pid"), `${child.pid}\n`, { mode: 0o600 });
  try {
    await cleanupInference({ RUNNER_TEMP: runnerTemp });
    assert.equal(fs.existsSync(gateway), false);
    assert.throws(() => process.kill(child.pid, 0), { code: "ESRCH" });
  } finally {
    try {
      process.kill(child.pid, "SIGKILL");
    } catch (error) {
      if (error?.code !== "ESRCH") throw error;
    }
    fs.rmSync(runnerTemp, { force: true, recursive: true });
  }
});

test("validates and resumes an orphaned release-documentation branch", () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "docs-agent-release-branch-"));
  const repository = path.join(directory, "repository");
  const remote = path.join(directory, "remote.git");
  const runGit = (args, options = {}) =>
    String(
      execFileSync("git", args, {
        cwd: repository,
        encoding: options.encoding ?? "utf8",
        input: options.input,
      }),
    ).trim();
  try {
    execFileSync("git", ["init", "--bare", remote]);
    execFileSync("git", ["init", "--initial-branch=develop", repository]);
    runGit(["config", "user.name", "release-test"]);
    runGit(["config", "user.email", "release-test@example.com"]);
    runGit(["config", "commit.gpgsign", "false"]);
    fs.mkdirSync(path.join(repository, "docs"), { recursive: true });
    fs.writeFileSync(path.join(repository, "docs", "README.mdx"), "before\n");
    runGit(["add", "docs/README.mdx"]);
    runGit(["commit", "--message", "chore: release baseline"]);
    const mergeSha = runGit(["rev-parse", "HEAD"]);
    runGit(["remote", "add", "origin", remote]);
    runGit(["push", "origin", "develop"]);
    const branch = `automation/release-docs-v0.25.0-${mergeSha.slice(0, 12)}`;
    runGit(["switch", "--create", branch]);
    fs.writeFileSync(path.join(repository, "docs", "README.mdx"), "after\n");
    runGit(["add", "docs/README.mdx"]);
    runGit(["config", "user.name", "github-actions[bot]"]);
    runGit(["config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com"]);
    runGit(["commit", "--signoff", "--message", "docs: publish v0.25.0 release notes and snapshot"]);
    const branchSha = runGit(["rev-parse", "HEAD"]);
    const patch = execFileSync("git", ["diff", "--binary", "--full-index", mergeSha, branchSha], {
      cwd: repository,
    });
    runGit(["push", "origin", `HEAD:refs/heads/${branch}`]);
    runGit(["switch", "develop"]);
    const remoteOutput = runGit(["ls-remote", "--heads", "origin", `refs/heads/${branch}`]);
    assert.equal(remoteBranchCommit(remoteOutput, branch), branchSha);
    assert.doesNotThrow(() =>
      validateManagedBranch(
        repository,
        branch,
        branchSha,
        { merge_sha: mergeSha, release_version: "0.25.0" },
        patch,
      ),
    );
    assert.throws(
      () =>
        validateManagedBranch(
          repository,
          branch,
          branchSha,
          { merge_sha: mergeSha, release_version: "0.25.0" },
          Buffer.from("unapproved patch\n"),
        ),
      /does not contain the approved patch/u,
    );
  } finally {
    fs.rmSync(directory, { force: true, recursive: true });
  }
});

test("runs cleanup after a failed short-lived agent operation", async () => {
  let cleaned = false;
  await assert.rejects(
    withCleanup(
      async () => {
        throw new Error("analysis failed");
      },
      async () => {
        cleaned = true;
      },
    ),
    /analysis failed/u,
  );
  assert.equal(cleaned, true);
});
