import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { copyFileSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");

test("Fern retains all hooks routes and the object-valued request examples", { timeout: 240_000 }, () => {
  const directory = mkdtempSync(path.join(os.tmpdir(), "guardrails-hooks-fern-"));
  try {
    const fern = path.join(directory, "fern");
    mkdirSync(fern);
    const environment = { ...process.env };
    delete environment.FERN_TOKEN;
    delete environment.DOCS_FERN_TOKEN;
    const generation = spawnSync("uv", ["run", "--locked", "--extra", "server", "python",
      "scripts/export_hooks_openapi.py", "--output", path.join(fern, "hooks-openapi.yml")], {
      cwd: root, env: environment, encoding: "utf8", timeout: 90_000,
    });
    assert.equal(generation.status, 0, `${generation.error ?? ""}\n${generation.stdout}\n${generation.stderr}`);
    for (const file of ["fern.config.json", "hooks-openapi-overlay.yml"]) {
      copyFileSync(path.join(root, "fern", file), path.join(fern, file));
    }
    writeFileSync(path.join(fern, "generators.yml"), `api:\n  specs:\n    - openapi: ./hooks-openapi.yml\n      overlays: ./hooks-openapi-overlay.yml\n`);
    const output = path.join(directory, "ir.json");
    const { version } = JSON.parse(readFileSync(path.join(fern, "fern.config.json"), "utf8"));
    const result = spawnSync("npx", ["--yes", `fern-api@${version}`, "ir", output], {
      cwd: fern, encoding: "utf8", timeout: 110_000,
    });
    assert.equal(result.status, 0, `${result.error ?? ""}\n${result.stdout}\n${result.stderr}`);
    const ir = JSON.parse(readFileSync(output, "utf8"));
    const endpoints = new Map(Object.values(ir.services).flatMap(({ endpoints }) => endpoints.map((e) => [e.id, e])));
    assert.deepEqual([...endpoints.keys()].sort(), [
      "endpoint_hooks.checkClaudeCodeToolCall",
      "endpoint_hooks.checkCodexToolCall", "endpoint_hooks.checkNemoRelayToolCall",
    ].sort());
    for (const [id, input, deny, allow] of [
      ["endpoint_hooks.checkClaudeCodeToolCall", { command: "git push --force origin main" },
        { hookSpecificOutput: { hookEventName: "PreToolUse", permissionDecision: "deny", permissionDecisionReason: "NeMo Guardrails: regex check tool call" } }, {}],
      ["endpoint_hooks.checkCodexToolCall", { command: "rm -rf /tmp/build" },
        { hookSpecificOutput: { hookEventName: "PreToolUse", permissionDecision: "deny", permissionDecisionReason: "NeMo Guardrails: regex check tool call" } }, {}],
      ["endpoint_hooks.checkNemoRelayToolCall", { query: "" },
        { decision: "deny", reason: "regex check tool call" }, { decision: "allow" }],
    ]) {
      const examples = endpoints.get(id).userSpecifiedExamples.map(({ example }) => example);
      const example = examples[0];
      assert.deepEqual(example.request.jsonExample.tool_input, input, id);
      assert.deepEqual(example.response.value.value.jsonExample, deny, `${id} deny`);
      assert.deepEqual(examples[1].response.value.value.jsonExample, allow, `${id} allow`);
      assert.match(endpoints.get(id).docs, /Retry-After: 30/);
      assert.match(endpoints.get(id).requestBody.properties.find(({ name }) => name === "tool_name").docs, /must not be empty/);
    }
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});
