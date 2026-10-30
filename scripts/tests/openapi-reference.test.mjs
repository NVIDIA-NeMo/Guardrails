import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { parse } from "yaml";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const baseline = JSON.parse(readFileSync(new URL("./fixtures/openapi-reference.json", import.meta.url), "utf8"));

function profile(endpoint) {
  return {
    name: endpoint.name,
    displayName: endpoint.displayName,
    method: endpoint.method,
    fullPath: endpoint.fullPath,
    docs: endpoint.docs,
    examples: endpoint.userSpecifiedExamples.map(({ example }) => ({
      request: example.request?.jsonExample ?? null,
      response: example.response?.value?.value?.jsonExample ?? null,
    })),
    errors: endpoint.errors.map(({ error }) => error.name),
  };
}

test("Fern integration preserves existing API reference routes and curated examples", { timeout: 240_000 }, () => {
  const directory = mkdtempSync(path.join(os.tmpdir(), "guardrails-openapi-reference-"));
  try {
    const generation = spawnSync("make", ["docs-fern-generate-openapi"], {
      cwd: root,
      encoding: "utf8",
      timeout: 90_000,
    });
    assert.equal(generation.status, 0, `${generation.error ?? ""}\n${generation.stdout}\n${generation.stderr}`);
    const output = path.join(directory, "api-ir.json");
    const { version } = JSON.parse(readFileSync(path.join(root, "fern/fern.config.json"), "utf8"));
    const result = spawnSync("npx", ["--yes", `fern-api@${version}`, "ir", output], {
      cwd: path.join(root, "fern"),
      encoding: "utf8",
      timeout: 110_000,
    });
    assert.equal(result.status, 0, `${result.error ?? ""}\n${result.stdout}\n${result.stderr}`);
    const ir = JSON.parse(readFileSync(output, "utf8"));
    const endpoints = new Map(
      Object.values(ir.services).flatMap(({ endpoints }) => endpoints.map((endpoint) => [endpoint.id, endpoint])),
    );
    // This baseline records the reviewed API routes, descriptions, and curated examples.
    for (const [id, expected] of Object.entries(baseline.endpoints)) {
      assert.ok(endpoints.has(id), `Missing API reference endpoint: ${id}`);
      assert.deepEqual(profile(endpoints.get(id)), expected, `Changed API reference: ${id}`);
    }
    assert.ok(endpoints.has("endpoint_checks.checkGuardrails"));
    const checks = profile(endpoints.get("endpoint_checks.checkGuardrails"));
    assert.deepEqual(checks.examples, [{
      request: {
        model: "nvidia/nemotron-3.5-lightning-30b-a3b",
        messages: [{ role: "user", content: "What is the capital of France?" }],
        guardrails: { config_id: "nemotron-3.5-content-safety", rail_types: ["input"] },
      },
      response: { status: "passed", content: "What is the capital of France?" },
    }]);

    const specPath = path.join(directory, "published-api.yml");
    const specExport = spawnSync("npx", ["--yes", `fern-api@${version}`, "export", specPath], {
      cwd: path.join(root, "fern"),
      encoding: "utf8",
      timeout: 110_000,
    });
    assert.equal(specExport.status, 0, specExport.stdout + specExport.stderr);
    const spec = parse(readFileSync(specPath, "utf8"));
    assert.deepEqual(spec.components.schemas.ErrorResponse.required, ["error"]);
    assert.deepEqual(spec.components.schemas.RailsConfigInfo.required, ["id"]);
    assert.equal(spec.paths["/v1/rails/configs"].get.responses["200"].content["application/json"].schema.items.$ref,
      "#/components/schemas/RailsConfigInfo");
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});
