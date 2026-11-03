import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { copyFileSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");

test("Fern converts the generated proxy operations and examples", { timeout: 240_000 }, () => {
  const directory = mkdtempSync(path.join(os.tmpdir(), "guardrails-proxy-fern-"));
  try {
    const fern = path.join(directory, "fern");
    mkdirSync(fern);
    const environment = { ...process.env };
    delete environment.FERN_TOKEN;
    delete environment.DOCS_FERN_TOKEN;
    const generation = spawnSync("uv", ["run", "--locked", "--extra", "server", "python",
      "scripts/export_proxy_openapi.py", "--output", path.join(fern, "proxy-openapi.yml")], {
      cwd: root, env: environment, encoding: "utf8", timeout: 90_000,
    });
    assert.equal(generation.status, 0, `${generation.error ?? ""}\n${generation.stdout}\n${generation.stderr}`);
    for (const file of ["fern.config.json", "proxy-openapi-overlay.yml"]) {
      copyFileSync(path.join(root, "fern", file), path.join(fern, file));
    }
    writeFileSync(path.join(fern, "generators.yml"), "api:\n  specs:\n    - openapi: ./proxy-openapi.yml\n      overlays: ./proxy-openapi-overlay.yml\n");
    const output = path.join(directory, "ir.json");
    const { version } = JSON.parse(readFileSync(path.join(fern, "fern.config.json"), "utf8"));
    const result = spawnSync("npx", ["--yes", `fern-api@${version}`, "ir", output], {
      cwd: fern, env: environment, encoding: "utf8", timeout: 110_000,
    });
    assert.equal(result.status, 0, `${result.error ?? ""}\n${result.stdout}\n${result.stderr}`);
    const ir = JSON.parse(readFileSync(output, "utf8"));
    const endpoints = new Map(Object.values(ir.services).flatMap(({ endpoints }) => endpoints.map((endpoint) => [endpoint.id, endpoint])));
    for (const [id, request, pathName] of [
      ["endpoint_openai.chatCompletionsCreate", { model: "example-model", messages: [{ role: "user", content: "Hello!" }] }, "v1/chat/completions"],
      ["endpoint_openai.responsesCreate", { model: "example-model", input: "Hello!", store: false }, "v1/responses"],
    ]) {
      const endpoint = endpoints.get(id);
      assert.ok(endpoint, id);
      assert.equal(endpoint.method, "POST");
      assert.equal(endpoint.fullPath.head, pathName);
      assert.deepEqual(endpoint.userSpecifiedExamples[0].example.request.jsonExample, request);
      assert.notDeepEqual(endpoint.userSpecifiedExamples[0].example.response.value.value.jsonExample, { key: "value" });
      assert.match(endpoint.docs, /provider/);
      assert.doesNotMatch(endpoint.docs, /### /);
      assert.match(endpoint.docs, /guard-contracts#read-the-field-policy/);
      assert.match(endpoint.docs, /providers-and-streaming#coverage-of-the-shipped-profiles/);
      assert.match(endpoint.docs, /providers-and-streaming#completion-and-errors/);
      const contractSlug = pathName.endsWith("responses") ? "responses" : "chat-completions";
      for (const section of ["request", "response", "stream"]) {
        assert.ok(endpoint.docs.includes(`/reference/proxy-guard-contracts/${contractSlug}#${section}`));
      }
      const contractPage = readFileSync(path.join(fern, "proxy-guard-contracts", `${contractSlug}.mdx`), "utf8");
      for (const heading of ["Request", "Response", "Stream", "Exact contract"]) {
        assert.ok(contractPage.includes(`## ${heading}`));
      }
      const properties = new Map(endpoint.requestBody.properties.map((property) => [property.name, property]));
      for (const name of ["model", "temperature"]) {
        assert.match(properties.get(name).docs, /<Badge intent="note" minimal>Opaque<\/Badge>/);
        assert.match(properties.get(name).docs, /Forwarded without inspecting or validating/);
        assert.match(properties.get(name).docs, /href="\/run-guardrailed-inference\/using-proxy\/guard-contracts#read-the-field-policy"/);
        assert.doesNotMatch(properties.get(name).docs, /guardrails-proxy-null-only/);
      }
      for (const name of id.endsWith("responsesCreate") ? ["instructions", "tools", "conversation"] : ["tools", "audio"]) {
        assert.match(properties.get(name).docs, /<Badge intent="danger" minimal>Disabled<\/Badge>/);
        assert.match(properties.get(name).docs, /guardrails-proxy-null-only/);
        assert.match(properties.get(name).docs, /Omit this field or set it to `null`/);
        assert.match(properties.get(name).docs, /non-null values are rejected/);
      }
      if (id.endsWith("responsesCreate")) {
        assert.match(properties.get("input").docs, /<Badge intent="success" minimal>Guarded<\/Badge>/);
        assert.match(properties.get("store").docs, /<Badge intent="warning" minimal>Constrained<\/Badge>/);
      }
      assert.equal(endpoint.errors.length, 6);
    }
    const reasoning = ir.types["type_openai:OpenaiResponsesCreateRequestReasoning"].shape;
    assert.match(reasoning.properties.find((property) => property.name === "summary").docs, /guardrails-proxy-null-only/);
    const message = ir.types["type_openai:OpenaiChatCompletionsCreateRequestMessagesItem"].shape;
    assert.match(message.properties.find((property) => property.name === "content").docs, />Guarded<\/Badge>/);
    assert.match(message.properties.find((property) => property.name === "name").docs, /guardrails-proxy-null-only/);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});
