import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { copyFileSync, mkdirSync, mkdtempSync, readFileSync, rmSync, utimesSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";

// Stub external tools while executing the repository's actual Make dependencies.
const toolStub = `#!${process.execPath}
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const root = process.env.GUARDRAILS_PUBLISH_TEST_ROOT;
const fernVersion = JSON.parse(fs.readFileSync(path.join(root, "fern/fern.config.json"), "utf8")).version;
const tool = path.basename(process.argv[1]);
const args = process.argv.slice(2);
const record = (event) => fs.appendFileSync(path.join(root, "events"), event + "\\n");
const fail = (stage) => {
  if (process.env.GUARDRAILS_PUBLISH_TEST_FAILURE === stage) process.exit(42);
};
if (tool === "uv") {
  assert.deepEqual(args.slice(0, 5), ["run", "--locked", "--extra", "server", "python"]);
  const spec = args[5] === "scripts/export_openapi.py" ? "openapi" : "proxy-openapi";
  assert.equal(args[5], "scripts/export_" + spec.replace("-", "_") + ".py");
  assert.equal(process.env.FERN_TOKEN, undefined);
  assert.equal(process.env.DOCS_FERN_TOKEN, undefined);
  record(spec);
  fail(spec);
  fs.writeFileSync(path.join(root, "fern/" + spec + ".yml"), "fresh");
} else if (tool === "npx") {
  assert.equal(fs.readFileSync(path.join(root, "fern/openapi.yml"), "utf8"), "fresh");
  assert.equal(fs.readFileSync(path.join(root, "fern/proxy-openapi.yml"), "utf8"), "fresh");
  assert.deepEqual(args, ["--yes", "fern-api@" + fernVersion, "docs", "md", "generate", "--library", "guardrails-python-sdk"]);
  record("sdk");
  fail("sdk");
  fs.writeFileSync(path.join(root, "sdk"), "fresh");
} else if (tool === "node" && args[0] === "-p") {
  process.stdout.write(fernVersion);
} else if (tool === "node" && args[0] === "scripts/normalize-fern-sdk-reference.mjs") {
  assert.equal(fs.readFileSync(path.join(root, "sdk"), "utf8"), "fresh");
  record("normalize");
  fs.writeFileSync(path.join(root, "normalized"), "fresh");
} else if (tool === "node" && args[0] === "scripts/run-fern-with-ref-sdk.mjs") {
  assert.deepEqual(args.slice(1, 4), ["generate", "--docs", "--instance"]);
  assert.equal(fs.readFileSync(path.join(root, "fern/openapi.yml"), "utf8"), "fresh");
  assert.equal(fs.readFileSync(path.join(root, "fern/proxy-openapi.yml"), "utf8"), "fresh");
  assert.equal(fs.readFileSync(path.join(root, "normalized"), "utf8"), "fresh");
  record("publish:" + args[4]);
} else {
  throw new Error("Unexpected external command: " + tool + " " + args.join(" "));
}
`;

function fixture() {
  const root = mkdtempSync(path.join(os.tmpdir(), "guardrails-docs-publish-"));
  mkdirSync(path.join(root, "bin"));
  mkdirSync(path.join(root, "fern"));
  mkdirSync(path.join(root, "node_modules"));
  copyFileSync(new URL("../../Makefile", import.meta.url), path.join(root, "Makefile"));
  copyFileSync(new URL("../../fern/fern.config.json", import.meta.url), path.join(root, "fern/fern.config.json"));
  for (const name of ["package.json", "package-lock.json", "node_modules/.package-lock.json"]) {
    writeFileSync(path.join(root, name), "{}");
    const timestamp = name.startsWith("node_modules") ? 1_000_000_001 : 1_000_000_000;
    utimesSync(path.join(root, name), timestamp, timestamp);
  }
  for (const tool of ["uv", "node", "npx", "npm"]) {
    writeFileSync(path.join(root, "bin", tool), toolStub, { mode: 0o755 });
  }
  // A file with the target's name must not disable a phony generation step.
  writeFileSync(path.join(root, "docs-fern-generate-openapi"), "");
  return root;
}

for (const destination of ["staging", "public"]) {
  for (const failure of ["", "openapi", "proxy-openapi", "sdk"]) {
    test(`${destination} publishing ${failure ? `stops when ${failure} fails` : "regenerates both references every time"}`, () => {
      const root = fixture();
      try {
        const instance = `test-${destination}.docs.buildwithfern.com/nemo/guardrails`;
        const runs = failure ? 1 : 2;
        for (let index = 0; index < runs; index += 1) {
          if (index === 0) {
            rmSync(path.join(root, "fern/openapi.yml"), { force: true });
          } else {
            writeFileSync(path.join(root, "fern/openapi.yml"), "stale");
          }
          rmSync(path.join(root, "normalized"), { force: true });
          const result = spawnSync("make", [
            "--no-print-directory", "-j2", `docs-fern-publish-${destination}`,
            `FERN_${destination.toUpperCase()}_INSTANCE=${instance}`,
          ], {
            cwd: root,
            encoding: "utf8",
            timeout: 15_000,
            env: {
              ...process.env,
              PATH: `${path.join(root, "bin")}${path.delimiter}${process.env.PATH}`,
              GUARDRAILS_PUBLISH_TEST_ROOT: root,
              GUARDRAILS_PUBLISH_TEST_FAILURE: failure,
              FERN_TOKEN: "test-publishing-credential",
              DOCS_FERN_TOKEN: "test-publishing-credential",
            },
          });
          assert.equal(result.error, undefined);
          if (failure) {
            assert.notEqual(result.status, 0, result.stdout + result.stderr);
          } else {
            assert.equal(result.status, 0, result.stdout + result.stderr);
          }
        }
        const events = readFileSync(path.join(root, "events"), "utf8").trim().split("\n");
        if (failure) {
          assert.ok(events.includes(failure));
          assert.ok(!events.some((event) => event.startsWith("publish:")));
        } else {
          for (const event of ["openapi", "proxy-openapi", "sdk", "normalize", `publish:${instance}`]) {
            assert.equal(events.filter((value) => value === event).length, runs, event);
          }
          assert.equal(events[4], `publish:${instance}`);
          assert.equal(events[9], `publish:${instance}`);
        }
      } finally {
        rmSync(root, { recursive: true, force: true });
      }
    });
  }
}
