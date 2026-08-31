// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { execFileSync, spawn } from "node:child_process";
import { closeSync, mkdirSync, openSync, writeFileSync } from "node:fs";
import path from "node:path";

import { fail } from "./contract.mjs";

const HOST_CREDENTIALS = [
  "DOCS_AGENT_API_KEY",
  "GH_TOKEN",
  "GITHUB_TOKEN",
  "NVIDIA_API_KEY",
  "OPENAI_API_KEY",
];

function required(value, name) {
  return value || fail(`${name} is required`);
}

export function credentialFreeEnvironment(env) {
  const result = { ...env };
  const binaryDirectory = env.XDG_BIN_HOME ?? path.join(required(env.HOME, "HOME"), ".local", "bin");
  result.PATH = [binaryDirectory, env.PATH ?? ""].filter(Boolean).join(path.delimiter);
  for (const name of HOST_CREDENTIALS) delete result[name];
  return result;
}

function run(command, args, env, options = {}) {
  return String(
    execFileSync(command, args, {
      encoding: "utf8",
      env,
      stdio: options.capture ? ["ignore", "pipe", "inherit"] : "inherit",
      timeout: options.timeout,
    }) ?? "",
  ).trim();
}

function gatewayConfiguration(directory, supervisor) {
  return `[openshell]
version = 1

[openshell.gateway]
bind_address = "127.0.0.1:8080"
compute_drivers = ["docker"]
disable_tls = true

[openshell.gateway.auth]
allow_unauthenticated_users = true

[openshell.gateway.gateway_jwt]
signing_key_path = ${JSON.stringify(path.join(directory, "jwt", "signing.pem"))}
public_key_path = ${JSON.stringify(path.join(directory, "jwt", "public.pem"))}
kid_path = ${JSON.stringify(path.join(directory, "jwt", "kid"))}
gateway_id = "guardrails-docs-agent"
ttl_secs = 3600

[openshell.drivers.docker]
grpc_endpoint = "http://host.openshell.internal:8080"
supervisor_bin = ${JSON.stringify(supervisor)}
enable_bind_mounts = true
`;
}

async function stopProcess(child) {
  if (child.exitCode !== null || child.signalCode !== null) return;
  child.kill("SIGTERM");
  const exited = await Promise.race([
    new Promise((resolve) => child.once("exit", () => resolve(true))),
    new Promise((resolve) => setTimeout(() => resolve(false), 5000)),
  ]);
  if (!exited) child.kill("SIGKILL");
}

export async function configureInference(env, modelId) {
  const providerApiKey = required(env.OPENAI_API_KEY, "OPENAI_API_KEY");
  const clean = credentialFreeEnvironment(env);
  const gatewayDirectory = path.join(required(env.RUNNER_TEMP, "RUNNER_TEMP"), "docs-agent-gateway");
  mkdirSync(gatewayDirectory, { recursive: true });
  const supervisor = run("which", ["openshell-sandbox"], clean, { capture: true });
  run("openshell-gateway", ["generate-certs", "--output-dir", gatewayDirectory], clean);
  const configuration = path.join(gatewayDirectory, "gateway.toml");
  writeFileSync(configuration, gatewayConfiguration(gatewayDirectory, supervisor), { mode: 0o600 });
  const log = openSync(path.join(gatewayDirectory, "gateway.log"), "w", 0o600);
  let child;
  try {
    child = spawn("openshell-gateway", ["--config", configuration], {
      detached: true,
      env: clean,
      stdio: ["ignore", log, log],
    });
    child.on("error", () => undefined);
    child.unref();
  } finally {
    closeSync(log);
  }
  try {
    let ready = false;
    for (let attempt = 0; attempt < 30; attempt += 1) {
      try {
        run("openshell", ["gateway", "info"], clean, { timeout: 10_000 });
        ready = true;
        break;
      } catch {
        await new Promise((resolve) => setTimeout(resolve, 1000));
      }
    }
    if (!ready) fail("OpenShell gateway did not become ready");
    run(
      "openshell",
      [
        "provider",
        "create",
        "--name",
        "docs",
        "--type",
        "openai",
        "--credential",
        "OPENAI_API_KEY",
        "--config",
        "OPENAI_BASE_URL=https://inference-api.nvidia.com/v1",
      ],
      { ...clean, OPENAI_API_KEY: providerApiKey },
      { timeout: 60_000 },
    );
    run(
      "openshell",
      ["inference", "set", "--provider", "docs", "--model", modelId, "--timeout", "900"],
      clean,
      { timeout: 930_000 },
    );
  } catch (error) {
    await stopProcess(child);
    throw error;
  }
}

export function createSandbox(env, input) {
  const upload = input.uploads.flatMap(({ source, destination }) => ["--upload", `${source}:${destination}`]);
  const driver = input.driverConfig ? ["--driver-config-json", JSON.stringify(input.driverConfig)] : [];
  run(
    "openshell",
    [
      "sandbox",
      "create",
      "--name",
      input.name,
      "--from",
      input.image,
      ...driver,
      "--policy",
      input.policy,
      ...upload,
      ...(upload.length ? ["--no-git-ignore"] : []),
      "--no-tty",
      "--",
      ...input.command,
    ],
    credentialFreeEnvironment(env),
  );
}

export function execSandbox(env, input) {
  const environment = Object.entries(input.environment ?? {}).flatMap(([name, value]) => {
    if (!/^[A-Z_][A-Z0-9_]*$/u.test(name) || /[\0\r\n]/u.test(value)) fail(`Unsafe sandbox environment: ${name}`);
    return ["--env", `${name}=${value}`];
  });
  run(
    "openshell",
    [
      "sandbox",
      "exec",
      "--name",
      input.name,
      "--timeout",
      String(input.timeoutSeconds ?? 1200),
      ...(input.workdir ? ["--workdir", input.workdir] : []),
      ...environment,
      "--",
      ...input.command,
    ],
    credentialFreeEnvironment(env),
  );
}

export function downloadSandboxPath(env, name, source, destination) {
  run("openshell", ["sandbox", "download", name, source, destination], credentialFreeEnvironment(env));
}

export function deleteSandbox(env, name) {
  const clean = credentialFreeEnvironment(env);
  let present = true;
  try {
    present = run("openshell", ["sandbox", "list", "--names"], clean, { capture: true })
      .split(/\r?\n/u)
      .includes(name);
  } catch {
    present = true;
  }
  if (present) run("openshell", ["sandbox", "delete", name], clean);
}
