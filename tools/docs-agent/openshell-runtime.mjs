// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { execFileSync, spawn } from "node:child_process";
import { closeSync, existsSync, mkdirSync, openSync, readFileSync, rmSync, writeFileSync } from "node:fs";
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
  try {
    return String(
      execFileSync(command, args, {
        encoding: "utf8",
        env,
        stdio: options.sensitive
          ? "ignore"
          : options.capture
            ? ["ignore", "pipe", "inherit"]
            : "inherit",
        timeout: options.timeout,
      }) ?? "",
    ).trim();
  } catch (error) {
    if (options.sensitive) fail(`Sensitive ${command} command failed`);
    throw error;
  }
}

function gatewayConfiguration(directory, supervisor, bindAddress) {
  return `[openshell]
version = 1

[openshell.gateway]
bind_address = ${JSON.stringify(bindAddress)}
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

function gatewayDirectory(env) {
  return path.join(required(env.RUNNER_TEMP, "RUNNER_TEMP"), "docs-agent-gateway");
}

export function loopbackGatewayAddress(env) {
  const endpoint = new URL(required(env.OPENSHELL_GATEWAY_ENDPOINT, "OPENSHELL_GATEWAY_ENDPOINT"));
  if (
    endpoint.protocol !== "http:" ||
    !["127.0.0.1", "[::1]"].includes(endpoint.hostname) ||
    endpoint.username ||
    endpoint.password ||
    endpoint.pathname !== "/" ||
    endpoint.search ||
    endpoint.hash
  ) {
    fail("OPENSHELL_GATEWAY_ENDPOINT must be an uncredentialed loopback HTTP origin");
  }
  return endpoint.host;
}

function processGroupExists(pid) {
  try {
    process.kill(-pid, 0);
    return true;
  } catch (error) {
    if (error?.code === "ESRCH") return false;
    throw error;
  }
}

function signalProcessGroup(pid, signal) {
  try {
    process.kill(-pid, signal);
  } catch (error) {
    if (error?.code !== "ESRCH") throw error;
  }
}

async function stopProcessGroup(pid) {
  if (!processGroupExists(pid)) return;
  signalProcessGroup(pid, "SIGTERM");
  for (let attempt = 0; attempt < 50; attempt += 1) {
    if (!processGroupExists(pid)) return;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  signalProcessGroup(pid, "SIGKILL");
  for (let attempt = 0; attempt < 50; attempt += 1) {
    if (!processGroupExists(pid)) return;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  fail(`OpenShell gateway process group ${pid} did not exit after SIGKILL`);
}

export async function cleanupInference(env) {
  const directory = gatewayDirectory(env);
  const pidFile = path.join(directory, "gateway.pid");
  try {
    if (existsSync(pidFile)) {
      const pid = Number(readFileSync(pidFile, "utf8").trim());
      if (!Number.isSafeInteger(pid) || pid < 2) fail("OpenShell gateway PID is invalid");
      await stopProcessGroup(pid);
    }
  } finally {
    rmSync(directory, { force: true, recursive: true });
  }
}

export async function configureInference(env, modelId) {
  const bindAddress = loopbackGatewayAddress(env);
  const providerApiKey = required(env.OPENAI_API_KEY, "OPENAI_API_KEY");
  const clean = credentialFreeEnvironment(env);
  const directory = gatewayDirectory(env);
  rmSync(directory, { force: true, recursive: true });
  mkdirSync(directory, { mode: 0o700, recursive: true });
  const supervisor = run("which", ["openshell-sandbox"], clean, { capture: true });
  run("openshell-gateway", ["generate-certs", "--output-dir", directory], clean);
  const configuration = path.join(directory, "gateway.toml");
  writeFileSync(configuration, gatewayConfiguration(directory, supervisor, bindAddress), { mode: 0o600 });
  const log = openSync(path.join(directory, "gateway.log"), "w", 0o600);
  let child;
  try {
    child = spawn("openshell-gateway", ["--config", configuration], {
      detached: true,
      env: clean,
      stdio: ["ignore", log, log],
    });
    child.on("error", () => undefined);
    if (!Number.isSafeInteger(child.pid) || child.pid < 2) fail("OpenShell gateway did not start");
    writeFileSync(path.join(directory, "gateway.pid"), `${child.pid}\n`, { mode: 0o600 });
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
      { sensitive: true, timeout: 60_000 },
    );
    run(
      "openshell",
      ["inference", "set", "--provider", "docs", "--model", modelId, "--timeout", "900"],
      clean,
      { timeout: 930_000 },
    );
  } catch (error) {
    try {
      if (child?.pid) await stopProcessGroup(child.pid);
    } finally {
      rmSync(directory, { force: true, recursive: true });
    }
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
