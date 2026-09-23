import { readFileSync } from "node:fs";
import path from "node:path";

export function createFernRefSdkEnvironment(repoRoot, baseEnvironment = process.env) {
  const environment = { ...baseEnvironment };
  const rawConfigCount = environment.GIT_CONFIG_COUNT ?? "0";
  const configCount = Number.parseInt(rawConfigCount, 10);
  if (!Number.isSafeInteger(configCount) || configCount < 0 || String(configCount) !== rawConfigCount) {
    throw new Error(`Invalid GIT_CONFIG_COUNT: ${rawConfigCount}`);
  }

  environment.GIT_CONFIG_COUNT = String(configCount + 1);
  environment[`GIT_CONFIG_KEY_${configCount}`] = "core.hooksPath";
  environment[`GIT_CONFIG_VALUE_${configCount}`] = path.join(repoRoot, "scripts", "fern-ref-sdk-hooks");
  environment.FERN_REF_SDK_HOOK = "1";
  environment.FERN_REF_SDK_NODE = process.execPath;
  environment.FERN_REF_SDK_HELPER = path.join(repoRoot, "scripts", "cache-fern-ref-sdk.mjs");
  environment.FERN_REF_SDK_REPO_ROOT = repoRoot;
  environment.FERN_REF_SDK_CACHE_ROOT = path.join(repoRoot, ".fern-cache", "fern-ref-sdk");
  environment.FERN_REF_SDK_VERSION = JSON.parse(
    readFileSync(path.join(repoRoot, "fern", "fern.config.json"), "utf8"),
  ).version;
  return environment;
}
