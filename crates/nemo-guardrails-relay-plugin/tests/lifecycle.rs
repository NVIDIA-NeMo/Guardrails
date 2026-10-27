// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Loads the built plugin library into a real NeMo Relay host, through the crate's own manifest
//! and config schema.
//!
//! No NeMo Guardrails service is needed: loading and validating the plugin makes no calls to it.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use nemo_relay::plugin::dynamic::{initialize, relay_version_matches};
use nemo_relay::plugin::{PluginConfig, list_plugin_kinds};
use semver::{Version, VersionReq};
use serde_json::{Value as Json, json};
use sha2::{Digest, Sha256};
use tempfile::TempDir;
use tokio::sync::Mutex;

const PLUGIN_ID: &str = "nvidia.nemo_guardrails";
const LIBRARY_STEM: &str = "nemo_guardrails_relay_plugin";

/// Relay allows one plugin host per process, so the tests take turns.
static HOST: Mutex<()> = Mutex::const_new(());
static LIBRARY: OnceLock<PathBuf> = OnceLock::new();

/// A full, valid config activates the plugin, and closing the host unloads it.
#[tokio::test]
async fn valid_config_activates_and_closes() {
    let _host = HOST.lock().await;
    let bundle = Bundle::new();

    let mut activation = initialize(
        PluginConfig::default(),
        Some(bundle.plugins_toml(json!({
            "endpoint": "http://127.0.0.1:8100",
            "headers": {"authorization": "Bearer test-token"},
            "timeout_millis": 1500,
            "priority": 0,
            "executor": {"worker_threads": 1}
        }))),
    )
    .await
    .expect("a valid config should activate the plugin");
    assert!(
        !activation.report().config.has_errors(),
        "{:?}",
        activation.report()
    );
    assert!(list_plugin_kinds().contains(&PLUGIN_ID.to_owned()));

    activation.close().expect("the host should close");
    assert!(!list_plugin_kinds().contains(&PLUGIN_ID.to_owned()));
}

/// An endpoint whose scheme is in capitals activates the plugin, which reads the scheme as a URL
/// parser does.
#[tokio::test]
async fn uppercase_scheme_activates() {
    let _host = HOST.lock().await;
    let bundle = Bundle::new();

    let mut activation = initialize(
        PluginConfig::default(),
        Some(bundle.plugins_toml(json!({"endpoint": "HTTPS://127.0.0.1:8100"}))),
    )
    .await
    .expect("an endpoint with an uppercase scheme should activate the plugin");
    assert!(list_plugin_kinds().contains(&PLUGIN_ID.to_owned()));

    activation.close().expect("the host should close");
}

/// The manifest's config schema rejects a config without `endpoint` before the plugin runs.
#[tokio::test]
async fn schema_rejects_config_without_endpoint() {
    let _host = HOST.lock().await;
    let bundle = Bundle::new();

    let error = initialize_error(bundle.plugins_toml(json!({"timeout_millis": 1500}))).await;

    assert!(error.contains("configuration does not satisfy"), "{error}");
    assert!(error.contains("endpoint"), "{error}");
    assert!(!list_plugin_kinds().contains(&PLUGIN_ID.to_owned()));
}

/// The manifest's config schema rejects a field it doesn't declare, such as `config_id`: the
/// NeMo Guardrails service serves one configuration, so the plugin can't pick one.
#[tokio::test]
async fn schema_rejects_unknown_field() {
    let _host = HOST.lock().await;
    let bundle = Bundle::new();

    let error = initialize_error(bundle.plugins_toml(json!({
        "endpoint": "http://127.0.0.1:8100",
        "config_id": "tool_checks"
    })))
    .await;

    assert!(error.contains("configuration does not satisfy"), "{error}");
    assert!(error.contains("config_id"), "{error}");
    assert!(!list_plugin_kinds().contains(&PLUGIN_ID.to_owned()));
}

/// The manifest's config schema rejects a `priority` outside the plugin's 32-bit range.
#[tokio::test]
async fn schema_rejects_priority_outside_32_bits() {
    let _host = HOST.lock().await;
    let bundle = Bundle::new();

    let error = initialize_error(bundle.plugins_toml(json!({
        "endpoint": "http://127.0.0.1:8100",
        "priority": 3_000_000_000_i64
    })))
    .await;

    assert!(error.contains("configuration does not satisfy"), "{error}");
    assert!(error.contains("priority"), "{error}");
    assert!(!list_plugin_kinds().contains(&PLUGIN_ID.to_owned()));
}

/// The manifest's `compat.relay` refuses Relay 0.10.0, whose CLI hooks let a call run when a
/// guardrail returns an error, and accepts 0.10.1 and later 0.10 releases.
#[test]
fn manifest_refuses_relay_hosts_before_0_10_1() {
    let manifest: toml::Table = toml::from_str(
        &std::fs::read_to_string(crate_dir().join("relay-plugin.toml"))
            .expect("manifest template should be readable"),
    )
    .expect("manifest template should be TOML");
    let requirement: VersionReq = manifest["compat"]["relay"]
        .as_str()
        .expect("compat.relay should be a string")
        .parse()
        .expect("compat.relay should be a version requirement");

    for (version, accepted) in [("0.10.0", false), ("0.10.1", true), ("0.10.4", true)] {
        let version = Version::parse(version).expect("test version should parse");
        assert_eq!(
            relay_version_matches(&requirement, &version),
            accepted,
            "{version}"
        );
    }
}

/// The plugin's own validation rejects endpoints the schema allows, without repeating a password,
/// and the host unloads the plugin again.
#[tokio::test]
async fn plugin_rejects_endpoints_the_schema_allows() {
    let _host = HOST.lock().await;

    // The schema leaves the scheme and credentials to the plugin, because Relay's schema errors
    // quote the value: a scheme pattern would quote a URL such as HTTPS://user:secret@host.
    for (endpoint, message) in [
        ("http://exa mple.com", "endpoint is not a valid URL"),
        (
            "https://user:secret@gw.example.com",
            "endpoint must not contain a username or password",
        ),
        (
            "HTTPS://user:secret@gw.example.com",
            "endpoint must not contain a username or password",
        ),
        (
            "ftp://user:secret@gw.example.com",
            "endpoint must use http or https",
        ),
    ] {
        let bundle = Bundle::new();
        let error = initialize_error(bundle.plugins_toml(json!({"endpoint": endpoint}))).await;

        assert!(
            error.contains(&format!("invalid NeMo Guardrails plugin config: {message}")),
            "{endpoint}: {error}"
        );
        assert!(!error.contains("secret"), "{endpoint}: {error}");
        assert!(!list_plugin_kinds().contains(&PLUGIN_ID.to_owned()));
    }
}

/// Headers of the wrong type are rejected by the plugin, not the schema, so the host's error
/// doesn't quote a value that may be a credential.
#[tokio::test]
async fn plugin_rejects_mistyped_headers_without_quoting_them() {
    let _host = HOST.lock().await;

    for (headers, message, value) in [
        (
            json!({"x-api-key": 8_675_309_123_u64}),
            "headers[x-api-key] must be a string",
            "8675309123",
        ),
        (
            json!("Bearer secret-token"),
            "headers must be an object",
            "secret-token",
        ),
    ] {
        let bundle = Bundle::new();
        let error = initialize_error(bundle.plugins_toml(json!({
            "endpoint": "http://127.0.0.1:8100",
            "headers": headers
        })))
        .await;

        assert!(
            error.contains(&format!("invalid NeMo Guardrails plugin config: {message}")),
            "{error}"
        );
        assert!(!error.contains(value), "{error}");
        assert!(!list_plugin_kinds().contains(&PLUGIN_ID.to_owned()));
    }
}

/// The error the host refuses a `plugins.toml` with.
async fn initialize_error(plugins_toml: PathBuf) -> String {
    match initialize(PluginConfig::default(), Some(plugins_toml)).await {
        Ok(_) => panic!("the host should refuse the config"),
        Err(error) => error.to_string(),
    }
}

/// A plugin bundle laid out as `scripts/package_bundle.py` writes it.
struct Bundle {
    directory: TempDir,
}

impl Bundle {
    /// Packages the built library with the crate's manifest and schema in a temporary directory.
    fn new() -> Self {
        let directory = TempDir::new().expect("bundle directory should be created");
        let library = built_library();
        let library_name = library
            .file_name()
            .and_then(|name| name.to_str())
            .expect("library file name is UTF-8");
        let artifact = directory.path().join(library_name);
        std::fs::copy(library, &artifact).expect("library should copy into the bundle");
        std::fs::copy(
            crate_dir().join("config.schema.json"),
            directory.path().join("config.schema.json"),
        )
        .expect("config schema should copy into the bundle");
        let manifest = std::fs::read_to_string(crate_dir().join("relay-plugin.toml"))
            .expect("manifest template should be readable")
            .replace("<platform-library-file>", library_name)
            .replace("<artifact-sha256>", &sha256_hex(&artifact));
        std::fs::write(directory.path().join("relay-plugin.toml"), manifest)
            .expect("manifest should write");
        Self { directory }
    }

    /// Writes a `plugins.toml` that loads this bundle with one component config.
    fn plugins_toml(&self, config: Json) -> PathBuf {
        let manifest = self.directory.path().join("relay-plugin.toml");
        let document = json!({
            "version": 1,
            "plugins": {
                "policy": {
                    "defaults": {"startup": "required", "attestation": "integrity_only"}
                },
                "dynamic": [{"manifest": manifest, "config": config}]
            }
        });
        let path = self.directory.path().join("plugins.toml");
        let text = toml::to_string(&document).expect("plugins.toml should serialize");
        std::fs::write(&path, text).expect("plugins.toml should write");
        path
    }
}

/// The crate's own directory, which holds the manifest and schema.
fn crate_dir() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// Builds the `cdylib` once per test run, in the same profile as the tests.
fn built_library() -> &'static Path {
    LIBRARY.get_or_init(build_library)
}

/// Builds the plugin library and returns the path Cargo reports for it.
fn build_library() -> PathBuf {
    let mut command = Command::new(env!("CARGO"));
    command
        .args([
            "build",
            "--locked",
            "--message-format=json-render-diagnostics",
            "--manifest-path",
        ])
        .arg(crate_dir().join("Cargo.toml"))
        .arg("--target-dir")
        .arg(Path::new(env!("CARGO_TARGET_TMPDIR")).join("lifecycle"));
    if !cfg!(debug_assertions) {
        command.arg("--release");
    }
    let output = command.output().expect("cargo build should start");
    assert!(
        output.status.success(),
        "cargo build should build the plugin library:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Cargo reports where it wrote the library, which already accounts for a target chosen by
    // CARGO_BUILD_TARGET (as the NeMo-Relay-Plugins pipeline does), `build.target` or `--target`.
    let messages = String::from_utf8(output.stdout).expect("cargo's messages should be UTF-8");
    let library = messages
        .lines()
        .filter_map(|line| serde_json::from_str::<Json>(line).ok())
        .filter(|message| {
            message["reason"] == "compiler-artifact" && message["target"]["name"] == LIBRARY_STEM
        })
        .filter_map(|message| message["filenames"].as_array().cloned())
        .flatten()
        .filter_map(|filename| filename.as_str().map(str::to_owned))
        .find(|filename| filename.ends_with(std::env::consts::DLL_SUFFIX))
        .map(PathBuf::from)
        .expect("cargo should report the plugin library");
    assert!(library.is_file(), "missing {}", library.display());
    library
}

/// The file's SHA-256 digest, as lowercase hex.
fn sha256_hex(path: &Path) -> String {
    let digest = Sha256::digest(std::fs::read(path).expect("library should be readable"));
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}
