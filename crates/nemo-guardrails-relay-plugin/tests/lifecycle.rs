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

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use nemo_relay::plugin::dynamic::initialize;
use nemo_relay::plugin::{PluginConfig, list_plugin_kinds};
use serde_json::{Value as Json, json};
use sha2::{Digest, Sha256};
use tempfile::TempDir;
use tokio::sync::Mutex;

const PLUGIN_ID: &str = "nvidia.nemo_guardrails_checks";
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
            "endpoint": "http://127.0.0.1:8000",
            "config_id": "tool_checks",
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

/// The manifest's config schema rejects a config without `endpoint` before the plugin runs.
#[tokio::test]
async fn schema_rejects_config_without_endpoint() {
    let _host = HOST.lock().await;
    let bundle = Bundle::new();

    let error = initialize_error(bundle.plugins_toml(json!({"config_id": "tool_checks"}))).await;

    assert!(error.contains("configuration does not satisfy"), "{error}");
    assert!(error.contains("endpoint"), "{error}");
    assert!(!list_plugin_kinds().contains(&PLUGIN_ID.to_owned()));
}

/// The manifest's config schema rejects an empty `config_id`.
#[tokio::test]
async fn schema_rejects_empty_config_id() {
    let _host = HOST.lock().await;
    let bundle = Bundle::new();

    let error = initialize_error(bundle.plugins_toml(json!({
        "endpoint": "http://127.0.0.1:8000",
        "config_id": ""
    })))
    .await;

    assert!(error.contains("configuration does not satisfy"), "{error}");
    assert!(error.contains("config_id"), "{error}");
    assert!(!list_plugin_kinds().contains(&PLUGIN_ID.to_owned()));
}

/// The plugin's own validation rejects an endpoint that passes the schema but isn't a URL, and
/// the host unloads the plugin again.
#[tokio::test]
async fn plugin_rejects_endpoint_the_schema_allows() {
    let _host = HOST.lock().await;
    let bundle = Bundle::new();

    let error = initialize_error(bundle.plugins_toml(json!({
        "endpoint": "http://exa mple.com",
        "config_id": "tool_checks"
    })))
    .await;

    assert!(
        error.contains("invalid NeMo Guardrails checks config: endpoint is not a valid URL"),
        "{error}"
    );
    assert!(!list_plugin_kinds().contains(&PLUGIN_ID.to_owned()));
}

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

fn crate_dir() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// Builds the `cdylib` once per test run, in the same profile as the tests.
fn built_library() -> &'static Path {
    LIBRARY.get_or_init(build_library)
}

fn build_library() -> PathBuf {
    let target_dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("lifecycle");
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    let mut command = Command::new(env!("CARGO"));
    command
        .args(["build", "--locked", "--manifest-path"])
        .arg(crate_dir().join("Cargo.toml"))
        .arg("--target-dir")
        .arg(&target_dir);
    if profile == "release" {
        command.arg("--release");
    }
    let status = command.status().expect("cargo build should start");
    assert!(
        status.success(),
        "cargo build should build the plugin library"
    );

    // The NeMo-Relay-Plugins release pipeline sets CARGO_BUILD_TARGET, which moves the output
    // into a directory named after the target triple.
    let library = target_dir
        .join(std::env::var("CARGO_BUILD_TARGET").unwrap_or_default())
        .join(profile)
        .join(format!(
            "{}{LIBRARY_STEM}{}",
            std::env::consts::DLL_PREFIX,
            std::env::consts::DLL_SUFFIX
        ));
    assert!(library.is_file(), "missing {}", library.display());
    library
}

fn sha256_hex(path: &Path) -> String {
    let digest = Sha256::digest(std::fs::read(path).expect("library should be readable"));
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}
