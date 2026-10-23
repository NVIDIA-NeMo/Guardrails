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

//! Helpers shared by the tests that load the built plugin into a real NeMo Relay host.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use serde_json::{Value as Json, json};
use sha2::{Digest, Sha256};
use tempfile::TempDir;
use tokio::sync::Mutex;

const LIBRARY_STEM: &str = "nemo_guardrails_relay_plugin";

/// Relay allows one plugin host per process, so the tests take turns.
pub static HOST: Mutex<()> = Mutex::const_new(());
static LIBRARY: OnceLock<PathBuf> = OnceLock::new();

/// A plugin bundle laid out as `scripts/package_bundle.py` writes it.
pub struct Bundle {
    directory: TempDir,
}

impl Bundle {
    /// Packages the built library with the crate's manifest and schema in a temporary directory.
    pub fn new() -> Self {
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
    pub fn plugins_toml(&self, config: Json) -> PathBuf {
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
pub fn crate_dir() -> &'static Path {
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
        .arg(Path::new(env!("CARGO_TARGET_TMPDIR")).join("plugin"));
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
