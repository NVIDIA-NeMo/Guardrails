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

mod common;

use std::path::PathBuf;

use nemo_relay::plugin::dynamic::{initialize, relay_version_matches};
use nemo_relay::plugin::{PluginConfig, list_plugin_kinds};
use semver::{Version, VersionReq};
use serde_json::json;

use common::{Bundle, HOST, crate_dir};

const PLUGIN_ID: &str = "nvidia.nemo_guardrails_checks";

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

/// The manifest's config schema rejects a field it doesn't declare, such as `config_id`.
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

    // The schema leaves credentials to the plugin, because Relay's schema errors quote the value.
    for (endpoint, message) in [
        ("http://exa mple.com", "endpoint is not a valid URL"),
        (
            "https://user:secret@gw.example.com",
            "endpoint must not contain a username or password",
        ),
    ] {
        let bundle = Bundle::new();
        let error = initialize_error(bundle.plugins_toml(json!({"endpoint": endpoint}))).await;

        assert!(
            error.contains(&format!("invalid NeMo Guardrails checks config: {message}")),
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
            error.contains(&format!("invalid NeMo Guardrails checks config: {message}")),
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
