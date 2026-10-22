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

//! NeMo Relay native plugin that checks tool calls with a NeMo Guardrails hooks server.
//!
//! Relay loads the `cdylib` through the `nemo_relay_register_plugin` symbol that
//! `relay-plugin.toml` names. The plugin validates its component configuration and registers
//! no checks yet.

pub mod config;

use nemo_relay_plugin::{ConfigDiagnostic, DiagnosticLevel, Json, NativePlugin, PluginContext};
use serde_json::Map;

use crate::config::{ChecksConfig, ConfigError};

/// Plugin kind. It must match `plugin.id` in `relay-plugin.toml`.
pub const PLUGIN_KIND: &str = "nvidia.nemo_guardrails_checks";

/// Diagnostic code for a component configuration that the plugin rejects.
pub const INVALID_CONFIG_CODE: &str = "nemo_guardrails_checks.invalid_config";

/// The NeMo Guardrails checks plugin.
#[derive(Debug, Default)]
pub struct NemoGuardrailsChecksPlugin;

impl NativePlugin for NemoGuardrailsChecksPlugin {
    fn plugin_kind(&self) -> &str {
        PLUGIN_KIND
    }

    fn allows_multiple_components(&self) -> bool {
        false
    }

    fn validate(&self, plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        let executor = self
            .executor_config_for_component(plugin_config)
            .err()
            .map(executor_diagnostic);
        let config = ChecksConfig::from_component_config(plugin_config)
            .err()
            .map(|error| invalid_config_diagnostic(&error));
        executor.into_iter().chain(config).collect()
    }

    fn register(
        &mut self,
        plugin_config: &Map<String, Json>,
        _ctx: &mut PluginContext<'_>,
    ) -> nemo_relay_plugin::Result<()> {
        ChecksConfig::from_component_config(plugin_config)
            .map_err(|error| invalid_config_message(&error))?;
        Ok(())
    }
}

/// Repeats the SDK's own executor diagnostic, which overriding `validate` would otherwise drop.
fn executor_diagnostic(message: String) -> ConfigDiagnostic {
    ConfigDiagnostic {
        level: DiagnosticLevel::Error,
        code: "native_executor_config.invalid".into(),
        component: None,
        field: Some("executor.worker_threads".into()),
        message,
    }
}

fn invalid_config_diagnostic(error: &ConfigError) -> ConfigDiagnostic {
    ConfigDiagnostic {
        level: DiagnosticLevel::Error,
        code: INVALID_CONFIG_CODE.into(),
        component: Some(PLUGIN_KIND.into()),
        field: error.field.map(Into::into),
        message: invalid_config_message(error),
    }
}

fn invalid_config_message(error: &ConfigError) -> String {
    format!("invalid NeMo Guardrails checks config: {error}")
}

nemo_relay_plugin::nemo_relay_plugin!(
    nemo_relay_register_plugin,
    NemoGuardrailsChecksPlugin::default
);

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn validate(config: Json) -> Vec<ConfigDiagnostic> {
        NemoGuardrailsChecksPlugin.validate(config.as_object().expect("config is an object"))
    }

    /// A valid config with an executor override produces no diagnostics.
    #[test]
    fn valid_config_has_no_diagnostics() {
        let diagnostics = validate(json!({
            "endpoint": "http://127.0.0.1:8100",
            "executor": {"worker_threads": 2}
        }));

        assert!(diagnostics.is_empty(), "{diagnostics:?}");
    }

    /// An invalid config becomes one plugin error that names the field.
    #[test]
    fn invalid_config_becomes_one_diagnostic() {
        let diagnostics = validate(json!({"endpoint": "ftp://127.0.0.1"}));

        assert_eq!(
            diagnostics,
            vec![ConfigDiagnostic {
                level: DiagnosticLevel::Error,
                code: INVALID_CONFIG_CODE.into(),
                component: Some(PLUGIN_KIND.into()),
                field: Some("endpoint".into()),
                message: "invalid NeMo Guardrails checks config: endpoint must use http or https"
                    .into(),
            }]
        );
    }

    /// A plugin that keeps the SDK's default `validate`, for comparison.
    struct SdkDefaultValidate;

    impl NativePlugin for SdkDefaultValidate {
        fn plugin_kind(&self) -> &str {
            PLUGIN_KIND
        }

        fn register(
            &mut self,
            _plugin_config: &Map<String, Json>,
            _ctx: &mut PluginContext<'_>,
        ) -> nemo_relay_plugin::Result<()> {
            Ok(())
        }
    }

    /// An invalid executor override yields exactly the diagnostic the SDK's own `validate` gives.
    #[test]
    fn invalid_executor_keeps_the_sdk_diagnostic() {
        for executor in [json!({"worker_threads": 0}), json!(5)] {
            let config = json!({"endpoint": "http://127.0.0.1:8100", "executor": executor});
            let sdk_diagnostics =
                SdkDefaultValidate.validate(config.as_object().expect("config is an object"));

            assert_eq!(sdk_diagnostics.len(), 1, "{executor}");
            assert_eq!(validate(config), sdk_diagnostics, "{executor}");
        }
    }
}
