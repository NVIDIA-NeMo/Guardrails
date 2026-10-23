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
//! `relay-plugin.toml` names. The plugin registers one tool conditional-execution guardrail,
//! which posts each tool call to the hooks server: a deny rejects the call, and a check that fails
//! returns an error, on which Relay also blocks the call.

pub mod cache;
pub mod client;
pub mod config;

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use nemo_relay_plugin::{
    ConfigDiagnostic, DiagnosticLevel, Json, NativePlugin, PluginContext, PluginRuntime,
};
use serde_json::{Map, json};

use crate::cache::{CallKey, MAX_VERDICTS, VERDICT_TTL, VerdictCache};
use crate::client::{CheckFailure, HooksClient, Verdict};
use crate::config::{ChecksConfig, ConfigError};

/// Plugin kind. It must match `plugin.id` in `relay-plugin.toml`.
pub const PLUGIN_KIND: &str = "nvidia.nemo_guardrails_checks";

/// Diagnostic code for a component configuration that the plugin rejects.
pub const INVALID_CONFIG_CODE: &str = "nemo_guardrails_checks.invalid_config";

/// Name of the tool conditional-execution guardrail the plugin registers.
pub const GUARDRAIL_NAME: &str = "nemo_guardrails_tool_call_check";

/// `event` field of the warning that a failed check logs.
pub const CHECK_FAILED_EVENT: &str = "nemo_guardrails_check_failed";

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
        ctx: &mut PluginContext<'_>,
    ) -> nemo_relay_plugin::Result<()> {
        let config = ChecksConfig::from_component_config(plugin_config)
            .map_err(|error| invalid_config_message(&error))?;
        let check = Arc::new(ToolCallCheck {
            client: HooksClient::new(&config)?,
            verdicts: Mutex::new(VerdictCache::new(VERDICT_TTL, MAX_VERDICTS)),
            runtime: ctx.runtime(),
        });
        ctx.register_tool_conditional_execution_guardrail(
            GUARDRAIL_NAME,
            config.priority,
            move |tool_name, tool_input| {
                let check = Arc::clone(&check);
                async move { check.run(&tool_name, &tool_input).await }
            },
        )
    }
}

/// The check that the plugin's guardrail runs on each tool call.
struct ToolCallCheck {
    client: HooksClient,
    verdicts: Mutex<VerdictCache>,
    runtime: PluginRuntime,
}

impl ToolCallCheck {
    /// `None` lets the call run, a reason rejects it, and an error blocks it as a failed check.
    async fn run(
        &self,
        tool_name: &str,
        tool_input: &Json,
    ) -> nemo_relay_plugin::Result<Option<String>> {
        let key = CallKey::new(tool_name, tool_input);
        let cached = self.verdicts().get(&key, Instant::now());
        if let Some(verdict) = cached {
            return Ok(rejection(tool_name, &verdict));
        }
        let verdict = self
            .client
            .check_tool_call(tool_name, tool_input)
            .await
            .map_err(|failure| self.report_failure(tool_name, &failure))?;
        self.verdicts().insert(key, verdict.clone(), Instant::now());
        Ok(rejection(tool_name, &verdict))
    }

    /// The verdict cache. A panic while it was locked leaves it consistent, so it stays usable.
    fn verdicts(&self) -> MutexGuard<'_, VerdictCache> {
        self.verdicts.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Logs a warning for the failed check and returns the guardrail's error for it.
    fn report_failure(&self, tool_name: &str, failure: &CheckFailure) -> String {
        // From Relay 0.10.1, an error that starts with `HTTP <status>: ` becomes an upstream HTTP
        // error, so this text must never start that way.
        let message =
            format!("NeMo Guardrails check failed for tool call '{tool_name}': {failure}");
        let fields = json!({
            "event": CHECK_FAILED_EVENT,
            "tool_name": tool_name,
            "status": failure.status.map(|status| status.as_u16()),
            "error_code": failure.code,
        });
        // The call is blocked either way, so a warning the host can't take changes nothing.
        let _ = self.runtime.warn(PLUGIN_KIND, &message, Some(&fields));
        message
    }
}

/// The guardrail's rejection for a verdict, or `None` when the call may run.
fn rejection(tool_name: &str, verdict: &Verdict) -> Option<String> {
    match verdict {
        Verdict::Allow => None,
        Verdict::Deny {
            reason: Some(reason),
        } => Some(format!(
            "NeMo Guardrails blocked tool call '{tool_name}': {reason}"
        )),
        Verdict::Deny { reason: None } => {
            Some(format!("NeMo Guardrails blocked tool call '{tool_name}'"))
        }
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

/// The plugin's diagnostic for an invalid component config.
fn invalid_config_diagnostic(error: &ConfigError) -> ConfigDiagnostic {
    ConfigDiagnostic {
        level: DiagnosticLevel::Error,
        code: INVALID_CONFIG_CODE.into(),
        component: Some(PLUGIN_KIND.into()),
        field: error.field.map(Into::into),
        message: invalid_config_message(error),
    }
}

/// The message that both `validate` and `register` give for an invalid component config.
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

    /// Validates a JSON component config as Relay does.
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
