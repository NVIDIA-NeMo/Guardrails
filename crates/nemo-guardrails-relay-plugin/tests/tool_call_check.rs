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

//! Checks tool calls through the plugin loaded in a real NeMo Relay host, against a stand-in hooks
//! server.

mod common;

use std::net::TcpListener;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Once};

use nemo_relay::api::runtime::ToolExecutionNextFn;
use nemo_relay::api::tool::{
    ToolCallExecuteParams, ToolExecutionResult, tool_call_execute, tool_conditional_execution,
};
use nemo_relay::error::FlowError;
use nemo_relay::plugin::PluginConfig;
use nemo_relay::plugin::dynamic::{PluginHostActivation, initialize};
use serde_json::{Map, Value as Json, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use common::{Bundle, HOST};

/// The log target of the plugin's records, as the host emits them.
const LOG_TARGET: &str = "nemo_relay.plugin.native.nvidia.nemo_guardrails";

/// The service's answer to a call it fails on a rail.
const RAIL_FAILED_MESSAGE: &str =
    "Rail 'tool safety check' failed: the judge model can't be reached";

/// An allowed call runs the tool and returns its result.
#[tokio::test]
async fn allowed_call_runs_the_tool() {
    let _host = HOST.lock().await;
    let server = hooks_server(decision(json!({"decision": "allow"}))).await;
    let bundle = Bundle::new();
    let _plugin = activate(&bundle, json!({"endpoint": server.uri()})).await;
    let runs = Arc::new(AtomicUsize::new(0));

    let result = run_search(&runs).await;

    let result = result.expect("an allowed call should run");
    assert_eq!(result.result, json!({"query": "weather"}));
    assert_eq!(runs.load(Ordering::SeqCst), 1);
    assert_eq!(received(&server).await, 1);
}

/// A denied call is rejected with the service's reason, or with none when the service gives none,
/// and the tool never runs.
#[tokio::test]
async fn denied_call_is_rejected_with_the_server_reason() {
    let _host = HOST.lock().await;

    for (response, expected) in [
        (
            json!({"decision": "deny", "reason": "search is not allowed"}),
            "NeMo Guardrails blocked tool call 'search': search is not allowed",
        ),
        (
            json!({"decision": "deny"}),
            "NeMo Guardrails blocked tool call 'search'",
        ),
    ] {
        let server = hooks_server(decision(response)).await;
        let bundle = Bundle::new();
        let plugin = activate(&bundle, json!({"endpoint": server.uri()})).await;
        let runs = Arc::new(AtomicUsize::new(0));

        let result = run_search(&runs).await;

        match result {
            Err(FlowError::GuardrailRejected(reason)) => assert_eq!(reason, expected),
            other => panic!("expected a guardrail rejection, got {other:?}"),
        }
        assert_eq!(runs.load(Ordering::SeqCst), 0, "{expected}");
        drop(plugin);
    }
}

/// A failed check blocks the call with an internal error, not a rejection or an upstream HTTP
/// error, and the tool never runs.
#[tokio::test]
async fn failed_check_blocks_the_call_with_an_internal_error() {
    let _host = HOST.lock().await;
    let server = hooks_server(rail_failed()).await;
    let bundle = Bundle::new();
    let _plugin = activate(&bundle, json!({"endpoint": server.uri()})).await;
    let runs = Arc::new(AtomicUsize::new(0));

    let result = run_search(&runs).await;

    match result {
        Err(FlowError::Internal(message)) => assert_eq!(
            message,
            format!(
                "NeMo Guardrails check failed for tool call 'search': server returned HTTP 500 \
                 Internal Server Error (rail_failed): {RAIL_FAILED_MESSAGE} (a rail failed \
                 instead of deciding, such as a judge model that can't be reached)"
            )
        ),
        other => panic!("expected an internal error, got {other:?}"),
    }
    assert_eq!(runs.load(Ordering::SeqCst), 0);
}

/// Every call is checked, including an identical one right after, as PreToolUse and then
/// PermissionRequest make through Relay 0.10.4.
#[tokio::test]
async fn every_call_is_checked() {
    let _host = HOST.lock().await;

    for response in [
        json!({"decision": "allow"}),
        json!({"decision": "deny", "reason": "search is not allowed"}),
    ] {
        let server = hooks_server(decision(response.clone())).await;
        let bundle = Bundle::new();
        let plugin = activate(&bundle, json!({"endpoint": server.uri()})).await;
        let arguments = json!({"query": "weather", "limit": 3});

        let first = tool_conditional_execution("search", &arguments).await;
        let second = tool_conditional_execution("search", &arguments).await;

        assert_eq!(format!("{first:?}"), format!("{second:?}"), "{response}");
        assert_eq!(received(&server).await, 2, "{response}");
        drop(plugin);
    }
}

/// A failed check logs one warning with its event, status and error code, with or without an
/// answer from the service, and never a header value or the tool's arguments.
#[tokio::test]
async fn failed_check_logs_a_warning() {
    let _host = HOST.lock().await;
    capture_plugin_logs();
    let server = hooks_server(rail_failed()).await;
    let (_silent, silent_endpoint) = silent_endpoint();

    for (endpoint, status, error_code) in [
        (server.uri(), json!(500), json!("rail_failed")),
        (silent_endpoint, Json::Null, Json::Null),
    ] {
        let bundle = Bundle::new();
        let plugin = activate(
            &bundle,
            json!({
                "endpoint": endpoint,
                "headers": {"authorization": "Bearer secret-token"},
                "timeout_millis": 100
            }),
        )
        .await;
        PLUGIN_LOGS.lock().expect("log capture lock").clear();

        let error = tool_conditional_execution("search", &json!({"query": "secret-argument"}))
            .await
            .expect_err("a failed check should block the call");

        let logs = std::mem::take(&mut *PLUGIN_LOGS.lock().expect("log capture lock"));
        let [record] = logs.as_slice() else {
            panic!("expected one warning, got {logs:?}");
        };
        let FlowError::Internal(message) = &error else {
            panic!("expected an internal error, got {error:?}");
        };
        assert_eq!(record.level, log::Level::Warn);
        assert_eq!(record.message, *message);
        assert_eq!(
            Json::Object(record.fields.clone()),
            json!({
                "event": "nemo_guardrails_check_failed",
                "tool_name": "search",
                "status": status,
                "error_code": error_code
            })
        );
        assert!(!format!("{record:?}").contains("secret"), "{record:?}");
        drop(plugin);
    }
}

/// Activates the plugin in the host with one component config.
async fn activate(bundle: &Bundle, config: Json) -> PluginHostActivation {
    initialize(PluginConfig::default(), Some(bundle.plugins_toml(config)))
        .await
        .expect("the plugin should activate")
}

/// Runs a managed `search` call whose tool counts its runs and returns its arguments.
async fn run_search(runs: &Arc<AtomicUsize>) -> Result<ToolExecutionResult, FlowError> {
    let runs = Arc::clone(runs);
    let tool: ToolExecutionNextFn = Arc::new(move |arguments| {
        runs.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move { Ok(ToolExecutionResult::new(arguments)) })
    });
    tool_call_execute(
        ToolCallExecuteParams::builder()
            .name("search")
            .args(json!({"query": "weather"}))
            .func(tool)
            .build(),
    )
    .await
}

/// A stand-in service that gives every check the same answer.
async fn hooks_server(response: ResponseTemplate) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/hooks/nemo-relay"))
        .respond_with(response)
        .mount(&server)
        .await;
    server
}

/// A 200 answer with the given decision.
fn decision(body: Json) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(body)
}

/// The service's answer when a rail fails instead of deciding.
fn rail_failed() -> ResponseTemplate {
    ResponseTemplate::new(500).set_body_json(json!({
        "error": {
            "message": RAIL_FAILED_MESSAGE,
            "type": "server_error",
            "param": null,
            "code": "rail_failed"
        }
    }))
}

/// How many checks the service received.
async fn received(server: &MockServer) -> usize {
    server
        .received_requests()
        .await
        .expect("request recording is on")
        .len()
}

/// A listener that never accepts, and its URL: a request to it gets no answer while it lives.
fn silent_endpoint() -> (TcpListener, String) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("a local port should bind");
    let address = listener.local_addr().expect("the listener has an address");
    (listener, format!("http://{address}"))
}

/// A log record the plugin emitted through the host.
#[derive(Debug)]
struct PluginLog {
    level: log::Level,
    message: String,
    fields: Map<String, Json>,
}

/// The plugin's log records, from when `capture_plugin_logs` installed the logger.
static PLUGIN_LOGS: Mutex<Vec<PluginLog>> = Mutex::new(Vec::new());

/// Installs a `log` logger that keeps the plugin's records, once per test process.
fn capture_plugin_logs() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        log::set_logger(&PluginLogCapture).expect("no other logger is installed");
        log::set_max_level(log::LevelFilter::Trace);
    });
}

/// A `log` logger that keeps the records with the plugin's target.
struct PluginLogCapture;

impl log::Log for PluginLogCapture {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.target() == LOG_TARGET
    }

    fn log(&self, record: &log::Record<'_>) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let mut fields = LogFields::default();
        record
            .key_values()
            .visit(&mut fields)
            .expect("log fields should be readable");
        PLUGIN_LOGS
            .lock()
            .expect("log capture lock")
            .push(PluginLog {
                level: record.level(),
                message: record.args().to_string(),
                fields: fields.0,
            });
    }

    fn flush(&self) {}
}

/// The key-value fields of a log record, as JSON.
#[derive(Default)]
struct LogFields(Map<String, Json>);

impl<'kvs> log::kv::VisitSource<'kvs> for LogFields {
    fn visit_pair(
        &mut self,
        key: log::kv::Key<'kvs>,
        value: log::kv::Value<'kvs>,
    ) -> Result<(), log::kv::Error> {
        let value = serde_json::to_value(&value).expect("log field values are JSON");
        self.0.insert(key.to_string(), value);
        Ok(())
    }
}
