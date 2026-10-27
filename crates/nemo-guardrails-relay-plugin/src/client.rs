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

//! Client for the `/v1/hooks/nemo-relay` endpoint of the NeMo Guardrails service, a hooks server.
//!
//! The endpoint's contract is the service's code in `nemoguardrails/server/hooks/`: the
//! request model in `schemas.py`, the decision in `adapters.py`, and the error statuses in `app.py`.

use std::fmt;
use std::time::Duration;

use reqwest::header::{CONTENT_TYPE, RETRY_AFTER};
use reqwest::redirect::Policy;
use reqwest::{StatusCode, Url};
use serde::{Deserialize, Deserializer};
use serde_json::{Value as Json, json};

use crate::config::NeMoGuardrailsClientConfig;

/// The endpoint's path, relative to the configured base URL.
pub const HOOK_PATH: &str = "v1/hooks/nemo-relay";

/// The `hook_event_name` the endpoint accepts.
pub const HOOK_EVENT_NAME: &str = "NemoRelayToolConditionalExecutionGuardrail";

/// Longest server message that a failure repeats.
const MAX_SERVER_MESSAGE_CHARS: usize = 512;

/// The service's decision on a tool call, as a 200 answer carries it.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(tag = "decision", rename_all = "lowercase")]
pub enum Verdict {
    /// The call may run.
    Allow,
    /// The call must not run.
    Deny {
        /// The service's reason, unless it gave none or an empty one.
        #[serde(default, deserialize_with = "non_empty")]
        reason: Option<String>,
    },
}

/// Reads an optional string, taking an empty one as none.
fn non_empty<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<String>, D::Error> {
    let text = Option::<String>::deserialize(deserializer)?;
    Ok(text.filter(|text| !text.is_empty()))
}

/// A check that ended without a decision. Each `detail` says what went wrong, and never contains
/// a header value or the tool's arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckFailure {
    /// No complete answer arrived: the request timed out or couldn't be sent.
    Unanswered { detail: String },
    /// The service answered, but not with a decision.
    Answered {
        /// The answer's status.
        status: StatusCode,
        /// The `error.code` of the answer, when it gave one.
        code: Option<String>,
        detail: String,
    },
}

impl CheckFailure {
    /// The status the service answered with, when it answered.
    pub fn status(&self) -> Option<StatusCode> {
        match self {
            Self::Unanswered { .. } => None,
            Self::Answered { status, .. } => Some(*status),
        }
    }

    /// The `error.code` of the service's answer, when it gave one.
    pub fn code(&self) -> Option<&str> {
        match self {
            Self::Unanswered { .. } => None,
            Self::Answered { code, .. } => code.as_deref(),
        }
    }

    /// What went wrong.
    pub fn detail(&self) -> &str {
        match self {
            Self::Unanswered { detail } | Self::Answered { detail, .. } => detail,
        }
    }
}

impl fmt::Display for CheckFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.detail())
    }
}

impl std::error::Error for CheckFailure {}

/// Posts tool calls to the NeMo Guardrails service.
#[derive(Debug, Clone)]
pub struct HooksClient {
    http: reqwest::Client,
    url: Url,
    timeout: Duration,
}

impl HooksClient {
    /// A client for the configured service, which sends the configured headers.
    pub fn new(config: &NeMoGuardrailsClientConfig) -> Result<Self, String> {
        let url = config
            .endpoint
            .join(HOOK_PATH)
            .map_err(|error| format!("cannot build the service URL: {error}"))?;
        let http = reqwest::Client::builder()
            .default_headers(config.headers.clone())
            .timeout(config.timeout)
            // The service never redirects, and following one would send the headers elsewhere.
            .redirect(Policy::none())
            .build()
            .map_err(|error| format!("cannot build the HTTP client: {error}"))?;
        Ok(Self {
            http,
            url,
            timeout: config.timeout,
        })
    }

    /// Asks the service whether the tool call may run.
    pub async fn check_tool_call(
        &self,
        tool_name: &str,
        tool_input: &Json,
    ) -> Result<Verdict, CheckFailure> {
        // Exactly the fields the endpoint accepts: it rejects any other with a 422.
        let body = json!({
            "hook_event_name": HOOK_EVENT_NAME,
            "tool_name": tool_name,
            "tool_input": tool_input,
        });
        let response = self
            .http
            .post(self.url.clone())
            .header(CONTENT_TYPE, "application/json")
            .body(body.to_string())
            .send()
            .await
            .map_err(|error| self.transport_failure(&error))?;
        let status = response.status();
        let retry_after = response
            .headers()
            .get(RETRY_AFTER)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        let payload = response
            .bytes()
            .await
            .map_err(|error| self.transport_failure(&error))?;
        if status != StatusCode::OK {
            return Err(status_failure(status, retry_after.as_deref(), &payload));
        }
        decision(&payload)
    }

    /// The failure for a request that got no complete answer.
    fn transport_failure(&self, error: &reqwest::Error) -> CheckFailure {
        let detail = if error.is_timeout() {
            format!("request timed out after {} ms", self.timeout.as_millis())
        } else {
            error_chain(error)
        };
        CheckFailure::Unanswered { detail }
    }
}

/// The verdict a 200 answer carries, or a failure when it carries none.
fn decision(payload: &[u8]) -> Result<Verdict, CheckFailure> {
    serde_json::from_slice(payload).map_err(|error| CheckFailure::Answered {
        status: StatusCode::OK,
        code: None,
        detail: format!("unexpected response: {error}"),
    })
}

/// The failure for an answer with any status but 200.
fn status_failure(status: StatusCode, retry_after: Option<&str>, payload: &[u8]) -> CheckFailure {
    let error = ErrorBody::parse(payload);
    let mut detail = format!("server returned HTTP {status}");
    if let Some(code) = &error.code {
        detail.push_str(&format!(" ({code})"));
    }
    if let Some(message) = &error.message {
        detail.push_str(&format!(": {message}"));
    }
    if let Some(hint) = hint(status, error.code.as_deref(), retry_after) {
        detail.push_str(&format!(" ({hint})"));
    }
    CheckFailure::Answered {
        status,
        code: error.code,
        detail,
    }
}

/// The parts of an error answer that a failure reports.
#[derive(Default)]
struct ErrorBody {
    code: Option<String>,
    message: Option<String>,
}

impl ErrorBody {
    /// Reads the OpenAI error envelope the service sends, or the `detail` of another server.
    fn parse(payload: &[u8]) -> Self {
        let Ok(body) = serde_json::from_slice::<Json>(payload) else {
            return Self::default();
        };
        let code = body.pointer("/error/code").and_then(Json::as_str);
        let message = body
            .pointer("/error/message")
            .or_else(|| body.get("detail"))
            .and_then(Json::as_str);
        Self {
            code: code.map(str::to_owned),
            message: message
                .map(|message| message.chars().take(MAX_SERVER_MESSAGE_CHARS).collect()),
        }
    }
}

/// The likely cause of an error the service answers with, keyed on its status and code.
fn hint(status: StatusCode, code: Option<&str>, retry_after: Option<&str>) -> Option<String> {
    match (status.as_u16(), code) {
        (404, _) => Some(
            "the endpoint isn't a NeMo Guardrails service; start one with \
             `nemoguardrails hooks-server --config <path>`, which listens on port 8100 by default"
                .into(),
        ),
        (422, None | Some("unsupported_hook_event")) => Some(
            "the service rejected the plugin's request; report it as a plugin bug, with the \
             service's Guardrails version"
                .into(),
        ),
        (500, Some("rail_failed")) => Some(
            "a rail failed instead of deciding, such as a judge model that can't be reached".into(),
        ),
        (500, None) => Some("the service couldn't finish the check; see its log".into()),
        (503, Some("queue_full")) => Some(at_capacity_hint(retry_after)),
        _ => None,
    }
}

/// The hint for a full admission queue, with the wait the service asks for when it gives one.
fn at_capacity_hint(retry_after: Option<&str>) -> String {
    match retry_after {
        Some(retry_after) => format!("the service is at capacity; Retry-After: {retry_after}"),
        None => "the service is at capacity".into(),
    }
}

/// An error's message followed by each of its causes.
fn error_chain(error: &dyn std::error::Error) -> String {
    let mut message = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        message.push_str(&format!(": {cause}"));
        source = cause.source();
    }
    message
}
