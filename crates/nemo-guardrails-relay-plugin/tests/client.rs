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

//! Checks the hooks client against a mock NeMo Guardrails service: the request it sends, and how
//! it reads each answer that the service's code (`nemoguardrails/server/hooks/`) can give.

use std::net::TcpListener;

use nemo_guardrails_relay_plugin::client::{CheckFailure, HooksClient, Verdict};
use nemo_guardrails_relay_plugin::config::NeMoGuardrailsClientConfig;
use reqwest::StatusCode;
use serde_json::{Value as Json, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

const HOOK_PATH: &str = "/v1/hooks/nemo-relay";

/// A client for the service at `endpoint`, with the other config fields in `config`.
fn client(endpoint: &str, config: Json) -> HooksClient {
    let mut config = config.as_object().cloned().expect("config is an object");
    config.insert("endpoint".into(), json!(endpoint));
    let config =
        NeMoGuardrailsClientConfig::from_component_config(&config).expect("config is valid");
    HooksClient::new(&config).expect("client builds")
}

/// A mock service that answers every check with `response`.
async fn hooks_server(response: ResponseTemplate) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(HOOK_PATH))
        .respond_with(response)
        .mount(&server)
        .await;
    server
}

/// Checks a `search` call with a mock service that answers with `response`.
async fn check_search(response: ResponseTemplate) -> Result<Verdict, CheckFailure> {
    let server = hooks_server(response).await;
    client(&server.uri(), json!({}))
        .check_tool_call("search", &json!({"query": "weather"}))
        .await
}

/// The service's answer for an allowed call.
fn allow() -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(json!({"decision": "allow"}))
}

/// An error answer in the OpenAI envelope that the service renders every error with.
fn envelope(
    status: u16,
    message: &str,
    error_type: &str,
    param: Json,
    code: Json,
) -> ResponseTemplate {
    ResponseTemplate::new(status).set_body_json(json!({
        "error": {"message": message, "type": error_type, "param": param, "code": code}
    }))
}

/// The requests the mock service received.
async fn received(server: &MockServer) -> Vec<Request> {
    server
        .received_requests()
        .await
        .expect("the mock server records requests")
}

/// A local listener that never accepts, so connections wait in its backlog unanswered until it
/// is dropped, and its endpoint.
fn silent_endpoint() -> (TcpListener, String) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("listener binds");
    let address = listener.local_addr().expect("listener has an address");
    (listener, format!("http://{address}"))
}

/// A local endpoint that refuses connections.
fn refusing_endpoint() -> String {
    let (_listener, endpoint) = silent_endpoint();
    endpoint
}

/// An allow decision lets the call run.
#[tokio::test]
async fn allow_decision_allows_the_call() {
    assert_eq!(check_search(allow()).await, Ok(Verdict::Allow));
}

/// A deny decision denies the call with the service's reason.
#[tokio::test]
async fn deny_decision_carries_the_server_reason() {
    let response = ResponseTemplate::new(200)
        .set_body_json(json!({"decision": "deny", "reason": "regex check tool call"}));

    assert_eq!(
        check_search(response).await,
        Ok(Verdict::Deny {
            reason: Some("regex check tool call".into())
        })
    );
}

/// A deny with a missing, null or empty reason still denies the call.
#[tokio::test]
async fn deny_decision_without_a_reason_still_denies() {
    for body in [
        json!({"decision": "deny"}),
        json!({"decision": "deny", "reason": null}),
        json!({"decision": "deny", "reason": ""}),
    ] {
        let response = ResponseTemplate::new(200).set_body_json(body.clone());

        assert_eq!(
            check_search(response).await,
            Ok(Verdict::Deny { reason: None }),
            "{body}"
        );
    }
}

/// Fields that the decision doesn't define are ignored, so the service can add some.
#[tokio::test]
async fn unknown_response_fields_are_ignored() {
    for (body, verdict) in [
        (json!({"decision": "allow", "rail": "x"}), Verdict::Allow),
        (
            json!({"decision": "deny", "reason": "no", "violation_type": "x"}),
            Verdict::Deny {
                reason: Some("no".into()),
            },
        ),
    ] {
        let response = ResponseTemplate::new(200).set_body_json(body.clone());

        assert_eq!(check_search(response).await, Ok(verdict), "{body}");
    }
}

/// The request is a JSON POST of exactly the three fields the endpoint accepts, with the
/// arguments as a JSON value rather than an encoded string, whatever their type.
#[tokio::test]
async fn request_carries_the_hook_event_tool_name_and_arguments() {
    for tool_input in [
        json!({"query": "weather"}),
        json!("ls -la"),
        Json::Null,
        json!([1, 2]),
    ] {
        let server = hooks_server(allow()).await;

        let verdict = client(&server.uri(), json!({}))
            .check_tool_call("search", &tool_input)
            .await;

        assert_eq!(verdict, Ok(Verdict::Allow), "{tool_input}");
        let requests = received(&server).await;
        assert_eq!(requests.len(), 1, "{tool_input}");
        assert_eq!(requests[0].headers["content-type"], "application/json");
        assert_eq!(
            requests[0].body_json::<Json>().expect("body is JSON"),
            json!({
                "hook_event_name": "NemoRelayToolConditionalExecutionGuardrail",
                "tool_name": "search",
                "tool_input": tool_input
            })
        );
    }
}

/// An endpoint with a path prefix posts below that prefix, with or without a trailing slash.
#[tokio::test]
async fn endpoint_path_prefix_is_kept() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/guardrails/v1/hooks/nemo-relay"))
        .respond_with(allow())
        .expect(2)
        .mount(&server)
        .await;

    for endpoint in [
        format!("{}/guardrails", server.uri()),
        format!("{}/guardrails/", server.uri()),
    ] {
        let verdict = client(&endpoint, json!({}))
            .check_tool_call("search", &json!({}))
            .await;

        assert_eq!(verdict, Ok(Verdict::Allow), "{endpoint}");
    }
}

/// The configured headers are sent with every check.
#[tokio::test]
async fn configured_headers_are_sent() {
    let server = hooks_server(allow()).await;
    let client = client(
        &server.uri(),
        json!({"headers": {"Authorization": "Bearer test-token", "x-team": "agents"}}),
    );

    for _ in 0..2 {
        client
            .check_tool_call("search", &json!({}))
            .await
            .expect("check passes");
    }

    let requests = received(&server).await;
    assert_eq!(requests.len(), 2);
    for request in requests {
        assert_eq!(request.headers["authorization"], "Bearer test-token");
        assert_eq!(request.headers["x-team"], "agents");
    }
}

/// Each error the service answers with fails the check, reporting the status, the
/// `error.code`, the service's message, and a hint at the likely cause.
#[tokio::test]
async fn hooks_server_errors_fail_with_status_code_message_and_hint() {
    for (response, status, code, detail) in [
        (
            envelope(
                422,
                "body.session_id: Extra inputs are not permitted",
                "invalid_request_error",
                json!("session_id"),
                Json::Null,
            ),
            422,
            None,
            "server returned HTTP 422 Unprocessable Entity: body.session_id: Extra inputs are not \
             permitted (the service rejected the plugin's request; report it as a plugin bug, \
             with the service's Guardrails version)",
        ),
        (
            envelope(
                422,
                "hook_event_name 'Other' is not handled on /v1/hooks/nemo-relay; send \
                 NemoRelayToolConditionalExecutionGuardrail",
                "invalid_request_error",
                json!("hook_event_name"),
                json!("unsupported_hook_event"),
            ),
            422,
            Some("unsupported_hook_event"),
            "server returned HTTP 422 Unprocessable Entity (unsupported_hook_event): \
             hook_event_name 'Other' is not handled on /v1/hooks/nemo-relay; send \
             NemoRelayToolConditionalExecutionGuardrail (the service rejected the plugin's \
             request; report it as a plugin bug, with the service's Guardrails version)",
        ),
        (
            envelope(
                500,
                "rail 'tool safety check' failed: model 'judge' timed out",
                "server_error",
                Json::Null,
                json!("rail_failed"),
            ),
            500,
            Some("rail_failed"),
            "server returned HTTP 500 Internal Server Error (rail_failed): rail 'tool safety \
             check' failed: model 'judge' timed out (a rail failed instead of deciding, such as \
             a judge model that can't be reached)",
        ),
        (
            envelope(
                500,
                "Internal server error",
                "server_error",
                Json::Null,
                Json::Null,
            ),
            500,
            None,
            "server returned HTTP 500 Internal Server Error: Internal server error (the service \
             couldn't finish the check; see its log)",
        ),
        (
            envelope(
                503,
                "IORails admission queue is full. Please retry later.",
                "server_error",
                Json::Null,
                json!("queue_full"),
            )
            .insert_header("retry-after", "30"),
            503,
            Some("queue_full"),
            "server returned HTTP 503 Service Unavailable (queue_full): IORails admission queue \
             is full. Please retry later. (the service is at capacity; Retry-After: 30)",
        ),
        (
            envelope(
                503,
                "IORails admission queue is full. Please retry later.",
                "server_error",
                Json::Null,
                json!("queue_full"),
            ),
            503,
            Some("queue_full"),
            "server returned HTTP 503 Service Unavailable (queue_full): IORails admission queue \
             is full. Please retry later. (the service is at capacity)",
        ),
        (
            envelope(404, "Not Found", "not_found_error", Json::Null, Json::Null),
            404,
            None,
            "server returned HTTP 404 Not Found: Not Found (the endpoint isn't a NeMo Guardrails \
             service; start one with `nemoguardrails hooks-server --config <path>`, which \
             listens on port 8100 by default)",
        ),
        (
            ResponseTemplate::new(404).set_body_json(json!({"detail": "Not Found"})),
            404,
            None,
            "server returned HTTP 404 Not Found: Not Found (the endpoint isn't a NeMo Guardrails \
             service; start one with `nemoguardrails hooks-server --config <path>`, which \
             listens on port 8100 by default)",
        ),
        (
            ResponseTemplate::new(502).set_body_string("<html>Bad Gateway</html>"),
            502,
            None,
            "server returned HTTP 502 Bad Gateway",
        ),
    ] {
        let failure = check_search(response)
            .await
            .expect_err("an error status fails the check");

        assert_eq!(
            failure,
            CheckFailure::Answered {
                status: StatusCode::from_u16(status).expect("valid status"),
                code: code.map(Into::into),
                detail: detail.into(),
            }
        );
    }
}

/// A long server message is cut to 512 characters, on a character boundary.
#[tokio::test]
async fn long_server_message_is_cut() {
    let message = "é".repeat(600);
    let response = envelope(500, &message, "server_error", Json::Null, Json::Null);

    let failure = check_search(response)
        .await
        .expect_err("an error status fails the check");

    assert_eq!(
        failure.detail(),
        format!(
            "server returned HTTP 500 Internal Server Error: {} (the service couldn't finish \
             the check; see its log)",
            "é".repeat(512)
        )
    );
}

/// A 200 answer that isn't an allow or a deny fails the check.
#[tokio::test]
async fn unexpected_200_answer_fails_the_check() {
    for (body, error) in [
        ("not json", "expected ident"),
        (r#""allow""#, "invalid type: string \"allow\""),
        ("[]", "missing field `decision`"),
        ("{}", "missing field `decision`"),
        (r#"{"decision": "maybe"}"#, "unknown variant `maybe`"),
        (
            r#"{"decision": "deny", "reason": 5}"#,
            "invalid type: integer `5`",
        ),
    ] {
        let response = ResponseTemplate::new(200).set_body_raw(body, "application/json");

        let failure = check_search(response)
            .await
            .expect_err("an unexpected answer fails the check");

        assert_eq!(failure.status(), Some(StatusCode::OK), "{body}");
        assert_eq!(failure.code(), None, "{body}");
        assert!(
            failure.detail().starts_with("unexpected response: ")
                && failure.detail().contains(error),
            "{body}: {failure}"
        );
    }
}

/// A success status other than 200 fails the check, since only a 200 carries a decision.
#[tokio::test]
async fn other_success_status_fails_the_check() {
    let failure = check_search(ResponseTemplate::new(204))
        .await
        .expect_err("a 204 fails the check");

    assert_eq!(failure.detail(), "server returned HTTP 204 No Content");
}

/// A redirect fails the check instead of being followed, so the configured headers never reach
/// another server.
#[tokio::test]
async fn redirect_fails_the_check_without_being_followed() {
    let server = MockServer::start().await;
    Mock::given(path(HOOK_PATH))
        .respond_with(
            ResponseTemplate::new(307)
                .insert_header("location", format!("{}/elsewhere", server.uri())),
        )
        .mount(&server)
        .await;
    Mock::given(path("/elsewhere"))
        .respond_with(allow())
        .expect(0)
        .mount(&server)
        .await;

    let failure = client(&server.uri(), json!({}))
        .check_tool_call("search", &json!({}))
        .await
        .expect_err("a redirect fails the check");

    assert_eq!(
        failure.detail(),
        "server returned HTTP 307 Temporary Redirect"
    );
}

/// A service that never answers fails the check once the configured time limit passes.
#[tokio::test]
async fn unanswered_check_times_out_at_the_configured_limit() {
    let (silent, endpoint) = silent_endpoint();

    let failure = client(&endpoint, json!({"timeout_millis": 100}))
        .check_tool_call("search", &json!({}))
        .await
        .expect_err("an unanswered check fails");

    assert_eq!(
        failure,
        CheckFailure::Unanswered {
            detail: "request timed out after 100 ms".into(),
        }
    );
    drop(silent);
}

/// A service that refuses the connection fails the check with the transport error and its causes.
#[tokio::test]
async fn unreachable_server_fails_with_the_transport_error() {
    let endpoint = refusing_endpoint();

    let failure = client(&endpoint, json!({}))
        .check_tool_call("search", &json!({}))
        .await
        .expect_err("an unreachable server fails the check");

    assert!(
        matches!(&failure, CheckFailure::Unanswered { detail }
            if detail.starts_with(&format!("error sending request for url ({endpoint}{HOOK_PATH}): "))),
        "{failure:?}"
    );
}

/// Neither a failure nor the client's debug output repeats a header value or the tool's arguments.
#[tokio::test]
async fn failures_never_repeat_header_values_or_arguments() {
    let erroring = hooks_server(envelope(
        500,
        "Internal server error",
        "server_error",
        Json::Null,
        Json::Null,
    ))
    .await;
    let (silent, silent_endpoint) = silent_endpoint();

    for endpoint in [erroring.uri(), silent_endpoint, refusing_endpoint()] {
        let client = client(
            &endpoint,
            json!({"headers": {"authorization": "Bearer secret-token"}, "timeout_millis": 100}),
        );

        let failure = client
            .check_tool_call("search", &json!({"query": "private-argument"}))
            .await
            .expect_err("the check fails");

        for text in [
            format!("{client:?}"),
            failure.to_string(),
            format!("{failure:?}"),
        ] {
            assert!(!text.contains("secret-token"), "{endpoint}: {text}");
            assert!(!text.contains("private-argument"), "{endpoint}: {text}");
        }
    }
    drop(silent);
}
