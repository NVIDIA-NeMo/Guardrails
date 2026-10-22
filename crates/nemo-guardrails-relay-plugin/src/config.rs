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

//! Component configuration, read from the plugin's entry in `plugins.toml`.

use std::fmt;
use std::time::Duration;

use reqwest::Url;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde::Deserialize;
use serde::de::IgnoredAny;
use serde_json::{Map, Value as Json};

/// Default per-check timeout.
///
/// It stays under the 30-second limit that Relay's CLI hooks allow, leaving time for the hook's own
/// round trip, so a slow check fails in the plugin rather than in hook delivery.
pub const DEFAULT_TIMEOUT_MILLIS: u64 = 25_000;

/// Headers that the plugin's HTTP client sets itself, or that control how a request is framed.
const CLIENT_OWNED_HEADERS: [&str; 10] = [
    "connection",
    "content-length",
    "content-type",
    "host",
    "keep-alive",
    "proxy-connection",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
];

/// A validated component configuration.
#[derive(Debug, Clone)]
pub struct ChecksConfig {
    /// Base URL of the NeMo Guardrails hooks server. Its path ends in `/`, so joining a relative
    /// path keeps a proxy's path prefix.
    pub endpoint: Url,
    /// Headers sent with every check, with their values marked sensitive.
    pub headers: HeaderMap,
    /// Time limit for one check.
    pub timeout: Duration,
    /// Tool conditional-execution guardrail priority. Lower values run earlier.
    pub priority: i32,
}

/// An invalid component configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigError {
    /// The config field at fault, when the error concerns one field.
    pub field: Option<&'static str>,
    /// What is wrong.
    pub message: String,
}

impl ConfigError {
    fn new(field: &'static str, message: impl Into<String>) -> Self {
        Self {
            field: Some(field),
            message: message.into(),
        }
    }
}

impl fmt::Display for ConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ConfigError {}

/// The component configuration as written, before validation.
#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RawConfig {
    endpoint: String,
    // Read as JSON and checked by `header_map`, because serde's type errors quote the value.
    headers: Json,
    timeout_millis: u64,
    priority: i32,
    // Reserved by the Relay SDK for its executor settings, which the SDK validates.
    #[serde(rename = "executor")]
    _executor: IgnoredAny,
}

impl Default for RawConfig {
    fn default() -> Self {
        Self {
            endpoint: String::new(),
            headers: Json::Object(Map::new()),
            timeout_millis: DEFAULT_TIMEOUT_MILLIS,
            priority: 0,
            _executor: IgnoredAny,
        }
    }
}

impl ChecksConfig {
    /// Parses and validates one component config, ignoring the SDK's `executor` key.
    pub fn from_component_config(config: &Map<String, Json>) -> Result<Self, ConfigError> {
        let raw = RawConfig::deserialize(config).map_err(|error| ConfigError {
            field: None,
            message: error.to_string(),
        })?;
        Ok(Self {
            endpoint: parse_endpoint(&raw.endpoint)?,
            headers: header_map(&raw.headers)?,
            timeout: parse_timeout(raw.timeout_millis)?,
            priority: raw.priority,
        })
    }
}

fn parse_endpoint(endpoint: &str) -> Result<Url, ConfigError> {
    if endpoint.trim().is_empty() {
        return Err(ConfigError::new("endpoint", "endpoint is required"));
    }
    let mut url = Url::parse(endpoint).map_err(|error| {
        ConfigError::new("endpoint", format!("endpoint is not a valid URL: {error}"))
    })?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(ConfigError::new(
            "endpoint",
            "endpoint must use http or https",
        ));
    }
    // Url's debug output shows the password, while header values are marked sensitive.
    if !url.username().is_empty() || url.password().is_some() {
        return Err(ConfigError::new(
            "endpoint",
            "endpoint must not contain a username or password; send credentials in headers",
        ));
    }
    // Joining a relative path replaces the last path segment unless the path ends in '/'.
    if !url.path().ends_with('/') {
        let path = format!("{}/", url.path());
        url.set_path(&path);
    }
    Ok(url)
}

fn header_map(headers: &Json) -> Result<HeaderMap, ConfigError> {
    let Json::Object(headers) = headers else {
        return Err(ConfigError::new("headers", "headers must be an object"));
    };
    let mut header_map = HeaderMap::new();
    for (name, value) in headers {
        let header_name = HeaderName::from_bytes(name.as_bytes()).map_err(|error| {
            ConfigError::new(
                "headers",
                format!("headers contains invalid header name '{name}': {error}"),
            )
        })?;
        if CLIENT_OWNED_HEADERS.contains(&header_name.as_str()) {
            return Err(ConfigError::new(
                "headers",
                format!(
                    "headers must not set '{header_name}', which the plugin's HTTP client controls"
                ),
            ));
        }
        let Json::String(value) = value else {
            return Err(ConfigError::new(
                "headers",
                format!("headers[{name}] must be a string"),
            ));
        };
        // The error describes the value without quoting it, so a malformed secret stays out of the message.
        let mut header_value = HeaderValue::from_str(value).map_err(|error| {
            ConfigError::new(
                "headers",
                format!("headers[{name}] has an invalid value: {error}"),
            )
        })?;
        // Values such as an authorization token must not appear in debug output or logs.
        header_value.set_sensitive(true);
        if header_map.insert(header_name, header_value).is_some() {
            return Err(ConfigError::new(
                "headers",
                format!("headers sets '{name}' more than once, ignoring case"),
            ));
        }
    }
    Ok(header_map)
}

fn parse_timeout(timeout_millis: u64) -> Result<Duration, ConfigError> {
    if timeout_millis == 0 {
        return Err(ConfigError::new(
            "timeout_millis",
            "timeout_millis must be greater than zero",
        ));
    }
    Ok(Duration::from_millis(timeout_millis))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn parse(config: Json) -> Result<ChecksConfig, ConfigError> {
        ChecksConfig::from_component_config(config.as_object().expect("config is an object"))
    }

    fn parse_error(config: Json) -> ConfigError {
        parse(config).expect_err("config should be rejected")
    }

    /// Only `endpoint` is required; the other fields take their documented defaults.
    #[test]
    fn defaults_apply_when_only_required_fields_are_set() {
        let config = parse(json!({"endpoint": "http://127.0.0.1:8100"})).expect("config is valid");

        assert_eq!(config.endpoint.as_str(), "http://127.0.0.1:8100/");
        assert!(config.headers.is_empty());
        assert_eq!(
            config.timeout,
            Duration::from_millis(DEFAULT_TIMEOUT_MILLIS)
        );
        assert_eq!(config.priority, 0);
    }

    /// Every field is read from the config, and the SDK's `executor` key is ignored.
    #[test]
    fn all_fields_are_read_and_executor_is_ignored() {
        let config = parse(json!({
            "endpoint": "https://guardrails.example.com/base",
            "headers": {"Authorization": "Bearer token"},
            "timeout_millis": 25_000,
            "priority": -5,
            "executor": {"worker_threads": 2}
        }))
        .expect("config is valid");

        assert_eq!(
            config.endpoint.as_str(),
            "https://guardrails.example.com/base/"
        );
        assert_eq!(config.headers["authorization"], "Bearer token");
        assert_eq!(config.timeout, Duration::from_millis(25_000));
        assert_eq!(config.priority, -5);
    }

    /// Joining the hook path keeps the endpoint's path prefix, with or without a trailing slash.
    #[test]
    fn endpoint_path_prefix_survives_joining_the_hook_path() {
        for endpoint in [
            "https://gw.example.com/guardrails",
            "https://gw.example.com/guardrails/",
        ] {
            let config = parse(json!({"endpoint": endpoint})).expect("config is valid");

            assert_eq!(
                config
                    .endpoint
                    .join("v1/hooks/nemo-relay")
                    .expect("the hook path joins")
                    .as_str(),
                "https://gw.example.com/guardrails/v1/hooks/nemo-relay",
                "{endpoint}"
            );
        }
    }

    /// Header values are marked sensitive, so debug output hides them.
    #[test]
    fn header_values_are_sensitive() {
        let config = parse(json!({
            "endpoint": "http://127.0.0.1:8100",
            "headers": {"authorization": "Bearer secret-token"}
        }))
        .expect("config is valid");

        assert!(config.headers["authorization"].is_sensitive());
        assert!(!format!("{config:?}").contains("secret-token"));
    }

    /// Each invalid value is rejected with a message that names its field.
    #[test]
    fn invalid_values_are_rejected_at_their_fields() {
        for (config, field, message) in [
            (json!({}), "endpoint", "endpoint is required"),
            (
                json!({"endpoint": "http://exa mple.com"}),
                "endpoint",
                "endpoint is not a valid URL: ",
            ),
            (
                json!({"endpoint": "localhost:8000"}),
                "endpoint",
                "endpoint must use http or https",
            ),
            (
                json!({"endpoint": "https://user:secret@gw.example.com"}),
                "endpoint",
                "endpoint must not contain a username or password; send credentials in headers",
            ),
            (
                json!({"endpoint": "https://user@gw.example.com"}),
                "endpoint",
                "endpoint must not contain a username or password; send credentials in headers",
            ),
            (
                json!({"endpoint": "http://h", "timeout_millis": 0}),
                "timeout_millis",
                "timeout_millis must be greater than zero",
            ),
            (
                json!({"endpoint": "http://h", "headers": {"bad name": "x"}}),
                "headers",
                "headers contains invalid header name 'bad name': ",
            ),
            (
                json!({"endpoint": "http://h", "headers": {"x-token": "a\nb"}}),
                "headers",
                "headers[x-token] has an invalid value: ",
            ),
            (
                json!({"endpoint": "http://h", "headers": {"X-Token": "a", "x-token": "b"}}),
                "headers",
                "headers sets 'x-token' more than once, ignoring case",
            ),
        ] {
            let error = parse_error(config.clone());

            assert_eq!(error.field, Some(field), "{config}");
            assert!(
                error.message.starts_with(message),
                "{config}: expected {message:?}, got {:?}",
                error.message
            );
        }
    }

    /// An invalid header value is reported without the value itself.
    #[test]
    fn invalid_header_value_is_not_echoed() {
        let error = parse_error(json!({
            "endpoint": "http://h",
            "headers": {"authorization": "Bearer secret-token\n"}
        }));

        assert!(!error.message.contains("secret-token"), "{}", error.message);
    }

    /// Headers that the HTTP client sets itself, or that frame the request, are rejected by name.
    #[test]
    fn client_owned_headers_are_rejected() {
        for name in [
            "Connection",
            "content-length",
            "Content-Type",
            "host",
            "keep-alive",
            "proxy-connection",
            "te",
            "trailer",
            "Transfer-Encoding",
            "upgrade",
        ] {
            let mut headers = Map::new();
            headers.insert(name.to_owned(), json!("x"));

            let error = parse_error(json!({"endpoint": "http://h", "headers": headers}));

            assert_eq!(error.field, Some("headers"), "{name}");
            assert_eq!(
                error.message,
                format!(
                    "headers must not set '{}', which the plugin's HTTP client controls",
                    name.to_ascii_lowercase()
                ),
                "{name}"
            );
        }
    }

    /// Headers of the wrong type are rejected at `headers`, without quoting the value.
    #[test]
    fn mistyped_headers_are_rejected_without_their_values() {
        for (headers, message) in [
            (
                json!({"x-api-key": 8_675_309_123_u64}),
                "headers[x-api-key] must be a string",
            ),
            (json!("Bearer secret-token"), "headers must be an object"),
        ] {
            let error = parse_error(json!({"endpoint": "http://h", "headers": headers}));

            assert_eq!(error.field, Some("headers"), "{headers}");
            assert_eq!(error.message, message, "{headers}");
        }
    }

    /// Unknown fields and values of the wrong type fail parsing, with no single field to blame.
    #[test]
    fn unknown_and_mistyped_fields_are_rejected() {
        for (config, message) in [
            (
                json!({"endpoint": "http://h", "mystery": true}),
                "unknown field `mystery`",
            ),
            (
                json!({"endpoint": "http://h", "timeout_millis": "fast"}),
                "invalid type: string \"fast\"",
            ),
        ] {
            let error = parse_error(config.clone());

            assert_eq!(error.field, None, "{config}");
            assert!(
                error.message.contains(message),
                "{config}: {}",
                error.message
            );
        }
    }
}
