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

//! Recent verdicts of the hooks server, so that a tool call checked twice costs one request.
//!
//! Through Relay 0.10.4, a call that reaches a permission prompt runs the guardrail twice: on
//! PreToolUse, and again on PermissionRequest. The cache keeps decisions only, never a failed
//! check, so a failed check is retried on the next call.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use serde_json::Value as Json;
use sha2::{Digest, Sha256};

use crate::client::Verdict;

/// How long a verdict is reused: long enough for the PermissionRequest run, which follows
/// PreToolUse at once, and short enough that a call the agent repeats later reaches the server.
pub const VERDICT_TTL: Duration = Duration::from_secs(10);

/// Most verdicts kept at once.
pub const MAX_VERDICTS: usize = 1024;

/// A tool call as the cache knows it: a SHA-256 digest of its name and arguments, so that every
/// key is small, however large the arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CallKey([u8; 32]);

impl CallKey {
    /// The key of a call. serde_json writes object keys in sorted order, since its
    /// `preserve_order` feature is off, so equal arguments give equal keys.
    pub fn new(tool_name: &str, tool_input: &Json) -> Self {
        let mut digest = Sha256::new();
        // The name, as a JSON string, ends at its closing quote, so the two parts can't run
        // together.
        digest.update(Json::from(tool_name).to_string());
        digest.update(tool_input.to_string());
        Self(digest.finalize().into())
    }
}

/// Verdicts keyed by tool call, each kept for a fixed time.
#[derive(Debug)]
pub struct VerdictCache {
    ttl: Duration,
    capacity: usize,
    entries: HashMap<CallKey, Entry>,
}

/// A verdict and when it was stored.
#[derive(Debug)]
struct Entry {
    verdict: Verdict,
    stored_at: Instant,
}

impl VerdictCache {
    /// An empty cache that keeps up to `capacity` verdicts, each for `ttl`.
    pub fn new(ttl: Duration, capacity: usize) -> Self {
        Self {
            ttl,
            capacity,
            entries: HashMap::new(),
        }
    }

    /// The verdict stored for this call, unless it has expired by `now`.
    pub fn get(&self, key: &CallKey, now: Instant) -> Option<Verdict> {
        self.entries
            .get(key)
            .filter(|entry| entry.is_fresh(self.ttl, now))
            .map(|entry| entry.verdict.clone())
    }

    /// Stores the verdict for this call at `now`, dropping expired verdicts and then the oldest
    /// when the cache is full.
    pub fn insert(&mut self, key: CallKey, verdict: Verdict, now: Instant) {
        if !self.entries.contains_key(&key) && self.entries.len() >= self.capacity {
            self.make_room(now);
        }
        let entry = Entry {
            verdict,
            stored_at: now,
        };
        self.entries.insert(key, entry);
    }

    /// Drops every expired verdict, then the oldest one if the cache is still full.
    fn make_room(&mut self, now: Instant) {
        let ttl = self.ttl;
        self.entries.retain(|_, entry| entry.is_fresh(ttl, now));
        if self.entries.len() < self.capacity {
            return;
        }
        let oldest = self
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.stored_at)
            .map(|(key, _)| *key);
        if let Some(oldest) = oldest {
            self.entries.remove(&oldest);
        }
    }
}

impl Entry {
    /// Whether the verdict is still within `ttl` at `now`.
    fn is_fresh(&self, ttl: Duration, now: Instant) -> bool {
        now.duration_since(self.stored_at) < ttl
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// A deny with a reason, as the hooks server gives it.
    fn deny() -> Verdict {
        Verdict::Deny {
            reason: Some("search is not allowed".into()),
        }
    }

    /// The key of a `search` call with the given query.
    fn search(query: &str) -> CallKey {
        CallKey::new("search", &json!({"query": query}))
    }

    /// A stored allow or deny is returned for the same tool and arguments.
    #[test]
    fn stored_verdict_is_returned_for_the_same_call() {
        let now = Instant::now();
        let mut cache = VerdictCache::new(VERDICT_TTL, MAX_VERDICTS);
        let delete = |path: &str| CallKey::new("delete", &json!({"path": path}));

        cache.insert(search("weather"), Verdict::Allow, now);
        cache.insert(delete("/tmp"), deny(), now);

        assert_eq!(cache.get(&search("weather"), now), Some(Verdict::Allow));
        assert_eq!(cache.get(&delete("/tmp"), now), Some(deny()));
    }

    /// The order of object keys, at any depth, doesn't change a call's key.
    #[test]
    fn object_key_order_does_not_matter() {
        let stored: Json =
            serde_json::from_str(r#"{"query": "weather", "options": {"limit": 3, "lang": "en"}}"#)
                .expect("test arguments are JSON");
        let reordered: Json =
            serde_json::from_str(r#"{"options": {"lang": "en", "limit": 3}, "query": "weather"}"#)
                .expect("test arguments are JSON");

        assert_eq!(
            CallKey::new("search", &stored),
            CallKey::new("search", &reordered)
        );
    }

    /// Another tool name, other arguments, or arguments of another type make another key.
    #[test]
    fn other_calls_have_other_keys() {
        let stored = CallKey::new("search", &json!({"ids": [1, 2]}));

        for (tool_name, tool_input) in [
            ("lookup", json!({"ids": [1, 2]})),
            ("search", json!({"ids": [2, 1]})),
            ("search", json!({"ids": ["1", "2"]})),
            ("search", json!({"ids": [1, 2], "limit": 3})),
            ("search", json!({})),
            ("search", Json::Null),
        ] {
            assert_ne!(
                CallKey::new(tool_name, &tool_input),
                stored,
                "{tool_name} {tool_input}"
            );
        }
    }

    /// The tool name and the arguments can't run together: unquoted, both of these calls would
    /// read `search12`.
    #[test]
    fn name_and_arguments_stay_apart() {
        assert_ne!(
            CallKey::new("search", &json!(12)),
            CallKey::new("search1", &json!(2))
        );
    }

    /// A verdict is returned until its TTL has passed, and not after.
    #[test]
    fn verdict_expires_after_the_ttl() {
        let stored_at = Instant::now();
        let mut cache = VerdictCache::new(Duration::from_secs(60), MAX_VERDICTS);

        cache.insert(search("weather"), Verdict::Allow, stored_at);

        let almost = stored_at + Duration::from_secs(60) - Duration::from_millis(1);
        assert_eq!(cache.get(&search("weather"), almost), Some(Verdict::Allow));
        let expired = stored_at + Duration::from_secs(60);
        assert_eq!(cache.get(&search("weather"), expired), None);
    }

    /// Storing a call again replaces its verdict and restarts its TTL.
    #[test]
    fn storing_again_replaces_the_verdict() {
        let first = Instant::now();
        let mut cache = VerdictCache::new(Duration::from_secs(60), MAX_VERDICTS);

        cache.insert(search("weather"), Verdict::Allow, first);
        let second = first + Duration::from_secs(30);
        cache.insert(search("weather"), deny(), second);

        let after_first_expired = first + Duration::from_secs(60);
        assert_eq!(
            cache.get(&search("weather"), after_first_expired),
            Some(deny())
        );
        assert_eq!(cache.entries.len(), 1);
    }

    /// A full cache drops the oldest verdict to store a new one.
    #[test]
    fn full_cache_drops_the_oldest_verdict() {
        let now = Instant::now();
        let mut cache = VerdictCache::new(VERDICT_TTL, 2);

        for (offset, query) in [(0, "first"), (1, "second"), (2, "third")] {
            let stored_at = now + Duration::from_secs(offset);
            cache.insert(search(query), Verdict::Allow, stored_at);
        }

        let later = now + Duration::from_secs(3);
        assert_eq!(cache.get(&search("first"), later), None);
        for query in ["second", "third"] {
            assert_eq!(
                cache.get(&search(query), later),
                Some(Verdict::Allow),
                "{query}"
            );
        }
        assert_eq!(cache.entries.len(), 2);
    }

    /// A full cache drops every expired verdict at once, not only the oldest.
    #[test]
    fn full_cache_drops_every_expired_verdict() {
        let now = Instant::now();
        let mut cache = VerdictCache::new(Duration::from_secs(60), 3);

        cache.insert(search("first"), Verdict::Allow, now);
        cache.insert(search("second"), Verdict::Allow, now);
        let recent = now + Duration::from_secs(30);
        cache.insert(search("third"), Verdict::Allow, recent);
        let later = now + Duration::from_secs(61);
        cache.insert(search("fourth"), Verdict::Allow, later);

        assert_eq!(cache.entries.len(), 2);
        for query in ["third", "fourth"] {
            assert_eq!(
                cache.get(&search(query), later),
                Some(Verdict::Allow),
                "{query}"
            );
        }
    }
}
