# Outbound requests to forbidden URL patterns

Source: https://example.org/advisories/2026-agent-ssrf-egress

A coding agent exposing an `http_request` tool can be steered into calling a URL
whose host falls into a class that should never be reachable — a cloud metadata
service address or an internal-range host used for SSRF. These are not a short
list of exact values but an open-ended family best described by a regular
expression over the `url` argument. The mitigation is to block the call when the
url pattern matches a forbidden shape, independent of the rest of the request.

<!-- params: {"arg_name": "url", "pattern": "(169\\.254\\.169\\.254|^https?://(localhost|127\\.|10\\.|192\\.168\\.))"} -->
