# Writes outside the principal's workspace prefix

Source: https://example.org/cve/2026-workspace-prefix-bypass

A coding agent exposing a `write_file` tool can be driven to write to a path
outside the directory namespace allocated to the acting principal. Each
principal is scoped to one or more workspace prefixes, and a crafted absolute or
relative path can land the write in another principal's tree. Because the allowed
set is expressed as path prefixes rather than a fixed list of files, the
mitigation is to require the `path` argument to begin with one of the principal's
allocated workspace prefixes before the call runs.

<!-- params: {"arg_name": "path", "owned_attr": "owned_paths"} -->
