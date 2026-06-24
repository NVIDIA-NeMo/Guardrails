# Path traversal in agent file reads

Source: https://example.org/research/2026-agent-path-traversal

When a coding agent forwards a free-form path into a `read_file` tool, a crafted
value can carry path traversal sequences or shell metacharacters that escape the
intended workspace. The mitigation is to require the `path` argument to fully
match an allowlisted pattern of safe characters before the call runs.

<!-- params: {"arg_name": "path", "pattern": "[\\w./-]+"} -->
