# Confused-deputy push to an unowned remote

Source: https://example.org/cve/2026-confused-deputy-push

A coding agent that exposes a `git_push` tool but performs no check that the
acting principal owns the target remote can be used as a confused deputy: the
agent holds push credentials, and a crafted request gets it to push to a remote
the principal does not own. The mitigation is to require that the `remote`
belongs to the principal before the call runs.

<!-- params: {"arg_name": "remote", "owned_attr": "owned_repos"} -->
