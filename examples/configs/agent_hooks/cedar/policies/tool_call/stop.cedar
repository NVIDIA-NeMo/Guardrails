// stop_tools. A forbid with @outcome("stop") ends the run, not only the call; @outcome("quarantine") would also
// quarantine the agent, and needs hooks_server.quarantine_store.

@id("stop/delete-index")
@outcome("stop")
forbid(principal, action == Hooks::Action::"call_tool", resource == Hooks::Tool::"delete_index");

@id("stop/admin-override")
@outcome("stop")
forbid(principal, action == Hooks::Action::"call_tool", resource == Hooks::Tool::"search")
when { context.argument_names.contains("admin_override") };
