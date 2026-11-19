// The Cedar form of examples/configs/agent_hooks/agent_start/config.yml. Nothing starts unless a policy permits it,
// so the first policy permits every start, and each forbid takes one rule away.

@id("start/permit")
permit(principal, action == Hooks::Action::"start", resource);

// allowed_sources: parts may come only from the approved repository.
@id("parts/approved-source")
forbid(principal, action == Hooks::Action::"start", resource is Hooks::Part)
unless { resource.local || resource.source like "git::https://git.example.com/workshop/agent-parts.git//*" };

// allow_local_sources: false. A local folder is a draft, not the approved repository.
@id("parts/no-local-folder")
forbid(principal, action == Hooks::Action::"start", resource is Hooks::Part)
when { resource.local };

// require_digests: true. Every part must say which bytes it is.
@id("parts/digest")
forbid(principal, action == Hooks::Action::"start", resource is Hooks::Part)
unless { resource has digest };

// allowed_models: the agent, and every agent it calls, may name only this model.
@id("models/listed")
forbid(principal, action == Hooks::Action::"start", resource is Hooks::Model)
unless { ["bedrock::nvidia.nemotron-super-3-120b"].contains(resource.name) };

// require_policy_enforcers: true. A required policy starts only where every enforcer it needs runs.
@id("policies/enforcers-present")
forbid(principal, action == Hooks::Action::"start", resource is Hooks::Policy)
when { resource.required && (resource.enforcers.isEmpty() || !context.enforcers_present.containsAll(resource.enforcers)) };
