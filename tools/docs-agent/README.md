<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Short-lived documentation agents

Guardrails uses separate short-lived Pi sandboxes for post-merge documentation authoring and pull
request documentation review. A sandbox exists only for one workflow job and is deleted after the
agent finishes or fails. The workflows do not provide the sandbox with a GitHub token or the upstream
model credential.

## Repository configuration

Repository administrators must configure:

- the `DOCS_AGENT_API_KEY` Actions secret for the OpenAI-compatible NVIDIA inference endpoint;
- the existing `DOCS_FERN_TOKEN` Actions secret for host-side Fern validation; and
- the `DOCS_MAINTAINERS` Actions variable as a comma-separated list of GitHub usernames that should
  review generated fast-follow documentation pull requests.

The OpenShell release archives, Pi sandbox image, model identifier, rubric copies, and workflow
actions are pinned in the repository. Update these pins through normal dependency and security review.

## Post-merge fast follow

`Docs / Author Post-Merge Fast Follow` runs when a non-documentation pull request merges into
`develop`. The selection job records the source pull request number, author, exact base and head
commits, and merge commit. It does not use a release tag or an earlier managed documentation pull
request as a range boundary.

The model-bearing job performs these steps:

1. Prepare the merged tree and the exact source pull request diff as inert data.
2. Start a fresh author sandbox with no direct network access.
3. Permit writes only while authoring and restrict the exported patch to `docs/**`, `fern/docs.yml`,
   and `fern/assets/**`.
4. Start a second, read-only sandbox to review the exact source diff and candidate patch.
5. Run host-side documentation validation when the approved patch is not empty.
6. Delete each sandbox and upload a revision-bound artifact.

The separate publisher has GitHub write permission but receives neither the model credential nor a
model sandbox. It applies the approved patch to the current `develop` branch, creates one draft pull
request for the triggering development pull request, and requests review from the configured
maintainers and the development pull request author. An empty approved patch creates no pull request.
The branch includes the source pull request number and merge commit, so different development pull
requests never accumulate into the same documentation pull request.

## Documentation review

`Docs / Review Documentation` runs automatically when a pull request changes public documentation or
its checked-in build and navigation surfaces. An owner, member, or collaborator can also add a pull
request comment containing exactly `/review-doc`. Maintainers can use manual workflow dispatch with a
pull request number.

The workflow checks out trusted reviewer code from the base revision and prepares the pull request as
inert data. The Pi sandbox mounts the pull request read-only, has no direct network policy, and receives
only read, search, and output-writing tools. It must not execute repository scripts, tests, package
managers, or changed workflow code.

The reviewer selects the task-guide, concept-guide, or API-reference rubric that matches the primary
changed content. The six weighted dimensions total 100 points. Hard-gate findings force a failing
decision; lower-confidence or below-threshold results use `human-review`. The trusted host validates
the score arithmetic, rubric maxima, hard gates, finding locations, and exact pull request revision.

A separate publisher rechecks the pull request head and posts or replaces one advisory comment. The
comment includes the score, dimension table, rationale, findings, reviewed commit, and workflow link.
The review never approves, requests changes, merges, labels, or pushes to the pull request.

## Recovery

- If the source branch, merge commit, or pull request head no longer matches the artifact, rerun the
  workflow. The publisher refuses stale output.
- If a managed documentation branch exists without an open pull request, inspect and remove or recover
  that exact branch before rerunning. The publisher does not overwrite it.
- If a sandbox cleanup fails, remove only the sandbox named in the workflow log before rerunning.
- If inference is unavailable, the model-bearing job fails closed and publishes no patch or review.
- If the configured maintainer list is empty or invalid, publication stops before creating a branch.
