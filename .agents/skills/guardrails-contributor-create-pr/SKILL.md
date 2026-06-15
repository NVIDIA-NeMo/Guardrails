---
name: "guardrails-contributor-create-pr"
description: "Drafts pull request text and verification summaries for Guardrails contributions without submitting them. Use when preparing a PR, PR description, PR title, verification checklist, DCO sign-off guidance, or review-readiness summary. Trigger keywords - create PR, pull request, PR draft, submit for review, PR template, verification checklist, DCO."
license: "Apache-2.0"
---

# Draft Pull Request Materials

Use this skill to prepare pull request materials for a human contributor.
Do not open issues, PRs, draft PRs, or submit GitHub changes through browser automation, the GitHub API, `gh`, or similar tooling.

`CONTRIBUTING.md` is canonical for public contribution workflow.
`AI_POLICY.md` is canonical for AI-assisted contribution policy.
`AGENTS.md` is canonical for repository-specific agent constraints.

## Required Boundaries

- Draft text only.
- Do not push branches unless the user explicitly asks and the repository rules allow it.
- Do not prepare public-submission-ready PR materials unless the linked issue is triaged and assigned to the human contributor.
- Do not edit `CHANGELOG.md` or `CHANGELOG-Colang.md`.
- Do not add AI tools as commit co-authors.
- Do not claim tests, reviews, approvals, or validation that did not happen.

## Gather Context

Inspect the local branch and summarize:

- The target branch, normally `develop`.
- The changed files and the behavior they affect.
- Whether the work is linked to a triaged issue assigned to the human contributor.
- Which validation commands were actually run.
- Whether docs were updated or intentionally not needed.

Use read-only GitHub checks only when needed to identify duplicate or in-flight work, as described in `AGENTS.md`.
Do not use write-side GitHub automation.

## Draft A Conventional Title

Use the Conventional Commit-style title from `CONTRIBUTING.md`.
Common types include:

- `fix: ...`
- `feat: ...`
- `docs: ...`
- `test: ...`
- `refactor: ...`
- `perf: ...`
- `style: ...`
- `chore: ...`
- `ci: ...`
- `revert: ...`

Use a scope only when it clarifies the changed area, such as `fix(server): ...` or `docs: ...`.

## Draft The PR Body

Follow `.github/PULL-REQUEST-TEMPLATE.md`.
Draft these sections for the human to review:

```markdown
## Description

<!-- Describe the big picture of your changes to communicate to the maintainers
  why we should accept this pull request. Include any areas that need careful
  review. -->

## Related Issue(s)

<!--
  Every PR must link to a triaged issue assigned to the PR author.
  - Fixes #<issue_number>
  - Issue assignee: @<username>
-->

## Verification

<!-- CI runs the automated test suite. Note any verification beyond CI (manual
  checks, live/credentialed paths, docs build) and any relevant check you could
  not run, with residual risk. -->

## AI Assistance

- [ ] No AI tools were used.
- [ ] AI tools were used; a human reviewed and can explain every change (tool: ___).

## Checklist

- [ ] I've read the CONTRIBUTING guidelines.
- [ ] This PR links to a triaged issue assigned to me.
- [ ] My PR title follows the project commit convention.
- [ ] I've updated the documentation if applicable.
- [ ] I've added tests if applicable.
- [ ] I've noted any verification beyond CI and any checks I couldn't run.
- [ ] I did not update generated changelog files manually.
- [ ] I addressed all CodeRabbit, Greptile, and other review comments, or replied with why no change is needed.
- [ ] @mentions of the person or team responsible for reviewing proposed changes.
```

### Description

Write one to three sentences that explain what changed and why.
Include any areas that need careful maintainer review.

### Related Issue(s)

Include the linked issue if known.
If the issue is not triaged and assigned to the human contributor, state that the PR should not be submitted yet.

### Verification

List only commands or checks that actually ran.
Use the validation guidance from `AGENTS.md` and `CONTRIBUTING.md`.
For docs-only changes, include docs checks when rendering, links, examples, or docs configuration may be affected.

### AI Assistance

If AI materially helped, include a concise disclosure for the human to review and edit.
Do not fabricate human review.

### Checklist

Mark only checkboxes supported by actual evidence.
Leave unchecked any item the human still needs to verify.
Remind the human contributor that public contributions must satisfy the DCO through signed commits or a `Signed-off-by` line, as described in `CONTRIBUTING.md`.

## Review Readiness

Before saying a draft is review-ready, verify the draft accounts for:

- Open automated review comments if the branch already has a PR.
- Open human review comments.
- Required docs updates for user-visible changes.
- No secrets, credentials, provider data, or generated assets.
- No manual changelog edits.

## Output Format

Return:

1. A proposed PR title.
2. A copyable PR body draft.
3. A short validation summary listing what ran and what remains.
4. Any blockers that prevent human submission.
