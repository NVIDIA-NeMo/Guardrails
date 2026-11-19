---
name: prepare-release-notes
description: Prepare NeMo Guardrails release documentation from the Prepare Release changelog PR. Use to refresh minor or patch notes, upgrade Fern, register the reviewed snapshot in a separate PR, and verify staged or public release docs.
license: Apache-2.0
---

# Prepare Guardrails release notes

Start when the **Prepare Release** GitHub Action opens or updates its release PR.
A version such as `X.Y.Z` or a link to that PR is sufficient to begin.
Follow the repository and documentation agent instructions. A request for a draft
can be completed locally. For a release, merge the notes PR first, freeze the
merged content, then register the tag in a second PR. Verify staging before
publishing to production. Carry out only the stages the operator requested.
The development release workflow owns the package version, generated changelog,
package tag, and distribution.

## Establish the release cut

1. Read `.github/workflows/release.yml`. Using the operator's authenticated
   GitHub CLI, locate the bot's `chore/release-X.Y.Z` PR targeting `develop`.
   Confirm its version against the new `CHANGELOG.md` heading and
   `pyproject.toml`. Capture its URL, state, and head SHA (or merge SHA).
   Read the version block **at that revision**, including human edits made
   after generation. If the head changes during preparation, compare the two
   revisions and revise the notes before handoff.
2. Find any in-flight docs PR for that version and check the existing Fern
   registry and remote snapshot tags. Resume matching work on its verified
   branch. For a new docs PR, branch from freshly fetched upstream `develop`
   in a clean worktree and follow the repository's linked-issue checks.
3. Choose the **snapshot source** separately from the changelog source. Use the
   stable release tag or exact release merge when available. If the operator
   names `develop` as the product source, fetch upstream and record its exact
   SHA. Reconcile the selected SHA with the final product cut. The docs PR's
   merge SHA supplies the reviewed page and Fern changes.

## Write the current release page

Read `docs/about/release-notes.mdx` and the most recent comparable minor or patch
release diff. Search the release's documentation tracking ticket (including Jira
comments when available) for promised bullets and follow-ups. Check candidate
claims against code, tests, guides, and linked PRs at the cut. In particular,
distinguish native integrations from plugin or middleware support, and retain
experimental status, supported modes, and relevant limits. Keep internal tracker
links and planning text out of public copy.

- **New minor series** (`0.X.0`). Replace the prior series' body with the new
  notes. Move the displaced minor and patch releases under **Previous Releases**
  as links to their respective versioned release-note pages. Preserve older
  links, frontmatter, and the existing heading and anchor pattern.
- **Patch in the same series** (`0.X.Y`, `Y > 0`). Add a concise patch section
  above that series' existing notes. On a rerun, update the existing section.
- Check every changelog group for breaking changes, including fixes and
  dependency updates. Give each one an upgrade action. Group related features
  by what users can do, and cover consequential fixes and deprecations.
  Link to canonical docs for setup and migration. Check for previously promised
  announcements even when the same capabilities appear across several PRs.

## PR 1. Merge the release notes and Fern upgrade

Run `fern upgrade` with the repository-supported Node.js version (currently 22).
Inspect the migration diff and record the old and new CLI pins in
`fern/fern.config.json`. Commit the upgrade separately when it produces changes.
Apply the same pin and any required migration changes in the snapshot. When
resuming, check for an existing upgrade commit before running the upgrade again.

Keep version registration for PR 2. For each PR, run `git diff --check`,
`uv run --locked pre-commit run --files <changed-files>`, and `make docs-fern`.
When authorized, open or update the PR against upstream `develop` from a branch
in `NVIDIA-NeMo/Guardrails` so CI can publish a Fern preview. Use the PR template
and record the release PR, revisions, CLI change, checks, and AI assistance.
Review the `Latest` preview, links, anchors and displaced versions. After PR 1
merges, fetch its actual merge commit and record it as `notes_merge`. Use this
merged revision for the permanent snapshot, including edits made during review.

## Freeze the documentation and Python SDK source

Follow **Creating a Curated Snapshot** in `docs/README.mdx`. The version
selector loads pages from the **docs snapshot tag**. Inside that snapshot,
`libraries.guardrails-python-sdk.input.ref` selects the **product source**
used to generate its historical Python SDK reference.

Before creating a commit or tag, inspect the registered ref and the remote
`fern-docs-snapshot-vX.Y.Z` tags, including revision suffixes. If a published
snapshot matches the product source, merged notes and Fern configuration,
verify its remote ref and proceed directly to registration. For an editorial
correction, keep the original product SHA and SDK pin, carry forward any
snapshot-specific fixes, and choose the next unused `-r2`, `-r3`, … suffix.
Use the unsuffixed name for the first snapshot. Published tags are immutable.

1. Create a separate worktree at the chosen product SHA. Bring in the release
   page from `notes_merge` and apply the merged Fern pin and required migrations.
   In that worktree's `fern/docs.yml`, set
   `libraries.guardrails-python-sdk.input.ref` to the full source SHA or the
   immutable package tag resolving to it. Keep this pin in the historical
   worktree only. Keep the snapshot's inherited registry free of a reference to
   itself. The build generates OpenAPI and SDK pages from source.
2. Run pre-commit on the changed files and `make docs-fern` in the snapshot.
   Create a DCO-signed commit with `git commit -s`. Verify its parent and SDK
   pin against the selected product source and its CLI pin against PR 1.
   Run `git diff --exit-code "$notes_merge" HEAD -- docs/about/release-notes.mdx`
   from the snapshot worktree. Once the reviewed page matches and snapshot
   review is complete, set `tag` to the chosen docs tag. Create the authorized
   annotated tag on that commit with `git tag -a "$tag" HEAD -m "Fern docs snapshot for $tag"`.
3. Confirm `origin` points upstream. When authorized, push that tag with
   `git push origin "refs/tags/$tag"`. Before pushing a branch that references
   it, compare the remote tag object and peeled commit from
   `git ls-remote --tags origin "refs/tags/$tag" "refs/tags/$tag^{}"` with
   `git rev-parse "$tag" "$tag^{}"`.
   If another publisher claims that name first, inspect its target before
   reusing it or selecting a revision. Pushing a snapshot tag makes it
   available to Fern. Publication uses the version registry in the checkout.

## PR 2. Register the verified snapshot

Branch from updated upstream `develop`. In `fern/docs.yml`, add `vX.Y.Z`
immediately after `Latest`, or update the existing entry's `ref` for a
correction. Mirror the entry in the example in `docs/README.mdx`. Keep the
display name and slug at `vX.Y.Z` even for an `-r2` tag. Record the tag and
its peeled commit in the PR. Follow **Check Fern docs** and **Publish Fern
preview** on the PR head. Open the version selector in the published preview
and visit the historical Python SDK reference.

On the preview, compare the release section served at both
`/latest/about-nemo-guardrails-library/release-notes` and
`/vX.Y.Z/about-nemo-guardrails-library/release-notes` with the merged notes.
Use the rendered body or append `.md` to compare text across version-specific
links. If only `Latest` shows a review edit, check the version entry in the
checkout being published, resolve its remote tag, and inspect the page in that
tag. A rebuild with the old ref will serve the old page again. For SDK issues,
inspect the SDK source pin inside the tagged tree.

## Verify staging and publish to production

1. After PR 2 merges, find the `develop` push run for its merge SHA in
   `.github/workflows/docs-build.yaml`. Wait for **Publish Fern staging docs**
   to succeed. For a requested manual staging publish, confirm the checkout's
   version entry points to the verified tag and run `make docs-fern-publish-staging`
   from the repository root. Both publish targets regenerate OpenAPI and the
   Python SDK reference. A Fern 403 during API upload requires an account with
   publishing access to the Fern organization.
2. Repeat the release-section comparison on the staging site and open the
   versioned SDK page there. Record the staging URLs and the source commit.
   If a later `develop` deployment supersedes the run, identify the revision
   actually being served before promoting it.
3. When production publication is authorized, use a clean checkout of the
   commit verified on staging and run `make docs-fern-publish-public`. Verify
   the same two release-note routes and versioned SDK page on the public site.
   The checked-in workflow has no public tag-publish job. If using a separate
   `docs-v*` publishing integration, confirm its trigger with the maintainer
   as described in `docs/README.mdx`.
