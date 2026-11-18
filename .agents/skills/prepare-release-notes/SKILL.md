---
name: prepare-release-notes
description: Prepare NeMo Guardrails release documentation from the Prepare Release changelog PR. Use for a minor-release page refresh or patch notes, a Fern CLI and docs version bump, an immutable SDK-backed snapshot, and verification of the versioned PR preview.
license: Apache-2.0
---

# Prepare Guardrails release notes

Start when the **Prepare Release** GitHub Action opens or updates its release PR.
A version such as `X.Y.Z` or a link to that PR is sufficient to begin.
Follow the repository and documentation agent instructions. A request for a draft
can be completed locally. For an end-to-end release-docs request, carry the
authorized tag and PR work through verification. The development release workflow
owns the package version, generated changelog, package tag, and distribution.

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
   stable release tag or exact release merge when available. For a pre-merge
   snapshot requested from latest `develop`, fetch upstream and record its
   exact SHA. Compare the release's product changes with that source before
   freezing it. Surface material differences. If the snapshot base remains
   undecided, prepare the page and CLI changes while it is being settled.

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

## Upgrade Fern and register the docs version

Run `fern upgrade` with the repository-supported Node.js version (currently 22).
Inspect the migration diff and record the old and new CLI pins in
`fern/fern.config.json`. Commit the upgrade separately when it produces changes.
Apply the same pin and any required migration changes in the snapshot. When
resuming, check for an existing upgrade commit before running the upgrade again.

In the docs PR, add the new entry immediately after `Latest` in `fern/docs.yml`
and mirror it in the versioning example in `docs/README.mdx`. Keep older entries
and use the same display name and slug, `vX.Y.Z`, in both lists. The `ref` is the
actual permanent docs tag, including its revision suffix if one is needed.
`Latest` continues to use the working-tree navigation.

## Freeze the documentation and Python SDK source

Follow **Creating a Curated Snapshot** in `docs/README.mdx`. The version
selector loads pages from the **docs snapshot tag**. Inside that snapshot,
`libraries.guardrails-python-sdk.input.ref` selects the **product source**
used to generate its historical Python SDK reference.

Before creating a commit or tag, inspect the registered ref and the remote
`fern-docs-snapshot-vX.Y.Z` tags, including revision suffixes. Reuse a verified
snapshot if its source, notes and Fern configuration match. For an editorial
correction, keep the original product SHA and SDK pin, carry forward any
snapshot-specific fixes, and choose the next unused `-r2`, `-r3`, … suffix.
Use the unsuffixed name for the first snapshot. Published tags are immutable.

1. Create a separate worktree at the chosen source SHA. Bring in the reviewed
   notes and the Fern upgrade. In that worktree's `fern/docs.yml`, set
   `libraries.guardrails-python-sdk.input.ref` to the full source SHA or the
   immutable package tag resolving to it. Keep this pin in the historical
   worktree only. Keep the snapshot's inherited registry free of a reference to
   itself. The build generates OpenAPI and SDK pages from source.
2. Run pre-commit on the changed snapshot files and create a DCO-signed commit
   with `git commit -s`. Verify its parent is the selected source and compare
   its notes and CLI version with the PR. Create the chosen annotated tag on
   that **snapshot commit** with `git tag -a "$tag" HEAD -m "Fern docs snapshot for $tag"`
   from the snapshot worktree.
3. With authorization to publish the snapshot, push that specific tag to the
   upstream repository. Verify the remote commit before pushing a PR branch
   that references it. With `tag` set to the chosen docs tag, compare
   `git ls-remote --tags origin "refs/tags/$tag" "refs/tags/$tag^{}"` with
   `git rev-parse "$tag^{}"` (verify `origin` points upstream first).
   If another publisher claims that name first, inspect its target before
   reusing it or selecting a revision. Update both version lists to the
   verified tag.

## Validate and hand off

- Run `git diff --check`, `uv run --locked pre-commit run --files <changed-files>`,
  and `make docs-fern` with the repository's Node.js version. Check the new
  page's links and anchors, each displaced version's URL, and agreement between
  the two version lists. Compare the snapshot content with the PR again after
  review edits.
- When asked to publish, push the docs branch directly to
  `NVIDIA-NeMo/Guardrails` and open or update its PR against `develop`.
  An upstream head enables the repository's Fern PR preview. Use the PR
  template and record the release PR, source and snapshot SHAs, docs tag,
  CLI change, checks, and AI assistance. Keep a pre-merge snapshot's
  provenance explicit and reconcile it against the eventual release cut.
- Follow **Check Fern docs** and **Publish Fern preview** on the current PR
  head. A successful check alone does not exercise remote snapshot resolution.
  Open the preview comment's URL at
  `/vX.Y.Z/about-nemo-guardrails-library/release-notes` under its site root.
  Verify the version selector, new content, and historical SDK reference.
  For a missing ref, inspect the upstream tag and peeled SHA. For an SDK
  generation failure, inspect the pin inside the tagged tree. Retry after
  correcting the cause. Report the PR, tag, and verified preview.
