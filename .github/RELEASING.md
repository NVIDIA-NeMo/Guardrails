# Releasing

Maintainer instructions for publishing a NeMo Guardrails release.

After merging the release-preparation PR, wait for **Build and Test
Distribution** to succeed on the merge commit and for the `vX.Y.Z` tag to
exist. The branch build's `pr-build` artifact contains both distributions;
a separate tag build is not required.

In GitHub Actions, select **Publish to PyPI (with Approval)**, choose the
repository's default branch (`develop`), and click **Run workflow**. Enter:

- `version`: the stable version without `v`, for example `0.25.0`.
- `build_run_id`: leave blank to select a build automatically. To select a
  specific build, enter the number after `/actions/runs/` in that build's URL.
- `validate_only`: leave checked for the first run. This performs the build
  and distribution checks, then skips PyPI upload and GitHub release creation
  without requesting deployment approval.

Automatic selection considers successful distribution builds for the release
tag's exact commit, newest first. It can try an older successful build for that
same commit if the newer build's artifact is missing or expired. Other
validation failures stop the workflow. An explicit build run ID is always used
exactly, with no fallback to automatic selection. The summary links to the
selected build.

The workflow accepts successful push or scheduled distribution builds from
`main`, `develop`, or the matching release tag in this repository. It verifies
that the build commit matches the tag, every required wheel test passed, and
the source, wheel, and source archive versions match. It downloads the artifact
by its immutable ID without installing or executing the package. If the
distribution test matrix changes, update `EXPECTED_JOBS` in
`.github/scripts/validate_pypi_release.py` alongside it.

After a successful validation-only run, review its summary. To publish, start
a new run with the same version, optionally copy the validated build's ID to
keep that exact selection, and uncheck `validate_only`. Review the new summary,
then approve the `pypi-production` deployment. The workflow rechecks the tag,
uploads the validated distributions to PyPI, and creates a GitHub release draft
if one does not already exist.
Review and publish that draft separately.

Validation-only mode does not test PyPI authentication. The workflow must be
merged into the default branch before it can be tested through GitHub Actions;
dispatches from other branches are skipped. If the selected artifact expires
or is deleted while approval is pending, the upload job fails safely; start
a new validation run to select an available build.

One-time setup:

- In the PyPI project's **Publishing** settings, register a GitHub Trusted
  Publisher for owner `NVIDIA-NeMo`, repository `Guardrails`, workflow filename
  `publish-pypi-approval.yml`, and environment `pypi-production`.
- In GitHub repository **Settings → Environments**, configure required reviewers
  for `pypi-production`. The workflow's environment declaration alone does not
  require approval.

The upload uses GitHub's OIDC identity; no `PYPI_API_TOKEN` secret is required.
See [PyPI's Trusted Publisher setup](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).

If validation fails, correct the selected version/build or obtain a new
successful build when artifacts have expired. If only GitHub release creation
fails after a successful upload, rerun **only the failed job**. If the upload
itself fails, inspect PyPI before retrying: files may have been partially
uploaded, and the workflow does not skip existing distributions.
