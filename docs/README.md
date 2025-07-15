# Documentation

## Product Documentation

Product documentation for the toolkit is available at
<https://docs.nvidia.com/nemo/guardrails>.

## Building the Documentation

1. Make sure you installed the `docs` dependencies.
   Refer to [CONTRIBUTING.md](../CONTRIBUTING.md) for more information about Poetry and dependencies.

1. Build the documentation:

   ```console
   make docs
   ```

   The HTML is created in the `_build/docs` directory.

## Publishing the Documentation

Tag the commit to publish with `docs-v<semver>`.
Push the tag to GitHub.

To avoid publishing the documentation as the latest, ensure the commit has `/not-latest` on a single line, tag that commit, and push to GitHub.
