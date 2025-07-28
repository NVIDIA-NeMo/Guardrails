# Release Checklist

## Quick Release via GitHub Actions

1. Go to **Actions** → **🚀 Prepare Release** workflow
2. Click **Run workflow**
3. Enter the new version (e.g., `0.15.1`)
4. Click **Run workflow**

✅ The automated release process:

- **🚀 Prepare Release**: Generates changelog, updates version, creates PR
- **🏷️ Create Release Tag**: Auto-creates tag when PR is merged
- **Build and Test Distribution**: Auto-builds artifacts on tag creation
- **📦 Publish to PyPI**: Auto-publishes to PyPI (can be triggered manually)

## Pre-Release Checklist

- [ ] All tests passing on develop branch
- [ ] No outstanding critical issues
- [ ] Dependencies are up to date
- [ ] No high-severity security vulnerabilities
- [ ] No pending PRs in the release milestone

## Post-Release Checklist

- [ ] Review the Pull Request created by the release workflow
- [ ] Merge the Pull Request (triggers automatic tag creation)
- [ ] Verify artifacts built successfully in **Build and Test Distribution** workflow
- [ ] Optionally trigger **📦 Publish to PyPI** workflow manually for PyPI release

## Manual Release Commands

If you need to release manually:

```bash
# checkout to prep release branch
git checkout -b chore/release-v0.15.1
# update version
poetry version 0.15.1

# generate changelog
git cliff \
  --latest \
  --tag v0.15.1 \
  --strip all \
  --prepend CHANGELOG.md

# move the generated diff above the prev entry

# update README
sed -i 's/\[0\.[0-9]*\.[0-9]*\]/[0.15.1]/g' README.md

# commit and tag
git add -A
git commit -m "chore(release): prepare for v0.15.1"

# go ahead and open a PR
# once the PR is merged, a workflow will be triggered to create a release tag

```
