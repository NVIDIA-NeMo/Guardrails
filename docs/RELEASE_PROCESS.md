# Automated Release Process

This document describes the automated release process for NeMo Guardrails using git-cliff and GitHub Actions.

## Overview

The release process is fully automated and handles:
- Changelog generation using git-cliff
- Version bumping in `pyproject.toml`
- Version updates in `README.md`
- Git tagging
- GitHub release creation

## Prerequisites

1. **Poetry**: This project uses Poetry for dependency management. Ensure Poetry is installed:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Conventional Commits**: All commits should follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:
   - `feat:` - New features (bumps MINOR version)
   - `fix:` - Bug fixes (bumps PATCH version)
   - `docs:` - Documentation changes
   - `chore:` - Maintenance tasks
   - `test:` - Test updates
   - `refactor:` - Code refactoring
   - `perf:` - Performance improvements
   - `style:` - Code style changes
   - `BREAKING CHANGE:` - Breaking changes (bumps MAJOR version)

3. **Branch Protection**: The `main` branch should be protected, and all changes should go through pull requests.

## Release Workflows

### 1. Manual Release (Recommended)

To create a new release:

1. Go to **Actions** → **📦 Release** workflow
2. Click **Run workflow**
3. Enter the new version number (e.g., `0.14.2`)
4. Click **Run workflow**

The workflow will:
- Generate changelog entries for all commits since the last release
- Update `CHANGELOG.md` with new entries (placed after the NOTE section)
- Bump version in `pyproject.toml`
- Update version references in `README.md`
- Create a git tag
- Create a GitHub release with release notes

### 2. Automatic Changelog Updates

The `📝 Update Changelog` workflow runs automatically on every push to `main` and:
- Updates `CHANGELOG.md` with unreleased changes
- Maintains an `[unreleased]` section at the top
- Helps track what will be included in the next release

## Version Bumping Strategy

### Using git-cliff's Auto-bump Feature

If you don't specify a version, git-cliff can automatically determine the next version based on commit types:

- `fix:` commits → PATCH bump (0.14.1 → 0.14.2)
- `feat:` commits → MINOR bump (0.14.1 → 0.15.0)
- `BREAKING CHANGE:` → MAJOR bump (0.14.1 → 1.0.0)

## Changelog Format

The changelog follows the [Keep a Changelog](https://keepachangelog.com/) format with these sections:
- 🚀 Features
- 🐛 Bug Fixes
- 📚 Documentation
- ⚡ Performance
- 🚜 Refactor
- 🎨 Styling
- 🧪 Testing
- ⚙️ Miscellaneous Tasks
- 🛡️ Security
- ◀️ Revert

## Configuration Files

### cliff.toml

The git-cliff configuration is in the root directory and defines:
- Commit parsing rules
- Changelog format
- Version bumping rules
- GitHub issue/PR linking

### GitHub Actions Workflows

- `.github/workflows/release.yml` - Main release workflow
- `.github/workflows/update-changelog.yml` - Automatic changelog updates

## Poetry-Specific Notes

### Version Management

Poetry handles version management through the `poetry version` command, which:
- Updates the version in `pyproject.toml`
- Accepts specific versions: `poetry version 0.14.2`
- Accepts bump rules: `poetry version patch/minor/major`
- Supports pre-release versions: `poetry version prepatch/preminor/premajor`

### Lock File Management

After version updates, always update the lock file:
```bash
poetry lock --no-update  # Updates lock file without updating dependencies
```

This ensures the lock file reflects the new version without changing dependency versions.

### Optional: poetry-bumpversion Plugin

For projects with version references in multiple files, consider using the `poetry-bumpversion` plugin:

```bash
# Install the plugin
poetry self add poetry-bumpversion

# Configure in pyproject.toml
[[tool.poetry_bumpversion.replacements]]
files = ["nemoguardrails/__init__.py"]
search = '__version__ = "{current_version}"'
replace = '__version__ = "{new_version}"'
```

This plugin automatically updates version strings in specified files when using `poetry version`.

## Best Practices

1. **Commit Messages**: Write clear, descriptive commit messages following Conventional Commits
2. **PR Titles**: Use conventional commit format for PR titles if using squash merges
3. **Breaking Changes**: Clearly document breaking changes in commit messages
4. **Release Notes**: The changelog content automatically becomes the GitHub release notes
5. **Lock File**: Always commit `poetry.lock` after version updates to maintain consistency

## Troubleshooting

### Common Issues

1. **Changelog not updating correctly**
   - Ensure commits follow Conventional Commits format
   - Check that git-cliff can parse your commits: `git cliff --unreleased`

2. **Version not bumping correctly**
   - Verify the version format in `pyproject.toml`
   - Check the git-cliff bump configuration in `cliff.toml`

3. **Workflow permissions**
   - Ensure GitHub Actions has write permissions for contents and pull requests
   - Check repository settings → Actions → General → Workflow permissions

## Manual Release (Fallback)

If automation fails, you can still do a manual release:

```bash
# 1. Generate changelog
git cliff --tag v0.14.2 --output CHANGELOG.md

# 2. Update version with Poetry
poetry version 0.14.2
poetry lock --no-update

# 3. Update README.md
sed -i 's/\[0\.[0-9]*\.[0-9]*\]/[0.14.2]/g' README.md

# 4. Commit changes
git add CHANGELOG.md pyproject.toml poetry.lock README.md
git commit -m "chore(release): prepare for v0.14.2"

# 5. Create tag
git tag -a v0.14.2 -m "Release v0.14.2"

# 6. Push
git push origin main --tags
```

## Related Documentation

- [Conventional Commits](https://www.conventionalcommits.org/)
- [Keep a Changelog](https://keepachangelog.com/)
- [git-cliff Documentation](https://git-cliff.org/)
- [Semantic Versioning](https://semver.org/)
