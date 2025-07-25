# Release Checklist

## Quick Release via GitHub Actions

1. Go to **Actions** → **📦 Release** workflow
2. Click **Run workflow**
3. Enter the new version (e.g., `0.14.2`)
4. Click **Run workflow**

✅ The workflow will automatically:
- Generate changelog using git-cliff
- Update version with Poetry
- Update README.md version references
- Commit all changes
- Create and push git tag
- Create GitHub release

## Pre-Release Checklist

- [ ] All tests passing on main branch
- [ ] No outstanding critical issues
- [ ] Dependencies are up to date
- [ ] Documentation is current

## Post-Release Checklist

- [ ] Verify GitHub release was created
- [ ] Check PyPI package publication (if configured)
- [ ] Update any external documentation
- [ ] Announce release in relevant channels

## Manual Release Commands

If you need to release manually:

```bash
# Update version
poetry version 0.14.2
poetry lock --no-update

# Generate changelog
git cliff --tag v0.14.2 -o CHANGELOG.md

# Update README
sed -i 's/\[0\.[0-9]*\.[0-9]*\]/[0.14.2]/g' README.md

# Commit and tag
git add -A
git commit -m "chore(release): prepare for v0.14.2"
git tag -a v0.14.2 -m "Release v0.14.2"
git push origin main --tags
```

## Troubleshooting

- **Workflow fails**: Check GitHub Actions logs
- **Version conflict**: Ensure no duplicate version tags exist
- **Poetry errors**: Run `poetry install` to sync dependencies
