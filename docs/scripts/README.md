# Documentation Scripts

This folder contains utility scripts for maintaining the documentation.

## Scripts

### Update Cards

#### Preview changes

```bash
make docs-check-cards
```

#### Apply updates

```bash
make docs-update-cards
```

#### Or directly

```bash
python3 docs/scripts/update_cards.py --dry-run --verbose
python3 docs/scripts/update_cards.py

### `update_cards.py`

Automatically updates grid cards in index files based on the content of linked pages.

**What it does:**

- Scans index files for Sphinx-Design grid cards (`:::{grid-item-card}`)
- Reads linked pages to extract their title (H1) and description (first paragraph)
- Updates card titles and descriptions to match the linked content

**Usage:**

```bash
# Preview changes (dry run)
python scripts/update_cards.py --dry-run --verbose

# Update all index files with grid cards
python scripts/update_cards.py

# Update specific file(s)
python scripts/update_cards.py configuration-guide/yaml-schema/index.md

# Show help
python scripts/update_cards.py --help
```

**Options:**

| Option | Description |
|--------|-------------|
| `--dry-run`, `-n` | Show what would change without making changes |
| `--verbose`, `-v` | Show detailed processing output |
| `--docs-dir` | Documentation root directory (default: `../`) |

**Example output:**

```
Checking 5 index file(s)...

Updated configuration-guide/yaml-schema/index.md:
  - 'Model Configuration' → 'Model Configuration' (from model-configuration.md)
  - 'Old Title' → 'New Title' (from some-page.md)

✅ 2 card(s) updated.
```

## Integration Options

### Pre-commit Hook

Add to `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: update-doc-cards
        name: Update documentation cards
        entry: python docs/scripts/update_cards.py
        language: python
        files: ^docs/.*\.md$
        pass_filenames: false
```

### Makefile Target

Add to the project `Makefile`:

```makefile
docs-update-cards:
 cd docs && python scripts/update_cards.py

docs-check-cards:
 cd docs && python scripts/update_cards.py --dry-run
```

### CI Check

Add a GitHub Actions step to verify cards are up to date:

```yaml
- name: Check documentation cards
  run: |
    cd docs
    python scripts/update_cards.py --dry-run
    if [ $? -ne 0 ]; then
      echo "Documentation cards are out of date. Run 'python docs/scripts/update_cards.py'"
      exit 1
    fi
```

## Adding Description Metadata

For more control over card descriptions, you can add frontmatter to your pages:

```markdown
---
description: Custom description for the grid card.
---

# Page Title

Page content...
```

The script will use the frontmatter description if available (future enhancement).
