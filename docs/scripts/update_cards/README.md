# Update Cards Script

Automatically updates and generates grid cards in index files based on linked page content.

## Quick Start

```bash
# Preview changes (dry run)
make docs-check-cards

# Apply updates
make docs-update-cards

# Watch mode (auto-update on file changes)
cd docs && python scripts/update_cards/update_cards.py watch

# Generate cards for a new directory
cd docs && python scripts/update_cards/update_cards.py generate ./new-section/
```

## Commands

### Update (Default)

Updates existing grid cards in index files based on the content of linked pages.

```bash
# Update all index files with grid cards
python scripts/update_cards/update_cards.py

# Update specific file(s)
python scripts/update_cards/update_cards.py update configuration-guide/yaml-schema/index.md

# Preview changes (dry run)
python scripts/update_cards/update_cards.py --dry-run --verbose
```

**What it does:**

- Scans index files for Sphinx-Design grid cards (`:::{grid-item-card}`)
- Reads linked pages to extract their title (H1) and description (first paragraph)
- Updates card titles and descriptions to match the linked content

**Options:**

| Option | Description |
|--------|-------------|
| `--dry-run`, `-n` | Show what would change without making changes |
| `--verbose`, `-v` | Show detailed processing output |
| `--docs-dir` | Documentation root directory (default: `../`) |

### Watch Mode

Auto-update cards when files change. Useful during documentation development.

```bash
# Start watching for changes
python scripts/update_cards/update_cards.py watch

# With verbose output
python scripts/update_cards/update_cards.py watch --verbose
```

**Requirements:**

Watch mode requires the `watchdog` package:

```bash
pip install watchdog
# Or with Poetry:
poetry add watchdog --group docs
```

**What it does:**

- Monitors the docs directory for changes to `.md` and `.rst` files
- Automatically updates affected index files when changes are detected
- Includes debouncing to prevent excessive updates during rapid edits

**Example output:**

```text
👀 Watching for changes in: /path/to/docs
   Press Ctrl+C to stop.

Found 5 index file(s) with grid cards.

🔄 Watching for changes...

📝 File changed: configuration-guide/yaml-schema/model-configuration.md
✅ Updated configuration-guide/yaml-schema/index.md:
  - 'Old Title' → 'Model Configuration' (from model-configuration.md)
```

### Generate Cards

Generate grid cards for a directory structure. Useful when creating new documentation sections.

```bash
# Generate cards for a directory
python scripts/update_cards/update_cards.py generate ./getting-started/

# Preview generated markup without writing
python scripts/update_cards/update_cards.py generate ./getting-started/ --dry-run --verbose

# Output to a specific file
python scripts/update_cards/update_cards.py generate ./tutorials/ --output ./tutorials/index.md

# Insert cards after a specific pattern in an existing file
python scripts/update_cards/update_cards.py generate ./advanced/ --insert-after "## Advanced Topics"

# Customize grid layout
python scripts/update_cards/update_cards.py generate ./topics/ --columns "1 2 3 3" --gutter 4
```

**What it does:**

- Scans a directory for documentable files (`.md`, `.rst`, excluding `index.*` and `README.md`)
- Extracts title and description from each file
- Generates Sphinx-Design grid card markup
- Creates or updates the index file

**Options:**

| Option | Description |
|--------|-------------|
| `--output`, `-o` | Output file (default: `directory/index.md`) |
| `--dry-run`, `-n` | Preview without writing |
| `--insert-after` | Pattern to insert cards after (for existing files) |
| `--columns` | Grid columns specification (default: `"1 1 2 2"`) |
| `--gutter` | Grid gutter size (default: `3`) |

**Example generated output:**

```markdown
::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Getting Started
:link: getting-started
:link-type: doc

Learn how to install and configure NeMo Guardrails for your first project.
:::

:::{grid-item-card} Configuration Guide
:link: configuration-guide/index
:link-type: doc

Complete reference for configuring guardrails, models, and behaviors.
:::

::::
```

## Integration Options

### Makefile Targets

The project Makefile includes these targets:

```makefile
docs-update-cards:
    cd docs && python scripts/update_cards/update_cards.py

docs-check-cards:
    cd docs && python scripts/update_cards/update_cards.py --dry-run
```

### Pre-commit Hook

Add to `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: update-doc-cards
        name: Update documentation cards
        entry: python docs/scripts/update_cards/update_cards.py
        language: python
        files: ^docs/.*\.md$
        pass_filenames: false
```

### CI Check

Add a GitHub Actions step to verify cards are up to date:

```yaml
- name: Check documentation cards
  run: |
    cd docs
    python scripts/update_cards/update_cards.py --dry-run
    if [ $? -ne 0 ]; then
      echo "Documentation cards are out of date. Run 'make docs-update-cards'"
      exit 1
    fi
```

### Development Workflow

For active documentation development, use watch mode:

```bash
# Terminal 1: Run documentation server
make docs-serve

# Terminal 2: Auto-update cards
cd docs && python scripts/update_cards/update_cards.py watch
```

## How It Works

### Title Extraction

The script extracts titles using the following priority:

1. **Frontmatter `title` field** (highest priority)
2. Markdown H1: `# Title`
3. RST H1: `Title` followed by `===` underline

### Description Extraction

The script extracts descriptions using the following priority:

1. **Frontmatter `description` field** (highest priority)
2. First non-empty paragraph after the title
3. Skips code blocks, directives, tables, and lists
4. Truncates to ~200 characters if too long
5. Falls back to "Documentation for {title}." if no description found

### Link Resolution

Links in grid cards are resolved relative to the index file:

- `model-configuration` → `./model-configuration.md`
- `../getting-started/index` → `../getting-started/index.md`
- Supports both `.md` and `.rst` files

## Using Frontmatter for Card Content

For precise control over card titles and descriptions, add frontmatter to your pages:

```markdown
---
title: Custom Card Title
description: Custom description for the grid card.
---

# Page Heading (can be different from card title)

Page content...
```

**Example:**

If your page has this frontmatter:

```markdown
---
title: Install
description: Install the toolkit with pip and set up your environment.
---

# Installation Guide

This guide walks you through installing NeMo Guardrails...
```

The generated card will use:

- **Title**: "Install" (from frontmatter, not "Installation Guide")
- **Description**: "Install the toolkit with pip and set up your environment."

This is useful when you want concise card titles that differ from the full page headings.
