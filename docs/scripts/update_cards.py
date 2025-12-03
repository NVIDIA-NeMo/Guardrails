#!/usr/bin/env python3
"""
Automatically update grid cards in index files based on linked page content.

This script scans index files for Sphinx-Design grid cards, reads the linked
pages to extract their title and description, and updates the cards accordingly.

Usage:
    python update_cards.py [--dry-run] [--verbose] [path/to/index.md ...]

Examples:
    # Update all index files in docs/
    python update_cards.py

    # Update specific index file
    python update_cards.py ../configuration-guide/yaml-schema/index.md

    # Preview changes without writing
    python update_cards.py --dry-run --verbose
"""

import argparse
import re
import sys
from pathlib import Path
from typing import NamedTuple


class CardInfo(NamedTuple):
    """Information about a grid card."""

    title: str
    link: str
    link_type: str
    description: str
    start_line: int
    end_line: int
    original_text: str


class PageInfo(NamedTuple):
    """Information extracted from a linked page."""

    title: str
    description: str
    path: Path


def extract_page_info(file_path: Path) -> PageInfo | None:
    """Extract title and description from a markdown/rst file."""
    if not file_path.exists():
        return None

    content = file_path.read_text(encoding="utf-8")
    lines = content.split("\n")

    title = None
    description = None

    # Skip frontmatter if present
    start_idx = 0
    if lines and lines[0].strip() == "---":
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == "---":
                start_idx = i + 1
                break

    # Extract title (first H1)
    for i, line in enumerate(lines[start_idx:], start_idx):
        stripped = line.strip()

        # Markdown H1: # Title
        if stripped.startswith("# ") and not stripped.startswith("##"):
            title = stripped[2:].strip()
            start_idx = i + 1
            break

        # RST H1: Title followed by === underline
        if i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if (
                next_line
                and all(c == "=" for c in next_line)
                and len(next_line) >= len(stripped)
            ):
                title = stripped
                start_idx = i + 2
                break

    if not title:
        return None

    # Extract description (first non-empty paragraph after title)
    description_lines = []
    in_code_block = False
    in_directive = False

    for line in lines[start_idx:]:
        stripped = line.strip()

        # Skip code blocks
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue

        # Skip directives (MyST ::: or RST ..)
        if stripped.startswith(":::") or stripped.startswith(".. "):
            in_directive = True
            continue
        if in_directive:
            if not stripped:
                in_directive = False
            continue

        # Skip admonitions and notes
        if stripped.startswith("{") or stripped.startswith("```{"):
            continue

        # Skip section headers
        if stripped.startswith("#") or stripped.startswith("=="):
            break

        # Skip horizontal rules
        if stripped == "---":
            continue

        # Collect paragraph lines
        if stripped:
            # Skip if it looks like a table or list
            if (
                stripped.startswith("|")
                or stripped.startswith("-")
                or stripped.startswith("*")
            ):
                if not description_lines:
                    continue
                break
            description_lines.append(stripped)
        elif description_lines:
            # End of paragraph
            break

    if description_lines:
        description = " ".join(description_lines)
        # Truncate if too long (aim for ~150 chars)
        if len(description) > 200:
            description = description[:197].rsplit(" ", 1)[0] + "..."
    else:
        description = f"Documentation for {title}."

    return PageInfo(title=title, description=description, path=file_path)


def parse_grid_cards(content: str) -> list[CardInfo]:
    """Parse grid cards from MyST markdown content."""
    cards = []
    lines = content.split("\n")

    i = 0
    while i < len(lines):
        line = lines[i]

        # Look for grid-item-card start
        if ":::{grid-item-card}" in line:
            card_start = i
            title_match = re.search(r":::\{grid-item-card\}\s*(.*)", line)
            title = title_match.group(1).strip() if title_match else ""

            link = ""
            link_type = "doc"
            description_lines = []

            i += 1
            # Parse card attributes and content
            while i < len(lines) and not lines[i].strip().startswith(":::"):
                current = lines[i]

                if current.strip().startswith(":link:"):
                    link = current.split(":link:")[1].strip()
                elif current.strip().startswith(":link-type:"):
                    link_type = current.split(":link-type:")[1].strip()
                elif current.strip() and not current.strip().startswith(":"):
                    description_lines.append(current.strip())

                i += 1

            card_end = i
            description = " ".join(description_lines)

            # Reconstruct original text
            original = "\n".join(lines[card_start : card_end + 1])

            cards.append(
                CardInfo(
                    title=title,
                    link=link,
                    link_type=link_type,
                    description=description,
                    start_line=card_start,
                    end_line=card_end,
                    original_text=original,
                )
            )

        i += 1

    return cards


def resolve_link_path(link: str, index_file: Path) -> Path | None:
    """Resolve a doc link to a file path."""
    if not link:
        return None

    # Get the directory containing the index file
    base_dir = index_file.parent

    # Handle relative paths
    if link.startswith("../"):
        link_path = link
    else:
        link_path = link

    # Try different file extensions
    for ext in [".md", ".rst", "/index.md", "/index.rst", ""]:
        candidate = base_dir / f"{link_path}{ext}"
        if candidate.exists():
            return candidate

    # Try without extension changes
    candidate = base_dir / link_path
    if candidate.exists():
        return candidate

    return None


def generate_card_text(card: CardInfo, page_info: PageInfo) -> str:
    """Generate updated card text from page info."""
    lines = [f":::{'{'}grid-item-card{'}'} {page_info.title}"]
    lines.append(f":link: {card.link}")
    lines.append(f":link-type: {card.link_type}")
    lines.append("")
    lines.append(page_info.description)
    lines.append(":::")

    return "\n".join(lines)


def update_index_file(
    index_path: Path,
    dry_run: bool = False,
    verbose: bool = False,
) -> tuple[int, list[str]]:
    """
    Update grid cards in an index file.

    Returns:
        Tuple of (number of cards updated, list of change descriptions)
    """
    content = index_path.read_text(encoding="utf-8")
    cards = parse_grid_cards(content)

    if not cards:
        if verbose:
            print(f"  No grid cards found in {index_path}")
        return 0, []

    changes = []
    lines = content.split("\n")
    updates_made = 0

    # Process cards in reverse order to maintain line numbers
    for card in reversed(cards):
        resolved_path = resolve_link_path(card.link, index_path)

        if not resolved_path:
            if verbose:
                print(f"  Warning: Could not resolve link '{card.link}'")
            continue

        page_info = extract_page_info(resolved_path)

        if not page_info:
            if verbose:
                print(f"  Warning: Could not extract info from '{resolved_path}'")
            continue

        # Check if update is needed
        new_card_text = generate_card_text(card, page_info)

        if card.original_text.strip() != new_card_text.strip():
            changes.append(
                f"  - '{card.title}' → '{page_info.title}' (from {resolved_path.name})"
            )

            # Replace the card in content
            new_lines = new_card_text.split("\n")
            lines = lines[: card.start_line] + new_lines + lines[card.end_line + 1 :]
            updates_made += 1

    if updates_made > 0 and not dry_run:
        new_content = "\n".join(lines)
        index_path.write_text(new_content, encoding="utf-8")

    return updates_made, changes


def find_index_files(docs_dir: Path) -> list[Path]:
    """Find all index.md files that might contain grid cards."""
    index_files = []

    for md_file in docs_dir.rglob("index.md"):
        content = md_file.read_text(encoding="utf-8")
        if "grid-item-card" in content:
            index_files.append(md_file)

    return sorted(index_files)


def main():
    parser = argparse.ArgumentParser(
        description="Update grid cards in index files based on linked page content.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Specific index files to update (default: all index.md files with grid cards)",
    )
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Show what would be changed without making changes",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Documentation root directory (default: ../)",
    )

    args = parser.parse_args()

    if args.files:
        index_files = [Path(f) for f in args.files]
    else:
        index_files = find_index_files(args.docs_dir)

    if not index_files:
        print("No index files with grid cards found.")
        return 0

    total_updates = 0
    all_changes = []

    print(
        f"{'[DRY RUN] ' if args.dry_run else ''}Checking {len(index_files)} index file(s)...\n"
    )

    for index_file in index_files:
        if args.verbose:
            print(f"Processing: {index_file}")

        updates, changes = update_index_file(
            index_file,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )

        if changes:
            print(f"{'Would update' if args.dry_run else 'Updated'} {index_file}:")
            for change in changes:
                print(change)
            print()

        total_updates += updates
        all_changes.extend(changes)

    if total_updates > 0:
        action = "would be updated" if args.dry_run else "updated"
        print(f"\n✅ {total_updates} card(s) {action}.")
    else:
        print("\n✅ All cards are up to date.")

    return 0 if not args.dry_run or total_updates == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
