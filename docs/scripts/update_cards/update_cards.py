#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Automatically update grid cards in index files based on linked page content.

This script scans index files for Sphinx-Design grid cards, reads the linked
pages to extract their title and description, and updates the cards accordingly.

Usage:
    python update_cards.py [--dry-run] [--verbose] [path/to/index.md ...]

Commands:
    update (default)  Update existing grid cards from linked pages
    watch             Watch for file changes and auto-update cards
    generate          Generate grid cards from directory structure

Examples:
    # Update all index files in docs/
    python update_cards.py

    # Update specific index file
    python update_cards.py update ../configuration-guide/yaml-schema/index.md

    # Watch mode: auto-update when files change
    python update_cards.py watch

    # Generate cards for a directory
    python update_cards.py generate ./getting-started/

    # Preview generated cards without writing
    python update_cards.py generate ./getting-started/ --dry-run
"""

import argparse
import re
import sys
import time
from pathlib import Path
from typing import NamedTuple

# Optional dependency for watch mode
try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None  # type: ignore[assignment, misc]
    FileSystemEventHandler = object  # type: ignore[assignment, misc]


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


def parse_frontmatter(content: str) -> tuple[dict[str, str], int]:
    """
    Parse YAML frontmatter from markdown content.

    Returns:
        Tuple of (frontmatter dict, line index after frontmatter)
    """
    lines = content.split("\n")
    frontmatter: dict[str, str] = {}
    start_idx = 0

    if lines and lines[0].strip() == "---":
        frontmatter_lines = []
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == "---":
                start_idx = i + 1
                break
            frontmatter_lines.append(line)

        # Simple YAML parsing for key: value pairs
        for line in frontmatter_lines:
            if ":" in line and not line.strip().startswith("#"):
                key, _, value = line.partition(":")
                key = key.strip()
                value = value.strip()
                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                if key and value:
                    frontmatter[key] = value

    return frontmatter, start_idx


def extract_page_info(file_path: Path) -> PageInfo | None:
    """Extract title and description from a markdown/rst file.

    Priority for description:
    1. Frontmatter 'description' field (if present)
    2. First non-empty paragraph after the title
    3. Default: "Documentation for {title}."
    """
    if not file_path.exists():
        return None

    content = file_path.read_text(encoding="utf-8")
    lines = content.split("\n")

    title = None
    description = None

    # Parse frontmatter
    frontmatter, start_idx = parse_frontmatter(content)

    # Check for description in frontmatter
    frontmatter_description = frontmatter.get("description")
    frontmatter_title = frontmatter.get("title")

    # Prioritize frontmatter title over H1 heading
    if frontmatter_title:
        title = frontmatter_title
        # Still need to find where content starts (after H1)
        for i, line in enumerate(lines[start_idx:], start_idx):
            stripped = line.strip()
            if not stripped or stripped.startswith("<!--"):
                continue
            if stripped.startswith("# ") and not stripped.startswith("##"):
                start_idx = i + 1
                break
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and all(c == "=" for c in next_line) and len(next_line) >= len(stripped):
                    start_idx = i + 2
                    break
    else:
        # Extract title from first H1
        for i, line in enumerate(lines[start_idx:], start_idx):
            stripped = line.strip()

            # Skip empty lines and comments
            if not stripped or stripped.startswith("<!--"):
                continue

            # Markdown H1: # Title
            if stripped.startswith("# ") and not stripped.startswith("##"):
                title = stripped[2:].strip()
                start_idx = i + 1
                break

            # RST H1: Title followed by === underline
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and all(c == "=" for c in next_line) and len(next_line) >= len(stripped):
                    title = stripped
                    start_idx = i + 2
                    break

    if not title:
        return None

    # Use frontmatter description if available
    if frontmatter_description:
        description = frontmatter_description
    else:
        # Extract description (first non-empty paragraph after title)
        description_lines: list[str] = []
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

            # Skip HTML comments
            if stripped.startswith("<!--"):
                continue

            # Collect paragraph lines
            if stripped:
                # Skip if it looks like a table or list
                if stripped.startswith("|") or stripped.startswith("-") or stripped.startswith("*"):
                    if not description_lines:
                        continue
                    break
                description_lines.append(stripped)
            elif description_lines:
                # End of paragraph
                break

        if description_lines:
            description = " ".join(description_lines)
        else:
            description = f"Documentation for {title}."

    # Truncate if too long (aim for ~200 chars)
    if len(description) > 200:
        description = description[:197].rsplit(" ", 1)[0] + "..."

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
            changes.append(f"  - '{card.title}' → '{page_info.title}' (from {resolved_path.name})")

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


# =============================================================================
# Watch Mode
# =============================================================================


class CardUpdateHandler(FileSystemEventHandler):
    """File system event handler for auto-updating cards."""

    def __init__(self, docs_dir: Path, verbose: bool = False, debounce_seconds: float = 1.0):
        self.docs_dir = docs_dir
        self.verbose = verbose
        self.debounce_seconds = debounce_seconds
        self._last_update: dict[str, float] = {}
        self._index_files: set[Path] = set()
        self._refresh_index_files()

    def _refresh_index_files(self):
        """Refresh the list of index files with grid cards."""
        self._index_files = set(find_index_files(self.docs_dir))

    def _should_process(self, path: str) -> bool:
        """Check if we should process this file change (debouncing)."""
        now = time.time()
        last = self._last_update.get(path, 0)
        if now - last < self.debounce_seconds:
            return False
        self._last_update[path] = now
        return True

    def _find_affected_index_files(self, changed_file: Path) -> list[Path]:
        """Find index files that might be affected by a file change."""
        affected = []
        changed_dir = changed_file.parent

        # If the changed file itself is an index file with cards, include it
        if changed_file in self._index_files:
            affected.append(changed_file)

        for index_file in self._index_files:
            # Check if the changed file is in the same directory or a subdirectory
            try:
                changed_file.relative_to(index_file.parent)
                affected.append(index_file)
            except ValueError:
                pass

            # Also check parent directories
            if index_file.parent in changed_dir.parents:
                affected.append(index_file)

        return list(set(affected))

    def on_modified(self, event):
        if event.is_directory:
            return

        path = Path(event.src_path)
        if path.suffix not in {".md", ".rst"}:
            return

        if not self._should_process(event.src_path):
            return

        self._handle_file_change(path)

    def on_created(self, event):
        if event.is_directory:
            return

        path = Path(event.src_path)
        if path.suffix not in {".md", ".rst"}:
            return

        if not self._should_process(event.src_path):
            return

        # Refresh index files in case a new index.md was created
        if path.name == "index.md":
            self._refresh_index_files()

        self._handle_file_change(path)

    def _handle_file_change(self, changed_file: Path):
        """Handle a file change event."""
        if self.verbose:
            print(f"\n📝 File changed: {changed_file}")

        affected_indexes = self._find_affected_index_files(changed_file)

        if not affected_indexes:
            if self.verbose:
                print("   No affected index files found.")
            return

        if self.verbose:
            print(f"   Found {len(affected_indexes)} affected index file(s)")

        for index_file in affected_indexes:
            if self.verbose:
                print(f"   Checking: {index_file}")

            _updates, changes = update_index_file(
                index_file,
                dry_run=False,
                verbose=self.verbose,
            )

            if changes:
                print(f"✅ Updated {index_file}:")
                for change in changes:
                    print(f"   {change}")
            elif self.verbose:
                print(f"   No card updates needed for {index_file.name}")


def run_watch_mode(docs_dir: Path, verbose: bool = False):
    """Run the script in watch mode, auto-updating cards on file changes."""
    if not WATCHDOG_AVAILABLE:
        print("❌ Watch mode requires the 'watchdog' package.")
        print("   Install it with: pip install watchdog")
        print("   Or: poetry add watchdog --group docs")
        return 1

    print(f"👀 Watching for changes in: {docs_dir}")
    print("   Press Ctrl+C to stop.\n")

    # Initial update
    index_files = find_index_files(docs_dir)
    print(f"Found {len(index_files)} index file(s) with grid cards.")

    for index_file in index_files:
        _updates, changes = update_index_file(index_file, dry_run=False, verbose=verbose)
        if changes:
            print(f"Updated {index_file}:")
            for change in changes:
                print(change)

    print("\n🔄 Watching for changes...\n")

    event_handler = CardUpdateHandler(docs_dir, verbose=verbose)
    observer = Observer()
    observer.schedule(event_handler, str(docs_dir), recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n👋 Stopping watch mode...")
        observer.stop()

    observer.join()
    return 0


# =============================================================================
# Generate Cards
# =============================================================================


def find_documentable_files(directory: Path, exclude_patterns: list[str] | None = None) -> list[Path]:
    """Find markdown/rst files in a directory that could have cards generated."""
    exclude_patterns = exclude_patterns or ["index.md", "index.rst", "README.md"]
    files = []

    for pattern in ["*.md", "*.rst"]:
        for file_path in directory.glob(pattern):
            if file_path.name not in exclude_patterns:
                files.append(file_path)

    # Also check for subdirectories with index files
    for subdir in directory.iterdir():
        if subdir.is_dir() and not subdir.name.startswith("."):
            index_path = subdir / "index.md"
            if not index_path.exists():
                index_path = subdir / "index.rst"
            if index_path.exists():
                files.append(index_path)

    return sorted(files)


def generate_grid_cards(
    directory: Path,
    verbose: bool = False,
    columns: str = "1 1 2 2",
    gutter: int = 3,
) -> tuple[str, int]:
    """
    Generate grid cards markup for files in a directory.

    Returns:
        Tuple of (generated markup, number of cards)
    """
    files = find_documentable_files(directory)

    if not files:
        if verbose:
            print(f"  No documentable files found in {directory}")
        return "", 0

    cards = []
    for file_path in files:
        page_info = extract_page_info(file_path)

        if not page_info:
            if verbose:
                print(f"  Warning: Could not extract info from '{file_path}'")
            continue

        # Determine the link path relative to the directory
        if file_path.parent == directory:
            # File is directly in the directory
            link = file_path.stem
        else:
            # File is an index in a subdirectory
            link = file_path.parent.name + "/index"

        card_lines = [
            f":::{{grid-item-card}} {page_info.title}",
            f":link: {link}",
            ":link-type: doc",
            "",
            page_info.description,
            ":::",
        ]
        cards.append("\n".join(card_lines))

    if not cards:
        return "", 0

    # Build the full grid markup
    markup_lines = [
        f"::::{{grid}} {columns}",
        f":gutter: {gutter}",
        "",
    ]

    for card in cards:
        markup_lines.append(card)
        markup_lines.append("")

    markup_lines.append("::::")

    return "\n".join(markup_lines), len(cards)


def run_generate_cards(
    directory: Path,
    output_file: Path | None = None,
    dry_run: bool = False,
    verbose: bool = False,
    insert_after: str | None = None,
) -> int:
    """
    Generate grid cards for a directory.

    Args:
        directory: Directory to scan for files
        output_file: Output file (default: directory/index.md)
        dry_run: Preview without writing
        verbose: Show detailed output
        insert_after: Pattern to insert cards after (for existing files)
    """
    if not directory.is_dir():
        print(f"❌ Not a directory: {directory}")
        return 1

    markup, card_count = generate_grid_cards(directory, verbose=verbose)

    if card_count == 0:
        print(f"No documentable files found in {directory}")
        return 0

    output_file = output_file or (directory / "index.md")

    print(f"{'[DRY RUN] ' if dry_run else ''}Generated {card_count} card(s) for {directory}\n")

    if verbose or dry_run:
        print("Generated markup:")
        print("-" * 40)
        print(markup)
        print("-" * 40)
        print()

    if output_file.exists() and insert_after:
        # Insert into existing file
        content = output_file.read_text(encoding="utf-8")

        if insert_after in content:
            # Find the position to insert
            insert_pos = content.find(insert_after) + len(insert_after)
            # Find the end of the line
            newline_pos = content.find("\n", insert_pos)
            if newline_pos == -1:
                newline_pos = len(content)

            new_content = content[:newline_pos] + "\n\n" + markup + "\n" + content[newline_pos:]

            if not dry_run:
                output_file.write_text(new_content, encoding="utf-8")
                print(f"✅ Inserted cards into {output_file}")
            else:
                print(f"Would insert cards into {output_file}")
        else:
            print(f"⚠️  Pattern '{insert_after}' not found in {output_file}")
            print("   Cards were not inserted. Use --verbose to see the generated markup.")
            return 1
    elif output_file.exists():
        # Check if file already has grid cards
        content = output_file.read_text(encoding="utf-8")
        if "::::{{grid}}" in content or "::::{grid}" in content:
            print(f"⚠️  {output_file} already contains grid cards.")
            print("   Use --insert-after to specify where to add new cards,")
            print("   or manually copy the generated markup above.")
            return 0

        # Append to existing file
        if not dry_run:
            with output_file.open("a", encoding="utf-8") as f:
                f.write("\n\n" + markup + "\n")
            print(f"✅ Appended cards to {output_file}")
        else:
            print(f"Would append cards to {output_file}")
    else:
        # Create new file with basic structure
        new_content = f"""# {directory.name.replace("-", " ").title()}

{markup}
"""
        if not dry_run:
            output_file.write_text(new_content, encoding="utf-8")
            print(f"✅ Created {output_file} with cards")
        else:
            print(f"Would create {output_file}")

    return 0


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Update grid cards in index files based on linked page content.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Common arguments
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path(__file__).parent.parent.parent,  # scripts/update_cards/ → scripts/ → docs/
        help="Documentation root directory (default: docs/)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Update command (default behavior)
    update_parser = subparsers.add_parser(
        "update",
        help="Update existing grid cards from linked pages (default)",
    )
    update_parser.add_argument(
        "files",
        nargs="*",
        help="Specific index files to update (default: all index.md files with grid cards)",
    )
    update_parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Show what would be changed without making changes",
    )

    # Watch command (no additional arguments needed)
    subparsers.add_parser(
        "watch",
        help="Watch for file changes and auto-update cards",
    )

    # Generate command
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate grid cards from directory structure",
    )
    generate_parser.add_argument(
        "directory",
        type=Path,
        help="Directory to scan for documentable files",
    )
    generate_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file (default: directory/index.md)",
    )
    generate_parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Preview without writing",
    )
    generate_parser.add_argument(
        "--insert-after",
        type=str,
        help="Pattern to insert cards after (for existing files)",
    )
    generate_parser.add_argument(
        "--columns",
        type=str,
        default="1 1 2 2",
        help="Grid columns specification (default: '1 1 2 2')",
    )
    generate_parser.add_argument(
        "--gutter",
        type=int,
        default=3,
        help="Grid gutter size (default: 3)",
    )

    args = parser.parse_args()

    # Default to 'update' command if no command specified
    # Handle both old-style (no subcommand) and new-style (with subcommand) invocations
    if args.command is None:
        # Check if there are positional arguments that look like files
        remaining = sys.argv[1:]
        # Filter out known flags
        files = [arg for arg in remaining if not arg.startswith("-") and not arg.startswith("--")]
        args.command = "update"
        args.files = files if files else []
        args.dry_run = "--dry-run" in remaining or "-n" in remaining

    # Route to appropriate command handler
    if args.command == "watch":
        return run_watch_mode(args.docs_dir, verbose=args.verbose)

    elif args.command == "generate":
        return run_generate_cards(
            directory=args.directory,
            output_file=args.output,
            dry_run=args.dry_run,
            verbose=args.verbose,
            insert_after=args.insert_after,
        )

    else:  # update (default)
        return run_update_command(args)


def run_update_command(args) -> int:
    """Run the update command."""
    if hasattr(args, "files") and args.files:
        index_files = [Path(f) for f in args.files]
    else:
        index_files = find_index_files(args.docs_dir)

    if not index_files:
        print("No index files with grid cards found.")
        return 0

    total_updates = 0
    all_changes = []

    dry_run = getattr(args, "dry_run", False)
    verbose = getattr(args, "verbose", False)

    print(f"{'[DRY RUN] ' if dry_run else ''}Checking {len(index_files)} index file(s)...\n")

    for index_file in index_files:
        if verbose:
            print(f"Processing: {index_file}")

        updates, changes = update_index_file(
            index_file,
            dry_run=dry_run,
            verbose=verbose,
        )

        if changes:
            print(f"{'Would update' if dry_run else 'Updated'} {index_file}:")
            for change in changes:
                print(change)
            print()

        total_updates += updates
        all_changes.extend(changes)

    if total_updates > 0:
        action = "would be updated" if dry_run else "updated"
        print(f"\n✅ {total_updates} card(s) {action}.")
    else:
        print("\n✅ All cards are up to date.")

    return 0 if not dry_run or total_updates == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
