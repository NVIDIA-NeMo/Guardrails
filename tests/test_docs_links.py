# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECK_DOCS_LINKS = REPO_ROOT / "scripts" / "check-docs-links.sh"


def run_link_check(file_path: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        ["bash", str(CHECK_DOCS_LINKS), "--local-only", str(file_path)],
        cwd=REPO_ROOT,
        env=merged_env,
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )


def test_reports_broken_local_markdown_links_with_source_line_numbers(
    tmp_path: Path,
) -> None:
    md_path = tmp_path / "guide.md"
    (tmp_path / "exists.md").write_text("# ok\n", encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# Guide",
                "",
                "[working](./exists.md)",
                "[broken](./missing.md)",
                "```md",
                "[ignored](./inside-code-fence.md)",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = run_link_check(md_path)
    output = f"{result.stdout}{result.stderr}"

    assert result.returncode == 1
    assert f"broken local link in {md_path}:4 -> ./missing.md" in output
    assert "inside-code-fence.md" not in output


def test_ignores_links_inside_inline_code_and_html_comments(tmp_path: Path) -> None:
    md_path = tmp_path / "guide.md"
    md_path.write_text(
        "\n".join(
            [
                "# Guide",
                "",
                "Use `refer to [DOC PAGE](/doc/path)` as placeholder text.",
                "<!-- [ignored](./inside-comment.md) -->",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = run_link_check(md_path)
    output = f"{result.stdout}{result.stderr}"

    assert result.returncode == 0
    assert "/doc/path" not in output
    assert "inside-comment.md" not in output


def test_resolves_guardrails_fern_routes(tmp_path: Path) -> None:
    md_path = tmp_path / "guide.mdx"
    md_path.write_text(
        "\n".join(
            [
                "# Guide",
                "",
                "[Install](/get-started/installation-guide)",
                '<Card title="Configure" href="/configure-guardrails/configure-rails">',
                "[SDK](/guardrails-python-sdk/nemoguardrails)",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = run_link_check(md_path)
    output = f"{result.stdout}{result.stderr}"

    assert result.returncode == 0, output


def test_rejects_mdx_suffixes_for_links_that_resolve_as_fern_routes() -> None:
    temp_dir = Path(tempfile.mkdtemp(prefix="check-docs-route-suffix-", dir=REPO_ROOT / "docs"))
    try:
        temp_path = temp_dir / "temp.mdx"
        target_path = temp_dir / "target.mdx"
        nav_path = temp_dir / "index.yml"
        temp_nav_path = temp_path.relative_to(REPO_ROOT / "docs")
        target_nav_path = target_path.relative_to(REPO_ROOT / "docs")
        temp_path.write_text(
            "\n".join(
                [
                    "---",
                    'title: "Temporary Link Check Page"',
                    "---",
                    "",
                    "[Wrong](target.mdx)",
                    "[Right](target)",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        target_path.write_text("# Target\n", encoding="utf-8")
        nav_path.write_text(
            "\n".join(
                [
                    "navigation:",
                    '  - page: "Temp"',
                    f"    path: {temp_nav_path}",
                    "    slug: temp",
                    '  - page: "Target"',
                    f"    path: {target_nav_path}",
                    "    slug: target",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        result = run_link_check(temp_path, {"CHECK_DOCS_FERN_NAV_YML": str(nav_path)})
        output = f"{result.stdout}{result.stderr}"

        assert result.returncode == 1
        assert (f"route-style link should omit .md/.mdx extension in {temp_path}:5 -> target.mdx") in output
        assert f"broken local link in {temp_path}:6 -> target" not in output
    finally:
        shutil.rmtree(temp_dir)


def test_fails_loudly_on_malformed_html_comments(tmp_path: Path) -> None:
    md_path = tmp_path / "guide.md"
    md_path.write_text(
        "\n".join(["# Guide", "<!-- missing close", "[ignored](./inside-comment.md)", ""]),
        encoding="utf-8",
    )

    result = run_link_check(md_path)
    output = f"{result.stdout}{result.stderr}"

    assert result.returncode == 1
    assert f"malformed HTML comment in {md_path}" in output
    assert "inside-comment.md" not in output
