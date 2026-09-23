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

from __future__ import annotations

import argparse
import html
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = REPO_ROOT / "docs" / "_static" / "images" / "community-telemetry" / "0.22.0-0.23.0.json"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA.parent

BLACK = "#000000"
LIGHT_GRAY = "#CDCDCD"
MEDIUM_GRAY = "#8C8C8C"
DARK_GRAY = "#5E5E5E"
NVIDIA_GREEN = "#76B900"
EMERALD = "#008564"
AMETHYST = "#5D1682"
CPU_BLUE = "#0071C5"
GARNET = "#890C58"
FLUORITE = "#FAC200"
WHITE = "#FFFFFF"


class Svg:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            f'<rect width="{width}" height="{height}" fill="{WHITE}"/>',
        ]

    def rect(self, x: float, y: float, width: float, height: float, fill: str, rx: float = 0) -> None:
        self.parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" rx="{rx:.2f}" fill="{fill}"/>'
        )

    def line(self, x1: float, y1: float, x2: float, y2: float, stroke: str, width: float = 1) -> None:
        self.parts.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" stroke-width="{width:.2f}"/>'
        )

    def text(
        self,
        x: float,
        y: float,
        value: str,
        size: int,
        fill: str = BLACK,
        weight: int = 400,
        anchor: str = "start",
    ) -> None:
        escaped = html.escape(value)
        self.parts.append(
            f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{escaped}</text>'
        )

    def finish(self) -> str:
        return "\n".join([*self.parts, "</svg>", ""])


def _load_data(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    snapshot = payload["snapshot"]
    total_sessions = snapshot["estimated_unique_startup_sessions"]
    if total_sessions <= 0:
        raise ValueError("estimated_unique_startup_sessions must be positive")
    configured = payload["configured_rails"]
    bucket_total = sum(item["records"] for item in configured["buckets"])
    if bucket_total != configured["eligible_startup_records"]:
        raise ValueError("configured rail buckets do not match eligible_startup_records")
    if configured["excluded_zero_records"] + bucket_total != snapshot["startup_records"]:
        raise ValueError("configured rail records do not match startup_records")
    return payload


def _snapshot_label(data: dict[str, Any]) -> str:
    snapshot = data["snapshot"]
    versions = " and ".join(f"v{version}" for version in snapshot["versions"])
    start = datetime.fromisoformat(snapshot["start_utc"].replace("Z", "+00:00")).astimezone(timezone.utc)
    end = datetime.fromisoformat(snapshot["end_utc"].replace("Z", "+00:00")).astimezone(timezone.utc)
    start_label = f"{start:%b} {start.day}"
    end_label = f"{end:%b} {end.day}"
    if start.year == end.year:
        date_range = f"{start_label}–{end_label}, {end.year}"
    else:
        date_range = f"{start_label}, {start.year}–{end_label}, {end.year}"
    return f"{versions} · {date_range}"


def _format_count(value: int) -> str:
    return f"{value:,}"


def _header(svg: Svg, title: str, subtitle: str) -> None:
    svg.rect(70, 58, 14, 112, NVIDIA_GREEN)
    svg.text(110, 78, "NVIDIA NeMo Guardrails", 22, NVIDIA_GREEN, 700)
    svg.text(110, 132, title, 46, BLACK, 700)
    svg.text(110, 174, subtitle, 23, DARK_GRAY)
    svg.line(70, 205, svg.width - 70, 205, LIGHT_GRAY, 2)


def _horizontal_bars(
    values: list[dict[str, Any]],
    denominator: int,
    title: str,
    subtitle: str,
    note_lines: list[str],
    output: Path,
    height: int,
    axis_max: float,
    row_step: int,
    colors: list[str],
    count_key: str,
) -> None:
    width = 1600
    svg = Svg(width, height)
    _header(svg, title, subtitle)
    chart_x = 430
    chart_width = 1010
    chart_top = 270
    chart_bottom = chart_top + row_step * len(values) - 22
    tick_step = 25 if axis_max >= 100 else 10
    tick = 0
    while tick <= axis_max:
        x = chart_x + chart_width * tick / axis_max
        svg.line(x, 238, x, chart_bottom + 18, LIGHT_GRAY, 1)
        svg.text(x, 232, f"{tick:g}%", 18, MEDIUM_GRAY, anchor="middle")
        tick += tick_step

    for index, item in enumerate(values):
        y = chart_top + index * row_step
        count = int(item[count_key])
        percent = count / denominator * 100
        bar_width = max(3.0, chart_width * percent / axis_max)
        color = colors[index % len(colors)]
        svg.text(chart_x - 28, y + 34, item["label"], 25, BLACK, 600, "end")
        svg.rect(chart_x, y, bar_width, 50, color, 5)
        value_text = f"{percent:.1f}%  ·  {_format_count(count)}"
        if bar_width >= 285:
            svg.text(chart_x + bar_width - 18, y + 34, value_text, 22, WHITE, 700, "end")
        else:
            svg.text(chart_x + bar_width + 18, y + 34, value_text, 22, DARK_GRAY, 700)

    note_y = height - 78 - 30 * (len(note_lines) - 1)
    for line in note_lines:
        svg.text(70, note_y, line, 19, DARK_GRAY)
        note_y += 30
    output.write_text(svg.finish())


def _render_rail_types(data: dict[str, Any], output_dir: Path) -> Path:
    sessions = data["snapshot"]["estimated_unique_startup_sessions"]
    output = output_dir / "sessions-by-rail-type.svg"
    _horizontal_bars(
        data["rail_types"],
        sessions,
        "Which rail types are most common?",
        f"Share of estimated unique startup sessions · {_snapshot_label(data)}",
        [
            f"Based on {_format_count(sessions)} estimated unique process sessions.",
            "A session can use multiple rail types, so percentages do not sum to 100%.",
        ],
        output,
        940,
        100,
        98,
        [NVIDIA_GREEN, EMERALD, CPU_BLUE, AMETHYST, GARNET, FLUORITE],
        "sessions",
    )
    return output


def _render_builtin_features(data: dict[str, Any], output_dir: Path) -> Path:
    sessions = data["snapshot"]["estimated_unique_startup_sessions"]
    feature_data = data["builtin_features"]
    output = output_dir / "sessions-by-built-in-feature.svg"
    _horizontal_bars(
        feature_data["values"],
        sessions,
        "Which built-in features are most common?",
        f"Top {feature_data['top_n']} by share of estimated unique startup sessions · {_snapshot_label(data)}",
        [
            f"Based on {_format_count(sessions)} estimated unique process sessions.",
            "A session can use multiple built-in features, so percentages do not sum to 100%.",
        ],
        output,
        1100,
        70,
        72,
        [NVIDIA_GREEN, EMERALD, CPU_BLUE, AMETHYST, GARNET],
        "sessions",
    )
    return output


def _render_configured_rails(data: dict[str, Any], output_dir: Path) -> Path:
    configured = data["configured_rails"]
    eligible = configured["eligible_startup_records"]
    output = output_dir / "configured-rails-at-startup.svg"
    _horizontal_bars(
        configured["buckets"],
        eligible,
        "How many rails are configured at startup?",
        f"Share of startup configurations with at least one counted rail · {_snapshot_label(data)}",
        [
            f"Based on {_format_count(eligible)} startup records after excluding zero configured rails.",
            "Dialog rails are represented separately and are not included in the configured-rail count.",
        ],
        output,
        720,
        50,
        112,
        [NVIDIA_GREEN, EMERALD, CPU_BLUE],
        "records",
    )
    return output


def _render_png(svg_path: Path, renderer: str) -> Path:
    png_path = svg_path.with_suffix(".png")
    subprocess.run([renderer, "--output", str(png_path), str(svg_path)], check=True)
    return png_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--svg-only", action="store_true")
    args = parser.parse_args()

    renderer = None if args.svg_only else shutil.which("rsvg-convert")
    if not args.svg_only and renderer is None:
        parser.error("rsvg-convert is required for PNG output; install librsvg or use --svg-only")

    data = _load_data(args.data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    svg_paths = [
        _render_rail_types(data, args.output_dir),
        _render_builtin_features(data, args.output_dir),
        _render_configured_rails(data, args.output_dir),
    ]

    for svg_path in svg_paths:
        print(svg_path)
        if renderer is not None:
            png_path = _render_png(svg_path, renderer)
            print(png_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
