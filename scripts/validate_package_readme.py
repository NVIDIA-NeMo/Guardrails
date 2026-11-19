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

# /// script
# requires-python = ">=3.11"
# dependencies = ["readme-renderer[md]==46.0", "twine==7.0.0"]
# ///
"""Validate the Markdown descriptions we publish, without fetching remote URLs."""

import argparse
import email
import html
import subprocess
import sys
import tarfile
import warnings
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

from readme_renderer.markdown import render, variants


class Destinations(HTMLParser):
    def __init__(self, content):
        super().__init__()
        self.ids = set()
        self.urls = []
        self.feed(content)

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key == "id" and value:
                self.ids.add(value)
            if key in ("href", "src"):
                self.urls.append((key, value or ""))


def url_errors(document, *, anchors):
    errors = []
    for attribute, value in document.urls:
        if attribute == "href" and value.startswith("#"):
            if anchors and unquote(value[1:]) not in document.ids:
                errors.append(f"Missing anchor: {value}")
            continue
        try:
            url = urlsplit(value)
            valid = url.scheme in ("https", "http") and bool(url.hostname)
            valid |= attribute == "href" and url.scheme == "mailto" and bool(url.path)
        except ValueError:
            valid = False
        if not valid:
            errors.append(f"Non-portable {attribute}: {value!r}")
    return errors


def description(path):
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if n.count("/") == 1 and n.endswith(".dist-info/METADATA")]
            if len(names) != 1:
                raise ValueError("Expected exactly one wheel METADATA")
            data = archive.read(names[0])
    else:
        with tarfile.open(path, "r:gz") as archive:
            members = [m for m in archive.getmembers() if m.name.count("/") == 1 and m.name.endswith("/PKG-INFO")]
            if len(members) != 1 or not members[0].isfile():
                raise ValueError("Expected exactly one top-level sdist PKG-INFO file")
            data = archive.extractfile(members[0]).read()
    metadata = email.message_from_string(data.decode("utf-8"))
    if metadata.get("Description-Content-Type", "").split(";", 1)[0].strip() != "text/markdown":
        raise ValueError("Expected Description-Content-Type: text/markdown")
    text = metadata.get_payload()
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Missing package description")
    return text


def preview(text, destination):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # Inspect the pinned renderer's GFM output before sanitization can drop URLs.
        raw_html = variants["GFM"](text)
        rendered = render(text)
    if not rendered:
        raise ValueError("Markdown rendering failed")
    destination.write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
        "img-src https: http:; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'\">"
        f"<title>{html.escape(destination.stem)}</title>"
        "<style>body{max-width:960px;margin:2rem auto;padding:0 1rem;font:16px/1.5 sans-serif}"
        "img{max-width:100%}pre{overflow:auto;padding:1rem;background:#f4f4f4}"
        "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:.4rem}</style>"
        "<body><p>Local package-description preview; PyPI styling may differ.</p>" + rendered + "</body></html>",
        encoding="utf-8",
    )
    return url_errors(Destinations(raw_html), anchors=False) + url_errors(Destinations(rendered), anchors=True)


def validate(distributions, output):
    output.mkdir(parents=True, exist_ok=True)
    wheels = sorted(distributions.glob("*.whl"))
    sdists = sorted(distributions.glob("*.tar.gz"))
    errors = []
    if not wheels or not sdists:
        errors.append("Both a wheel and source distribution are required")
    descriptions = []
    paths = wheels + sdists
    for path in paths:
        try:
            text = description(path)
            descriptions.append(text)
            errors.extend(f"{path.name}: {e}" for e in preview(text, output / f"{path.name}.html"))
        except (ValueError, OSError, tarfile.TarError, zipfile.BadZipFile, Warning) as exc:
            errors.append(f"{path.name}: {exc}")
    if descriptions and any(text != descriptions[0] for text in descriptions[1:]):
        errors.append("Descriptions differ between built artifacts")
    if paths:
        result = subprocess.run([sys.executable, "-m", "twine", "check", "--strict", *map(str, paths)], check=False)
        if result.returncode:
            errors.append("twine check --strict failed")
    return list(dict.fromkeys(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    parser.add_argument("--output-dir", type=Path, default=Path("readme-preview"))
    args = parser.parse_args()
    errors = validate(args.dist_dir, args.output_dir)
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        return 1
    print(f"Package descriptions validated. HTML previews: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
