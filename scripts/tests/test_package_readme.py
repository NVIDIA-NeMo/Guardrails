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

"""Offline regression cases for the packaged README gate."""

import importlib.util
import io
import subprocess
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("validator", Path(__file__).parents[1] / "validate_package_readme.py")
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class PackageReadmeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def artifacts(self, wheel_text, sdist_text=None, content_type="text/markdown"):
        def metadata(text):
            return f"Metadata-Version: 2.4\nName: example\nVersion: 1.0\nDescription-Content-Type: {content_type}\n\n{text}".encode()

        with zipfile.ZipFile(self.root / "example.whl", "w") as archive:
            archive.writestr("example.dist-info/METADATA", metadata(wheel_text))
        with tarfile.open(self.root / "example.tar.gz", "w:gz") as archive:
            data = metadata(wheel_text if sdist_text is None else sdist_text)
            member = tarfile.TarInfo("example/PKG-INFO")
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))

    def validate(self):
        with patch.object(validator.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)):
            return validator.validate(self.root, self.root / "preview")

    def test_unicode_and_valid_links(self):
        self.artifacts(
            "# Café\n\n[Jump](#caf%C3%A9) [Docs](https://example.org) [Mail](mailto:a@example.org)\n\n![Image](https://example.org/image.png)"
        )
        self.assertEqual(self.validate(), [])
        self.assertEqual(len(list((self.root / "preview").glob("*.html"))), 2)

    def test_nonportable_destinations(self):
        for text in [
            "[docs](./docs/a.md)",
            "![image](images/a.png)",
            "[docs][ref]\n\n[ref]: /docs",
            '<img src="./x.png">',
            '<a href="//example.org">link</a>',
            "![image](mailto:a@example.org)",
            "[bad](https:///path)",
        ]:
            with self.subTest(text=text):
                self.artifacts(text)
                self.assertTrue(any("Non-portable" in e for e in self.validate()))
                self.assertEqual(len(list((self.root / "preview").glob("*.html"))), 2)

    def test_missing_anchor(self):
        self.artifacts("# Present\n\n[Missing](#absent)")
        self.assertTrue(any("Missing anchor" in e for e in self.validate()))

    def test_code_examples_are_not_links(self):
        self.artifacts('```html\n<img src="./example.png">\n```')
        self.assertEqual(self.validate(), [])

    def test_mismatched_descriptions(self):
        self.artifacts("# Wheel", "# Source")
        self.assertIn("Descriptions differ between built artifacts", self.validate())

    def test_missing_artifacts(self):
        self.assertIn("Both a wheel and source distribution are required", self.validate())

    def test_empty_or_wrong_content_type(self):
        for text, kind in [("", "text/markdown"), ("# Hello", "text/plain")]:
            with self.subTest(kind=kind):
                self.artifacts(text, content_type=kind)
                self.assertTrue(self.validate())

    def test_ambiguous_metadata(self):
        self.artifacts("# Hello")
        with zipfile.ZipFile(self.root / "example.whl", "a") as archive:
            archive.writestr("other.dist-info/METADATA", "")
        self.assertTrue(any("exactly one wheel METADATA" in e for e in self.validate()))

    def test_render_failure(self):
        self.artifacts("# Hello")
        with patch.object(validator, "render", return_value=None):
            self.assertTrue(any("rendering failed" in e for e in self.validate()))

    def test_twine_failure_keeps_previews(self):
        self.artifacts("# Hello")
        with patch.object(validator.subprocess, "run", return_value=subprocess.CompletedProcess([], 1)):
            errors = validator.validate(self.root, self.root / "preview")
        self.assertIn("twine check --strict failed", errors)
        self.assertEqual(len(list((self.root / "preview").glob("*.html"))), 2)


if __name__ == "__main__":
    unittest.main()
