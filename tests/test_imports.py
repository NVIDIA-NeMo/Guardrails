# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import warnings
from unittest.mock import MagicMock, patch

import pytest

from nemoguardrails.imports import (
    check_optional_dependency,
    get_optional_dependency,
    import_optional_dependency,
    optional_import,
)


class TestOptionalImport:
    def test_successful_import(self):
        module = optional_import("sys")
        assert module is not None
        assert hasattr(module, "path")

    def test_missing_module_raise(self):
        with pytest.raises(ImportError) as exc_info:
            optional_import("nonexistent_module_xyz", error="raise")
        assert "Missing optional dependency" in str(exc_info.value)
        assert "nonexistent_module_xyz" in str(exc_info.value)

    def test_missing_module_raise_with_extra(self):
        with pytest.raises(ImportError) as exc_info:
            optional_import("nonexistent_module_xyz", error="raise", extra="test")
        assert "Missing optional dependency" in str(exc_info.value)
        assert "poetry install -E test" in str(exc_info.value)

    def test_missing_module_raise_with_package_name(self):
        with pytest.raises(ImportError) as exc_info:
            optional_import("nonexistent_xyz", package_name="different-package", error="raise")
        assert "different-package" in str(exc_info.value)

    def test_missing_module_warn(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = optional_import("nonexistent_module_xyz", error="warn")
            assert result is None
            assert len(w) == 1
            assert issubclass(w[0].category, ImportWarning)
            assert "Missing optional dependency" in str(w[0].message)
            assert "nonexistent_module_xyz" in str(w[0].message)

    def test_missing_module_warn_with_extra(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = optional_import("nonexistent_module_xyz", error="warn", extra="test")
            assert result is None
            assert len(w) == 1
            assert "poetry install -E test" in str(w[0].message)

    def test_missing_module_ignore(self):
        result = optional_import("nonexistent_module_xyz", error="ignore")
        assert result is None


class TestCheckOptionalDependency:
    def test_available_dependency(self):
        assert check_optional_dependency("sys") is True

    def test_unavailable_dependency(self):
        assert check_optional_dependency("nonexistent_module_xyz") is False

    def test_with_package_name(self):
        assert check_optional_dependency("sys", package_name="system") is True

    def test_with_extra(self):
        assert check_optional_dependency("nonexistent_xyz", extra="test") is False


class TestImportOptionalDependency:
    def test_successful_import(self):
        module = import_optional_dependency("sys", errors="raise")
        assert module is not None
        assert hasattr(module, "path")

    def test_missing_module_raise(self):
        with pytest.raises(ImportError) as exc_info:
            import_optional_dependency("nonexistent_module_xyz", errors="raise")
        assert "Missing optional dependency" in str(exc_info.value)
        assert "nonexistent_module_xyz" in str(exc_info.value)

    def test_missing_module_raise_with_extra(self):
        with pytest.raises(ImportError) as exc_info:
            import_optional_dependency("nonexistent_module_xyz", errors="raise", extra="test")
        assert "Missing optional dependency" in str(exc_info.value)
        assert "poetry install -E test" in str(exc_info.value)

    def test_missing_module_warn(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = import_optional_dependency("nonexistent_module_xyz", errors="warn")
            assert result is None
            assert len(w) == 1
            assert issubclass(w[0].category, ImportWarning)
            assert "Missing optional dependency" in str(w[0].message)

    def test_missing_module_warn_with_extra(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = import_optional_dependency("nonexistent_module_xyz", errors="warn", extra="test")
            assert result is None
            assert len(w) == 1
            assert "poetry install -E test" in str(w[0].message)

    def test_missing_module_ignore(self):
        result = import_optional_dependency("nonexistent_module_xyz", errors="ignore")
        assert result is None

    def test_invalid_errors_parameter(self):
        with pytest.raises(AssertionError):
            import_optional_dependency("sys", errors="invalid")

    @patch("nemoguardrails.imports.importlib.import_module")
    def test_version_check_success(self, mock_import):
        mock_module = MagicMock()
        mock_module.__version__ = "2.0.0"
        mock_import.return_value = mock_module

        result = import_optional_dependency("test_module", min_version="1.0.0", errors="raise")
        assert result == mock_module

    def test_version_check_fail_raise(self):
        with pytest.raises(ImportError) as exc_info:
            import_optional_dependency("pytest", min_version="999.0.0", errors="raise")
        assert "requires version '999.0.0' or newer" in str(exc_info.value)
        assert "currently installed" in str(exc_info.value)

    def test_version_check_fail_warn(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = import_optional_dependency("pytest", min_version="999.0.0", errors="warn")
            assert result is not None
            assert len(w) == 1
            assert "requires version '999.0.0' or newer" in str(w[0].message)
            assert "currently installed" in str(w[0].message)

    @patch("nemoguardrails.imports.importlib.import_module")
    def test_version_check_no_version_attribute(self, mock_import):
        mock_module = MagicMock(spec=[])
        del mock_module.__version__
        mock_import.return_value = mock_module

        result = import_optional_dependency("test_module", min_version="1.0.0", errors="raise")
        assert result == mock_module

    @patch("nemoguardrails.imports.importlib.import_module")
    def test_version_check_packaging_not_available(self, mock_import):
        mock_module = MagicMock()
        mock_module.__version__ = "1.0.0"
        mock_import.return_value = mock_module

        with patch("nemoguardrails.imports.importlib.import_module") as mock_inner_import:

            def side_effect(name):
                if name == "test_module":
                    return mock_module
                if name == "packaging":
                    raise ImportError("packaging not available")
                raise ImportError(f"Module {name} not found")

            mock_inner_import.side_effect = side_effect

            result = import_optional_dependency("test_module", min_version="1.0.0", errors="raise")
            assert result == mock_module


class TestGetOptionalDependency:
    def test_get_known_dependency_available(self):
        module = get_optional_dependency("langchain", errors="ignore")
        if module:
            assert hasattr(module, "__name__")

    def test_get_unknown_dependency_raise(self):
        with pytest.raises(ImportError):
            get_optional_dependency("nonexistent_xyz_module", errors="raise")

    def test_get_unknown_dependency_warn(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = get_optional_dependency("nonexistent_xyz_module", errors="warn")
            assert result is None
            assert len(w) == 1

    def test_get_unknown_dependency_ignore(self):
        result = get_optional_dependency("nonexistent_xyz_module", errors="ignore")
        assert result is None

    def test_get_dependency_with_extra(self):
        try:
            import openai  # noqa: F401

            pytest.skip("openai is installed, cannot test missing dependency")
        except ImportError:
            with pytest.raises(ImportError) as exc_info:
                get_optional_dependency("openai", errors="raise")
            assert "openai" in str(exc_info.value)
