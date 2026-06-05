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

"""Tests for domain hallucination config module."""

import os
import tempfile
import unittest
from unittest.mock import patch

from nemoguardrails.library.domain_hallucination.config import (
    DomainHallucinationGuardConfig,
    get_config,
    set_config,
)


class TestConfig(unittest.TestCase):
    """Test configuration defaults and persistence."""

    def setUp(self):
        self._original_config = get_config()

    def tearDown(self):
        set_config(self._original_config)

    def test_default_config_has_thresholds(self):
        """Test default scoring thresholds exist."""
        cfg = DomainHallucinationGuardConfig()
        assert cfg.scoring.fail_threshold == 60.0
        assert cfg.scoring.refine_threshold == 40.0
        assert cfg.scoring.warn_threshold == 20.0

    def test_enforcement_config_defaults(self):
        """Test default enforcement fields are populated."""
        cfg = DomainHallucinationGuardConfig()
        assert cfg.enforcement.block_message
        assert cfg.enforcement.refine_message
        assert cfg.enforcement.warn_message
        assert cfg.enforcement.append_verification_notice is True

    def test_set_and_get_config(self):
        """Test global config setter/getter."""
        cfg = DomainHallucinationGuardConfig()
        cfg.scoring.fail_threshold = 75.0
        set_config(cfg)
        assert get_config().scoring.fail_threshold == 75.0

    def test_config_to_dict_roundtrip(self):
        """Test config serializes to dict with key sections."""
        cfg = DomainHallucinationGuardConfig()
        data = cfg.to_dict()
        assert "scoring" in data
        assert "enforcement" in data
        assert "verification" in data
        restored = DomainHallucinationGuardConfig._from_dict(data)
        assert restored.scoring.fail_threshold == cfg.scoring.fail_threshold

    def test_config_from_env(self):
        """Test environment variables override defaults."""
        env = {
            "DOMAIN_HALLUCINATION_FAIL_THRESHOLD": "75",
            "DOMAIN_HALLUCINATION_REFINE_THRESHOLD": "55",
            "DOMAIN_HALLUCINATION_WARN_THRESHOLD": "25",
            "DOMAIN_HALLUCINATION_VERIFICATION_LEVEL": "full",
            "DOMAIN_HALLUCINATION_GITHUB_TOKEN": "token-123",
            "DOMAIN_HALLUCINATION_SEMANTIC_CHECK": "true",
            "DOMAIN_HALLUCINATION_ADVANCED_VERIFICATION": "true",
            "DOMAIN_HALLUCINATION_DEBUG": "true",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = DomainHallucinationGuardConfig.from_env()
        assert cfg.scoring.fail_threshold == 75.0
        assert cfg.scoring.refine_threshold == 55.0
        assert cfg.scoring.warn_threshold == 25.0
        assert cfg.verification.level == "full"
        assert cfg.verification.github_token == "token-123"
        assert cfg.detection.enable_semantic_check is True
        assert cfg.detection.enable_advanced_verification is True
        assert cfg.debug is True

    def test_config_from_env_all_thresholds(self):
        """Test all threshold env vars are loaded."""
        env = {
            "DOMAIN_HALLUCINATION_FAIL_THRESHOLD": "80",
            "DOMAIN_HALLUCINATION_REFINE_THRESHOLD": "50",
            "DOMAIN_HALLUCINATION_WARN_THRESHOLD": "25",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = DomainHallucinationGuardConfig.from_env()
        assert cfg.scoring.fail_threshold == 80.0
        assert cfg.scoring.refine_threshold == 50.0
        assert cfg.scoring.warn_threshold == 25.0

    def test_config_enforcement_fields_all_populated(self):
        """Test all enforcement fields are available."""
        enforcement = DomainHallucinationGuardConfig().enforcement
        assert len(enforcement.block_message) > 0
        assert len(enforcement.refine_message) > 0
        assert len(enforcement.warn_message) > 0
        assert enforcement.append_verification_notice is True

    def test_config_save_load(self):
        """Test config can be saved and loaded from JSON."""
        cfg = DomainHallucinationGuardConfig()
        cfg.scoring.warn_threshold = 12.5
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "domain_config.json")
            cfg.save(path)
            loaded = DomainHallucinationGuardConfig.load(path)
        assert loaded.scoring.warn_threshold == 12.5
        assert loaded.enforcement.block_message == cfg.enforcement.block_message

    def test_config_load_missing_file_raises(self):
        """Test loading a missing config file raises clearly."""
        with self.assertRaises(FileNotFoundError):
            DomainHallucinationGuardConfig.load("missing-domain-config.json")

    def test_from_dict_partial_sections_use_defaults(self):
        """Test partial dict config keeps defaults for missing sections."""
        cfg = DomainHallucinationGuardConfig._from_dict(
            {
                "verification": {"level": "http"},
                "debug": True,
            }
        )
        assert cfg.verification.level == "http"
        assert cfg.scoring.fail_threshold == 60.0
        assert cfg.enforcement.block_message
        assert cfg.debug is True


if __name__ == "__main__":
    unittest.main()
