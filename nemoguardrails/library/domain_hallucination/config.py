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

"""Configuration management for domain hallucination guard."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class VerificationConfig:
    """Verification settings."""

    level: str = "dns"  # none, dns, http, full
    dns_timeout: float = 4.0
    http_timeout: float = 6.0
    tls_timeout: float = 5.0
    whois_timeout: float = 6.0
    github_timeout: float = 6.0
    github_token: Optional[str] = None


@dataclass
class DetectionConfig:
    """Detection settings."""

    enable_semantic_check: bool = False
    enable_advanced_verification: bool = False
    no_link_fast_pass: bool = True


@dataclass
class ScoringConfig:
    """Scoring and threshold settings."""

    fail_threshold: float = 60.0
    refine_threshold: float = 40.0
    warn_threshold: float = 20.0
    issue_type_scores: Dict[str, float] = field(
        default_factory=lambda: {
            "fake_github_repo": 85.0,
            "non_existent_domain": 80.0,
            "delegated_no_address_record": 50.0,
            "blacklisted_domain": 95.0,
            "tls_certificate_expired": 70.0,
            "tls_hostname_mismatch": 70.0,
            "tls_untrusted_chain": 55.0,
            "tls_verification_failed": 50.0,
            "tls_certificate_expiring_soon": 15.0,
            "recent_domain": 20.0,
            "no_local_kb_evidence": 15.0,
            "semantic_mismatch": 30.0,
            "advanced_verification_failed": 40.0,
        }
    )
    severity_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "critical": 1.5,
            "high": 1.3,
            "medium": 1.0,
            "low": 0.7,
        }
    )
    confidence_boosts: Dict[str, float] = field(
        default_factory=lambda: {
            "high": 1.0,
            "medium": 0.8,
            "low": 0.6,
        }
    )


@dataclass
class KnowledgeBaseConfig:
    """Knowledge base settings."""

    seed_kb_path: Optional[str] = None
    external_kb_root: Optional[str] = None
    auto_load: bool = True


@dataclass
class EnforcementConfig:
    """Enforcement action settings."""

    block_message: str = "[BLOCKED] The response contains unverified information and has been blocked."
    refine_message: str = "[NOTICE] This response may contain unverified information."
    warn_message: str = "[WARNING] Potential unverified information detected."
    append_verification_notice: bool = True


@dataclass
class DomainHallucinationGuardConfig:
    """Main configuration class."""

    verification: VerificationConfig = field(default_factory=VerificationConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    kb: KnowledgeBaseConfig = field(default_factory=KnowledgeBaseConfig)
    enforcement: EnforcementConfig = field(default_factory=EnforcementConfig)
    debug: bool = False
    log_level: str = "INFO"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def save(self, path: str | Path) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> DomainHallucinationGuardConfig:
        """Load configuration from JSON file."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> DomainHallucinationGuardConfig:
        """Create from dictionary."""
        verification_data = data.get("verification", {})
        detection_data = data.get("detection", {})
        scoring_data = data.get("scoring", {})
        kb_data = data.get("kb", {})
        enforcement_data = data.get("enforcement", {})

        return cls(
            verification=VerificationConfig(**verification_data) if verification_data else VerificationConfig(),
            detection=DetectionConfig(**detection_data) if detection_data else DetectionConfig(),
            scoring=ScoringConfig(**scoring_data) if scoring_data else ScoringConfig(),
            kb=KnowledgeBaseConfig(**kb_data) if kb_data else KnowledgeBaseConfig(),
            enforcement=EnforcementConfig(**enforcement_data) if enforcement_data else EnforcementConfig(),
            debug=data.get("debug", False),
            log_level=data.get("log_level", "INFO"),
        )

    @classmethod
    def from_env(cls, prefix: str = "DOMAIN_HALLUCINATION_") -> DomainHallucinationGuardConfig:
        """Create from environment variables."""
        config = cls()

        # Verification
        if f"{prefix}VERIFICATION_LEVEL" in os.environ:
            config.verification.level = os.environ[f"{prefix}VERIFICATION_LEVEL"]
        if f"{prefix}DNS_TIMEOUT" in os.environ:
            config.verification.dns_timeout = float(os.environ[f"{prefix}DNS_TIMEOUT"])
        if f"{prefix}HTTP_TIMEOUT" in os.environ:
            config.verification.http_timeout = float(os.environ[f"{prefix}HTTP_TIMEOUT"])
        if f"{prefix}TLS_TIMEOUT" in os.environ:
            config.verification.tls_timeout = float(os.environ[f"{prefix}TLS_TIMEOUT"])
        if f"{prefix}WHOIS_TIMEOUT" in os.environ:
            config.verification.whois_timeout = float(os.environ[f"{prefix}WHOIS_TIMEOUT"])
        if f"{prefix}GITHUB_TOKEN" in os.environ:
            config.verification.github_token = os.environ[f"{prefix}GITHUB_TOKEN"]

        # Detection
        if f"{prefix}SEMANTIC_CHECK" in os.environ:
            config.detection.enable_semantic_check = os.environ[f"{prefix}SEMANTIC_CHECK"].lower() == "true"
        if f"{prefix}ADVANCED_VERIFICATION" in os.environ:
            config.detection.enable_advanced_verification = (
                os.environ[f"{prefix}ADVANCED_VERIFICATION"].lower() == "true"
            )

        # Scoring
        if f"{prefix}FAIL_THRESHOLD" in os.environ:
            config.scoring.fail_threshold = float(os.environ[f"{prefix}FAIL_THRESHOLD"])
        if f"{prefix}REFINE_THRESHOLD" in os.environ:
            config.scoring.refine_threshold = float(os.environ[f"{prefix}REFINE_THRESHOLD"])
        if f"{prefix}WARN_THRESHOLD" in os.environ:
            config.scoring.warn_threshold = float(os.environ[f"{prefix}WARN_THRESHOLD"])

        # KB
        if f"{prefix}SEED_KB_PATH" in os.environ:
            config.kb.seed_kb_path = os.environ[f"{prefix}SEED_KB_PATH"]
        if f"{prefix}EXTERNAL_KB_ROOT" in os.environ:
            config.kb.external_kb_root = os.environ[f"{prefix}EXTERNAL_KB_ROOT"]

        # Debug
        if f"{prefix}DEBUG" in os.environ:
            config.debug = os.environ[f"{prefix}DEBUG"].lower() == "true"

        return config


# Global config instance
_config_instance: Optional[DomainHallucinationGuardConfig] = None


def get_config() -> DomainHallucinationGuardConfig:
    """Get or create global config instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = DomainHallucinationGuardConfig()
    return _config_instance


def set_config(config: DomainHallucinationGuardConfig) -> None:
    """Set global config instance."""
    global _config_instance
    _config_instance = config


def load_config(path: str | Path) -> DomainHallucinationGuardConfig:
    """Load and set global config from file."""
    config = DomainHallucinationGuardConfig.load(path)
    set_config(config)
    return config
