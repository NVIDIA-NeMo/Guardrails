from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from nemoguardrails.server import api
from tests.recorded.conftest import build_vcr_config

CONFIGS_DIR = str(Path(__file__).parent / "configs")


@pytest.fixture(scope="session")
def vcr_config() -> Dict[str, Any]:
    # TestClient drives the app in-process over httpx to host ``testserver``; ignore it so
    # only the real outbound provider call is recorded.
    return {**build_vcr_config(), "ignore_hosts": ["testserver"]}


@pytest.fixture
def server_client() -> Iterator[TestClient]:
    original_path = api.app.rails_config_path
    api.app.rails_config_path = CONFIGS_DIR
    api.llm_rails_instances.clear()
    try:
        yield TestClient(api.app)
    finally:
        api.app.rails_config_path = original_path
        api.llm_rails_instances.clear()
