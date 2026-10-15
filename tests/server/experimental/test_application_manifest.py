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

"""Test the manifest trust boundary and the server-owned command surface."""

import asyncio
import json

import pytest
import yaml
from jsonschema import Draft202012Validator
from typer.testing import CliRunner

from nemoguardrails.server.experimental import _application, _application_manifest, cli
from nemoguardrails.server.experimental._application_manifest import ApplicationManifest, ManifestError, load_manifest
from nemoguardrails.server.experimental._configuration import resolve_provider
from nemoguardrails.server.experimental.application import (
    ConfigEntry,
    ProviderConfig,
    ProxyConfig,
    create_proxy_application,
)
from nemoguardrails.server.experimental.rails import _repository

runner = CliRunner()


@pytest.fixture
def document(tmp_path):
    directory = tmp_path / "rails"
    directory.mkdir()
    (directory / "config.yml").write_text("models: []\n", encoding="utf-8")
    return {
        "version": 1,
        "rails_configs": [{"id": "standard", "config_path": "rails"}],
        "providers": [
            {
                "provider": "openai",
                "instance_id": "public",
                "base_url": "https://provider.test/v1",
                "operations": ["chat_completions", "responses"],
                "rails_config_id": "standard",
            }
        ],
    }


def write_manifest(tmp_path, document):
    path = tmp_path / "proxy.yaml"
    path.write_text(yaml.safe_dump(json.loads(json.dumps(document)), sort_keys=False), encoding="utf-8")
    return path


@pytest.mark.parametrize("dialect", [None, "litellm"])
def test_yaml_and_python_resolve_identical_bindings_from_another_directory(tmp_path, document, monkeypatch, dialect):
    document["providers"][0]["dialect"] = dialect
    path = write_manifest(tmp_path, document)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    config = load_manifest(path)
    expected = ProxyConfig(
        rails_configs=(ConfigEntry("standard", tmp_path / "rails"),),
        providers=(
            ProviderConfig(
                provider="openai",
                instance_id="public",
                base_url="https://provider.test/v1",
                operations=("chat_completions", "responses"),
                rails_config_id="standard",
                dialect=dialect,
            ),
        ),
    )
    assert config == expected
    assert resolve_provider(config.providers[0]) == resolve_provider(expected.providers[0])
    actual_app = create_proxy_application(config)
    expected_app = create_proxy_application(expected)
    assert actual_app.openapi() == expected_app.openapi()


@pytest.mark.parametrize(
    "source",
    [
        "",
        "[]",
        "version: 1\nversion: 1",
        "providers: [{instance_id: a, instance_id: b}]",
        "version: [",
        "version: 1\n---\nversion: 1",
        "version: !!int 1",
        "x: !secret value",
        "x: !!python/object/apply:os.system [false]",
        "x: &a [1]\ny: *a",
        "x: *undefined",
        "<<: {version: 1}",
        "true: value",
        "[a, b]: value",
        "x: 2026-10-15",
        "x: ${PRIVATE_VALUE}",
        "x: " + "[" * 33 + "0" + "]" * 33,
    ],
)
def test_rejects_unsafe_ambiguous_or_unsupported_yaml(tmp_path, source):
    path = tmp_path / "proxy.yaml"
    path.write_text(source, encoding="utf-8")
    with pytest.raises(ManifestError):
        load_manifest(path)


def test_byte_and_node_limits(tmp_path, monkeypatch):
    path = tmp_path / "proxy.yaml"
    path.write_bytes(b"x" * (_application_manifest.MAX_MANIFEST_BYTES + 1))
    with pytest.raises(ManifestError, match="size limit"):
        load_manifest(path)
    monkeypatch.setattr(_application_manifest, "MAX_MANIFEST_NODES", 8)
    path.write_text("x: [1, 2, 3, 4, 5, 6, 7, 8]", encoding="utf-8")
    with pytest.raises(ManifestError, match="structure"):
        load_manifest(path)


@pytest.mark.parametrize("kind", ["missing", "directory", "utf8"])
def test_manifest_file_errors_are_sanitized(tmp_path, kind):
    path = tmp_path / "SECRET"
    if kind == "directory":
        path.mkdir()
    elif kind == "utf8":
        path.write_bytes(b"\xff\xfe")
    result = runner.invoke(cli.app, ["validate", str(path)])
    assert result.exit_code == 2
    assert "SECRET" not in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize("version", [None, True, False, 1.0, "1", 2])
def test_version_requires_exact_supported_integer(tmp_path, document, version):
    document["version"] = version
    with pytest.raises(ManifestError):
        load_manifest(write_manifest(tmp_path, document))


@pytest.mark.parametrize(
    "change",
    [
        {"profile": "single_text.v1"},
        {"variant": "text"},
        {"operations": {"chat_completions": {"variant": "text"}}},
        {"operations": []},
        {"operations": ["chat_completions", "chat_completions"]},
        {"operations": ["unimplemented"]},
        {"provider": "anthropic"},
        {"dialect": "openai"},
        {"rails_config_id": None},
        {"rails_config_id": "unknown"},
        {"factory": "untrusted.module:create"},
        {"base_url": "https://user:SECRET@provider.test/v1"},
        {"base_url": "https://provider.test/v1?api_key=SECRET"},
        {"base_url": "ftp://provider.test"},
        {"base_url": "https://provider.test:SECRET"},
        {"mount_path": "invalid"},
    ],
)
def test_invalid_declarations_fail_closed_without_echoing_values(tmp_path, document, change):
    document["providers"][0].update(change)
    result = runner.invoke(cli.app, ["inspect", str(write_manifest(tmp_path, document))])
    assert result.exit_code == 2
    assert "SECRET" not in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize("target", ["root", "provider", "entry"])
def test_unknown_keys_rejected_with_coordinates_not_source_values(tmp_path, document, target):
    mapping = {"root": document, "provider": document["providers"][0], "entry": document["rails_configs"][0]}[target]
    mapping["SECRET_KEY"] = "SECRET_VALUE"
    result = runner.invoke(cli.app, ["validate", str(write_manifest(tmp_path, document))])
    assert result.exit_code == 2
    assert ("unexpected_keyword_argument" if target == "entry" else "extra_forbidden") in result.output
    assert "line " in result.output and "column " in result.output
    assert "SECRET" not in result.output


@pytest.mark.parametrize("fault", ["duplicate_mount", "duplicate_id", "missing_directory", "missing_config", "prefix"])
def test_assembly_validation_is_shared(tmp_path, document, fault):
    if fault.startswith("duplicate"):
        document["providers"].append(
            {
                **document["providers"][0],
                "instance_id": "other" if fault == "duplicate_mount" else "public",
                "mount_path": "" if fault == "duplicate_mount" else "/other",
            }
        )
    elif fault == "missing_directory":
        document["rails_configs"][0]["config_path"] = "SECRET_MISSING"
    elif fault == "missing_config":
        (tmp_path / "rails/config.yml").unlink()
    else:
        document["path_prefix"] = "invalid"
    result = runner.invoke(cli.app, ["validate", str(write_manifest(tmp_path, document))])
    assert result.exit_code == 2
    assert "Invalid application" in result.output
    assert "SECRET" not in result.output


def test_validation_and_inspection_do_not_load_rails_or_create_owners(tmp_path, document, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("structural commands must not enter runtime construction or load Rails")

    monkeypatch.setattr(_repository.RailsConfig, "from_path", forbidden)
    monkeypatch.setattr(_repository, "Guardrails", forbidden)
    monkeypatch.setattr(_application.ServerRuntime, "create", forbidden)
    (tmp_path / "rails/actions.py").write_text("raise RuntimeError('must not execute')", encoding="utf-8")
    path = write_manifest(tmp_path, document)
    assert runner.invoke(cli.app, ["validate", str(path)]).exit_code == 0
    result = runner.invoke(cli.app, ["inspect", str(path)])
    assert result.exit_code == 0
    inspected = json.loads(result.output)
    provider = inspected["providers"][0]
    assert provider["dialect"] == "native"
    assert provider["rails_config_id"] == "standard"
    assert len(provider["operations"]) == 2
    for operation in provider["operations"]:
        assert operation["profiles"] == {
            "request": "single_text.v1",
            "response": "single_text.v1",
            "stream": "single_text_delta.v1",
        }
        assert operation["buffered"] and operation["streaming"]
    assert inspected["routing"]["unselected_routes"].startswith("passthrough")
    assert inspected["checks"].startswith("Not loaded")


def test_inspection_uses_actual_bound_profiles(tmp_path, document, monkeypatch):
    from dataclasses import replace

    from nemoguardrails.server.experimental.provider.payload import PayloadCapabilityProfile

    actual = cli.resolve_provider

    def resolve(config):
        provider = actual(config)
        endpoint = provider.endpoints[0]
        contract = endpoint.guarded_request_model.projection_contract
        assert contract.profile == PayloadCapabilityProfile.SINGLE_TEXT_V1
        return replace(provider, endpoints=(replace(endpoint, stream=None),))

    monkeypatch.setattr(cli, "resolve_provider", resolve)
    inspected = cli.inspect_config(load_manifest(write_manifest(tmp_path, document)))
    operation = inspected["providers"][0]["operations"][0]
    assert operation["streaming"] is False
    assert operation["profiles"]["stream"] is None


def test_schema_is_derived_and_can_validate_the_example(document):
    result = runner.invoke(cli.app, ["schema"])
    assert result.exit_code == 0
    schema = json.loads(result.output)
    assert schema == ApplicationManifest.model_json_schema()
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(document)
    assert schema["additionalProperties"] is False
    provider_schema = schema["$defs"]["ManifestProviderConfig"]
    assert provider_schema["additionalProperties"] is False
    assert "rails_config_id" in provider_schema["required"]
    assert provider_schema["properties"]["rails_config_id"]["type"] == "string"
    assert schema["$defs"]["ConfigEntry"]["additionalProperties"] is False


def test_serve_uses_existing_assembly_once_and_loopback_defaults(tmp_path, document, monkeypatch):
    import uvicorn

    calls = []
    actual = cli.create_proxy_application

    def create(config):
        calls.append(config)
        return actual(config)

    observed = {}
    monkeypatch.setattr(cli, "create_proxy_application", create)
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: observed.update(app=app, **kwargs))
    result = runner.invoke(cli.app, ["serve", str(write_manifest(tmp_path, document))])
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert observed["host"] == "127.0.0.1"
    assert observed["port"] == 8000
    assert observed["access_log"] is False
    assert observed["proxy_headers"] is False
    assert not hasattr(observed["app"].state, "experimental_proxy_runtime")


@pytest.mark.parametrize("command", ["validate", "inspect", "serve", "schema"])
def test_command_help(command):
    assert runner.invoke(cli.app, [command, "--help"]).exit_code == 0


@pytest.mark.parametrize(
    "arguments", [[], ["serve"], ["serve", "unused", "--port", "-1"], ["serve", "unused", "--port", "65536"]]
)
def test_invalid_cli_usage(arguments):
    assert runner.invoke(cli.app, arguments).exit_code == 2


@pytest.mark.asyncio
async def test_partial_catalog_startup_closes_real_core_workers_and_http_owners(tmp_path, document, monkeypatch):
    """A later parse failure must roll back the already-started real core."""
    from nemoguardrails.guardrails.iorails import IORails
    from nemoguardrails.server.experimental._runtime import ServerRuntime

    invalid = tmp_path / "invalid"
    invalid.mkdir()
    (invalid / "config.yml").write_text("models: [", encoding="utf-8")
    document["rails_configs"].append({"id": "broken", "config_path": "invalid"})
    path = write_manifest(tmp_path, document)
    assert runner.invoke(cli.app, ["validate", str(path)]).exit_code == 0
    application = create_proxy_application(load_manifest(path))
    runtimes = []
    cores = []
    workers = []
    actual_create = ServerRuntime.create
    actual_start = IORails.start

    def create(cls, *args, **kwargs):
        runtime = actual_create(*args, **kwargs)
        runtimes.append(runtime)
        return runtime

    async def start(core):
        await actual_start(core)
        cores.append(core)
        workers.extend(core._generate_async_queue._workers)

    monkeypatch.setattr(ServerRuntime, "create", classmethod(create))
    monkeypatch.setattr(IORails, "start", start)
    try:
        with pytest.raises(yaml.YAMLError):
            async with application.router.lifespan_context(application):
                pytest.fail("invalid catalogue must not become ready")
        assert len(cores) == len(runtimes) == 1
        assert workers and all(worker.done() for worker in workers)
        assert not cores[0]._running
        assert cores[0].rails_manager._http_client._client.is_closed
        assert runtimes[0].http_client.is_closed
        assert not runtimes[0].content_checkers._checkers
        assert not hasattr(application.state, "experimental_proxy_runtime")
    finally:
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        for core in cores:
            await core.rails_manager._http_client.close()
        for runtime in runtimes:
            await runtime.close()
