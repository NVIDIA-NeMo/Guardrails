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

"""Offline release provenance and distribution validation tests."""

import base64
import importlib.util
import io
import sys
import tarfile
from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[2] / ".github/scripts/validate_pypi_release.py"
spec = importlib.util.spec_from_file_location("validate_pypi_release", SCRIPT)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)

REPOSITORY = "NVIDIA-NeMo/Guardrails"
VERSION = "0.25.0"
RUN_ID = "44480300529"
SHA = "a" * 40
TAG_SHA = "b" * 40
DISCOVERY_ENDPOINT = f"actions/workflows/test-and-build-wheel.yml/runs?head_sha={SHA}&status=success&per_page=100"


@pytest.fixture
def responses(monkeypatch):
    payloads = {
        f"actions/runs/{RUN_ID}": {
            "repository": {"full_name": REPOSITORY},
            "head_repository": {"full_name": REPOSITORY},
            "path": release.BUILD_WORKFLOW,
            "event": "push",
            "head_branch": "develop",
            "status": "completed",
            "conclusion": "success",
            "head_sha": SHA,
            "run_attempt": 2,
        },
        f"git/ref/tags/v{VERSION}": {"object": {"type": "tag", "sha": TAG_SHA}},
        f"git/tags/{TAG_SHA}": {"object": {"type": "commit", "sha": SHA}},
        f"actions/runs/{RUN_ID}/jobs?filter=all&per_page=100": [
            {"jobs": [{"name": name, "conclusion": "success", "run_attempt": 2}]}
            for name in sorted(release.EXPECTED_JOBS)
        ],
        f"contents/pyproject.toml?ref={SHA}": {
            "content": base64.b64encode(f'[project]\nversion = "{VERSION}"\n'.encode()).decode()
        },
        f"actions/runs/{RUN_ID}/artifacts?per_page=100": [
            {"artifacts": []},
            {"artifacts": [{"id": 123, "name": "pr-build", "expired": False}]},
        ],
    }

    def api(repository, endpoint, *, paginate=False):
        assert repository == REPOSITORY
        assert paginate == endpoint.endswith("per_page=100")
        return payloads[endpoint]

    monkeypatch.setattr(release, "api", api)
    return payloads


@pytest.fixture
def discovered_runs(responses):
    run = dict(responses[f"actions/runs/{RUN_ID}"], id=int(RUN_ID), created_at="2026-11-19T07:00:00Z")
    pages = [{"workflow_runs": []}, {"workflow_runs": [run]}]
    responses[DISCOVERY_ENDPOINT] = pages
    return pages


def add_newer_run(responses, discovered_runs):
    run_id = str(int(RUN_ID) + 1)
    prefix = f"actions/runs/{RUN_ID}"
    for endpoint, payload in list(responses.items()):
        if endpoint.startswith(prefix):
            responses[endpoint.replace(prefix, f"actions/runs/{run_id}", 1)] = deepcopy(payload)
    run = dict(responses[f"actions/runs/{run_id}"], id=int(run_id), created_at="2026-11-19T08:00:00Z")
    discovered_runs[0]["workflow_runs"].append(run)
    return run_id


def test_discovers_build_at_exact_release_commit(responses, discovered_runs):
    assert release.select_build(REPOSITORY, VERSION) == {
        "run_id": RUN_ID,
        "artifact_id": "123",
        "sha": SHA,
        "tag": f"v{VERSION}",
    }


def test_discovers_newest_build_across_pages(responses, discovered_runs):
    newer_id = add_newer_run(responses, discovered_runs)
    assert release.select_build(REPOSITORY, VERSION)["run_id"] == newer_id


def test_discovery_orders_equal_timestamps_by_id(responses, discovered_runs):
    newer_id = add_newer_run(responses, discovered_runs)
    discovered_runs[0]["workflow_runs"][0]["created_at"] = discovered_runs[1]["workflow_runs"][0]["created_at"]
    assert release.select_build(REPOSITORY, VERSION)["run_id"] == newer_id


@pytest.mark.parametrize("problem", ["missing", "expired"])
def test_discovery_skips_unavailable_artifact_for_same_commit(responses, discovered_runs, problem):
    newer_id = add_newer_run(responses, discovered_runs)
    artifacts = responses[f"actions/runs/{newer_id}/artifacts?per_page=100"][1]["artifacts"]
    if problem == "missing":
        artifacts.clear()
    else:
        artifacts[0]["expired"] = True
    assert release.select_build(REPOSITORY, VERSION)["run_id"] == RUN_ID


@pytest.mark.parametrize("problem", ["wrong-name", "duplicate", "failed-check", "wrong-repository"])
def test_discovery_does_not_fallback_on_invalid_build(responses, discovered_runs, problem):
    newer_id = add_newer_run(responses, discovered_runs)
    artifacts = responses[f"actions/runs/{newer_id}/artifacts?per_page=100"][1]["artifacts"]
    if problem == "wrong-name":
        artifacts[0]["name"] = "unexpected-build"
    elif problem == "duplicate":
        artifacts.append(dict(artifacts[0], id=456))
    elif problem == "failed-check":
        responses[f"actions/runs/{newer_id}/jobs?filter=all&per_page=100"][0]["jobs"][0]["conclusion"] = "failure"
    else:
        responses[f"actions/runs/{newer_id}"]["head_repository"]["full_name"] = "fork/repository"
    with pytest.raises(ValueError):
        release.select_build(REPOSITORY, VERSION)


@pytest.mark.parametrize(
    "field,value",
    [
        ("head_sha", "c" * 40),
        ("event", "pull_request"),
        ("head_branch", "feature"),
        ("status", "in_progress"),
        ("conclusion", "failure"),
    ],
)
def test_discovery_ignores_ineligible_list_results(responses, discovered_runs, field, value):
    add_newer_run(responses, discovered_runs)
    discovered_runs[0]["workflow_runs"][0][field] = value
    assert release.select_build(REPOSITORY, VERSION)["run_id"] == RUN_ID


def test_discovery_reports_no_available_build(responses, discovered_runs):
    responses[f"actions/runs/{RUN_ID}/artifacts?per_page=100"][1]["artifacts"].clear()
    with pytest.raises(ValueError, match="No successful distribution build"):
        release.select_build(REPOSITORY, VERSION)


def test_explicit_run_override_does_not_search(responses):
    assert release.select_build(REPOSITORY, VERSION, RUN_ID)["run_id"] == RUN_ID


def test_explicit_expired_run_does_not_fallback(responses, discovered_runs):
    add_newer_run(responses, discovered_runs)
    responses[f"actions/runs/{RUN_ID}/artifacts?per_page=100"][1]["artifacts"][0]["expired"] = True
    with pytest.raises(release.ArtifactUnavailable):
        release.select_build(REPOSITORY, VERSION, RUN_ID)


def test_discovery_rejects_tag_moved_while_selecting(responses, discovered_runs, monkeypatch):
    monkeypatch.setattr(release, "validate_build", lambda *_: {"sha": "c" * 40})
    with pytest.raises(ValueError, match="tag changed during build selection"):
        release.select_build(REPOSITORY, VERSION)


def test_validation_only_workflow_prevents_writes_and_uses_resolved_run():
    workflow_path = SCRIPT.parents[1] / "workflows/publish-pypi-approval.yml"
    workflow = yaml.load(workflow_path.read_text(), Loader=yaml.BaseLoader)
    inputs = workflow["on"]["workflow_dispatch"]["inputs"]
    assert inputs["validate_only"]["type"] == "boolean"
    assert inputs["validate_only"]["default"] == "true"
    assert inputs["build_run_id"]["required"] == "false"
    jobs = workflow["jobs"]
    assert set(jobs["validate"]["permissions"].values()) == {"read"}
    for name, job in jobs.items():
        if "write" in job.get("permissions", {}).values():
            assert job["if"] == "${{ !inputs.validate_only }}", name
            assert "validate" in job["needs"]
        for step in job["steps"]:
            if step.get("uses", "").startswith("actions/download-artifact@"):
                assert "outputs.run_id" in step["with"]["run-id"]
                assert "outputs.artifact_id" in step["with"]["artifact-ids"]


def test_branch_build_with_annotated_tag_and_paginated_results(responses):
    assert release.validate_build(REPOSITORY, VERSION, RUN_ID) == {
        "artifact_id": "123",
        "sha": SHA,
        "tag": f"v{VERSION}",
    }


def test_tag_build_with_lightweight_tag(responses):
    responses[f"actions/runs/{RUN_ID}"]["head_branch"] = f"v{VERSION}"
    responses[f"git/ref/tags/v{VERSION}"]["object"] = {"type": "commit", "sha": SHA}
    responses[f"actions/runs/{RUN_ID}/artifacts?per_page=100"][1]["artifacts"][0]["name"] = f"v{VERSION}-build"
    assert release.validate_build(REPOSITORY, VERSION, RUN_ID)["sha"] == SHA


def test_scheduled_build(responses):
    responses[f"actions/runs/{RUN_ID}"]["event"] = "schedule"
    assert release.validate_build(REPOSITORY, VERSION, RUN_ID)["artifact_id"] == "123"


@pytest.mark.parametrize(
    "field,value",
    [
        ("repository", {"full_name": "other/repository"}),
        ("head_repository", {"full_name": "fork/Guardrails"}),
        ("head_repository", None),
        ("path", ".github/workflows/another.yml"),
        ("event", "pull_request"),
        ("event", "workflow_dispatch"),
        ("head_branch", "feature"),
        ("head_sha", "c" * 40),
        ("status", "in_progress"),
        ("conclusion", "failure"),
        ("conclusion", "cancelled"),
    ],
)
def test_rejects_untrusted_or_unsuccessful_run(responses, field, value):
    responses[f"actions/runs/{RUN_ID}"][field] = value
    with pytest.raises(ValueError):
        release.validate_build(REPOSITORY, VERSION, RUN_ID)


@pytest.mark.parametrize("name", sorted(release.EXPECTED_JOBS))
@pytest.mark.parametrize("conclusion", ["failure", "skipped", None, "missing"])
def test_rejects_failed_skipped_or_missing_checks(responses, name, conclusion):
    pages = responses[f"actions/runs/{RUN_ID}/jobs?filter=all&per_page=100"]
    for page in pages:
        for job in list(page["jobs"]):
            if job["name"] == name:
                if conclusion == "missing":
                    page["jobs"].remove(job)
                else:
                    job["conclusion"] = conclusion
    with pytest.raises(ValueError, match="Required build check"):
        release.validate_build(REPOSITORY, VERSION, RUN_ID)


def test_accepts_rerunning_only_failed_jobs(responses):
    pages = responses[f"actions/runs/{RUN_ID}/jobs?filter=all&per_page=100"]
    for page in pages:
        page["jobs"][0]["run_attempt"] = 1
    pages.append({"jobs": [{"name": "test-wheel (3.12)", "conclusion": "success", "run_attempt": 2}]})
    for page in pages[:-1]:
        if page["jobs"][0]["name"] == "test-wheel (3.12)":
            page["jobs"][0]["conclusion"] = "failure"
    assert release.validate_build(REPOSITORY, VERSION, RUN_ID)["artifact_id"] == "123"


@pytest.mark.parametrize("problem", ["failure", "skipped", "duplicate", "new-attempt"])
def test_rejects_invalid_latest_job_execution(responses, problem):
    pages = responses[f"actions/runs/{RUN_ID}/jobs?filter=all&per_page=100"]
    job = dict(pages[0]["jobs"][0])
    if problem == "new-attempt":
        job["run_attempt"] = 3
    elif problem != "duplicate":
        pages[0]["jobs"][0]["run_attempt"] = 1
        job["conclusion"] = problem
    pages.append({"jobs": [job]})
    with pytest.raises(ValueError):
        release.validate_build(REPOSITORY, VERSION, RUN_ID)


@pytest.mark.parametrize("problem", ["expired", "missing", "duplicate", "wrong-name"])
def test_rejects_unavailable_or_ambiguous_artifacts(responses, problem):
    artifacts = responses[f"actions/runs/{RUN_ID}/artifacts?per_page=100"][1]["artifacts"]
    if problem == "expired":
        artifacts[0]["expired"] = True
    elif problem == "missing":
        artifacts.clear()
    elif problem == "duplicate":
        artifacts.append(dict(artifacts[0], id=456))
    else:
        artifacts[0]["name"] = f"v{VERSION}-build"
    with pytest.raises(ValueError):
        release.validate_build(REPOSITORY, VERSION, RUN_ID)


def test_rejects_source_version_mismatch(responses):
    responses[f"contents/pyproject.toml?ref={SHA}"]["content"] = base64.b64encode(
        b'[project]\nversion = "0.24.0"\n'
    ).decode()
    with pytest.raises(ValueError, match="Source package version"):
        release.validate_build(REPOSITORY, VERSION, RUN_ID)


@pytest.mark.parametrize(
    "version,run_id",
    [
        ("v0.25.0", RUN_ID),
        ("0.25.0rc1", RUN_ID),
        ("0.25.0\n", RUN_ID),
        ("00.25.0", RUN_ID),
        ("0.25.0;echo bad", RUN_ID),
        (VERSION, "0"),
        (VERSION, "-1"),
        (VERSION, "123\n"),
        (VERSION, "123/other"),
    ],
)
@pytest.mark.parametrize("validate", [release.validate_build, release.select_build])
def test_rejects_invalid_inputs_before_api(monkeypatch, version, run_id, validate):
    def no_api(*args, **kwargs):
        pytest.fail("Invalid input must not reach the API")

    monkeypatch.setattr(release, "api", no_api)
    with pytest.raises(ValueError):
        validate(REPOSITORY, version, run_id)


@pytest.fixture
def distributions(tmp_path):
    def write(*, wheel_version=VERSION, sdist_version=VERSION, name="nemoguardrails", sdist_link=False):
        wheel_metadata = f"Name: {name}\nVersion: {wheel_version}\n\n".encode()
        with ZipFile(tmp_path / f"nemoguardrails-{VERSION}-py3-none-any.whl", "w") as wheel:
            wheel.writestr(f"nemoguardrails-{VERSION}.dist-info/METADATA", wheel_metadata)
        metadata = f"Name: {name}\nVersion: {sdist_version}\n\n".encode()
        with tarfile.open(tmp_path / f"nemoguardrails-{VERSION}.tar.gz", "w:gz") as sdist:
            member = tarfile.TarInfo(f"nemoguardrails-{VERSION}/PKG-INFO")
            if sdist_link:
                member.type = tarfile.SYMTYPE
                member.linkname = "/etc/passwd"
                sdist.addfile(member)
            else:
                member.size = len(metadata)
                sdist.addfile(member, io.BytesIO(metadata))
        return tmp_path

    return write


def test_validates_both_distributions_without_installing(distributions):
    release.validate_distributions(distributions(), VERSION)


@pytest.mark.parametrize(
    "options",
    [
        {"wheel_version": "0.24.0"},
        {"sdist_version": "0.24.0"},
        {"name": "other-package"},
        {"sdist_link": True},
    ],
)
def test_rejects_invalid_distribution_metadata(distributions, options):
    with pytest.raises(ValueError):
        release.validate_distributions(distributions(**options), VERSION)


@pytest.mark.parametrize("problem", ["extra", "missing", "symlink"])
def test_rejects_unexpected_distribution_files(distributions, problem):
    directory = distributions()
    wheel = directory / f"nemoguardrails-{VERSION}-py3-none-any.whl"
    if problem == "extra":
        (directory / "unexpected.whl").touch()
    elif problem == "missing":
        wheel.unlink()
    else:
        wheel.unlink()
        wheel.symlink_to(directory / f"nemoguardrails-{VERSION}.tar.gz")
    with pytest.raises(ValueError):
        release.validate_distributions(directory, VERSION)


@pytest.mark.parametrize("run_id", [RUN_ID, ""])
def test_cli_writes_validated_outputs_and_summary(responses, discovered_runs, monkeypatch, tmp_path, run_id):
    output = tmp_path / "output"
    summary = tmp_path / "summary"
    for key, value in {
        "RELEASE_VERSION": VERSION,
        "BUILD_RUN_ID": run_id,
        "GITHUB_REPOSITORY": REPOSITORY,
        "GITHUB_OUTPUT": str(output),
        "GITHUB_STEP_SUMMARY": str(summary),
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "build"])
    release.main()
    assert output.read_text() == f"run_id={RUN_ID}\nartifact_id=123\nsha={SHA}\ntag=v{VERSION}\n"
    assert f"actions/runs/{RUN_ID}" in summary.read_text()


def test_cli_rejects_tag_moved_during_approval(responses, monkeypatch):
    monkeypatch.setenv("RELEASE_VERSION", VERSION)
    monkeypatch.setenv("GITHUB_REPOSITORY", REPOSITORY)
    monkeypatch.setenv("RELEASE_SHA", "c" * 40)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "tag"])
    with pytest.raises(ValueError, match="tag changed"):
        release.main()
