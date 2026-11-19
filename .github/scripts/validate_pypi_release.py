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

"""Validate an existing CI build for manual PyPI publication without executing it."""

import argparse
import base64
import json
import os
import re
import subprocess
import tarfile
import tomllib
from email.parser import BytesParser
from pathlib import Path
from zipfile import ZipFile

BUILD_WORKFLOW = ".github/workflows/test-and-build-wheel.yml"
EXPECTED_JOBS = {"build-wheel", *(f"test-wheel ({v})" for v in ("3.11", "3.12", "3.13"))}


class ArtifactUnavailable(ValueError):
    """A successful build no longer has a downloadable distribution artifact."""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_inputs(version, run_id=None):
    require(
        re.fullmatch(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", version),
        "Version must use stable X.Y.Z format, without v.",
    )
    if run_id is not None:
        require(re.fullmatch(r"[1-9][0-9]*", run_id), "Build run ID must be a positive integer.")


def api(repository, endpoint, *, paginate=False):
    command = ["gh", "api", f"repos/{repository}/{endpoint}"]
    if paginate:
        command.extend(["--paginate", "--slurp"])
    return json.loads(subprocess.check_output(command, text=True))


def tag_commit(repository, version):
    obj = api(repository, f"git/ref/tags/v{version}")["object"]
    while obj["type"] == "tag":
        obj = api(repository, f"git/tags/{obj['sha']}")["object"]
    require(obj["type"] == "commit", "Release tag must resolve to a commit.")
    return obj["sha"]


def validate_build(repository, version, run_id):
    validate_inputs(version, run_id)
    run = api(repository, f"actions/runs/{run_id}")
    for field in ("repository", "head_repository"):
        require(
            (run.get(field) or {}).get("full_name", "").casefold() == repository.casefold(),
            "Build must originate from this repository, not a fork.",
        )
    require(run["path"] == BUILD_WORKFLOW, "Run must use the distribution build workflow.")
    require(run["event"] in {"push", "schedule"}, "Build must be a trusted push or scheduled run.")
    require(run["head_branch"] in {"main", "develop", f"v{version}"}, "Build ref is not a release source.")
    require(run["status"] == "completed" and run["conclusion"] == "success", "Build must have succeeded.")
    sha = tag_commit(repository, version)
    require(run["head_sha"] == sha, "Build commit does not match the release tag.")

    pages = api(repository, f"actions/runs/{run_id}/jobs?filter=all&per_page=100", paginate=True)
    jobs = [job for page in pages for job in page["jobs"]]
    for name in EXPECTED_JOBS:
        matching = [job for job in jobs if job["name"] == name]
        require(matching, f"Required build check is missing: {name}")
        latest_attempt = max(job["run_attempt"] for job in matching)
        require(latest_attempt <= run["run_attempt"], "Build was rerun during validation; start validation again.")
        matching = [job for job in matching if job["run_attempt"] == latest_attempt]
        require(
            len(matching) == 1 and matching[0]["conclusion"] == "success",
            f"Required build check did not succeed: {name}",
        )

    content = api(repository, f"contents/pyproject.toml?ref={sha}")
    project = tomllib.loads(base64.b64decode(content["content"]).decode())
    require(project["project"]["version"] == version, "Source package version does not match the release.")

    pages = api(repository, f"actions/runs/{run_id}/artifacts?per_page=100", paginate=True)
    artifacts = [artifact for page in pages for artifact in page["artifacts"]]
    expected_name = f"v{version}-build" if run["head_branch"] == f"v{version}" else "pr-build"
    matching = [artifact for artifact in artifacts if artifact["name"] == expected_name]
    if not artifacts:
        raise ArtifactUnavailable("Build artifact is missing; select a new successful build.")
    require(len(matching) == 1, f"Expected exactly one {expected_name} artifact.")
    artifact = matching[0]
    if artifact["expired"]:
        raise ArtifactUnavailable("Build artifact has expired; select a new successful build.")
    return {"artifact_id": str(artifact["id"]), "sha": sha, "tag": f"v{version}"}


def select_build(repository, version, run_id=""):
    validate_inputs(version, run_id if run_id else None)
    if run_id:
        return {"run_id": run_id, **validate_build(repository, version, run_id)}

    sha = tag_commit(repository, version)
    pages = api(
        repository,
        f"actions/workflows/{Path(BUILD_WORKFLOW).name}/runs?head_sha={sha}&status=success&per_page=100",
        paginate=True,
    )
    runs = [run for page in pages for run in page["workflow_runs"]]
    runs.sort(key=lambda run: (run["created_at"], run["id"]), reverse=True)
    for run in runs:
        if (
            run["head_sha"] != sha
            or run["event"] not in {"push", "schedule"}
            or run["head_branch"] not in {"main", "develop", f"v{version}"}
            or run["status"] != "completed"
            or run["conclusion"] != "success"
        ):
            continue
        selected_id = str(run["id"])
        try:
            outputs = validate_build(repository, version, selected_id)
        except ArtifactUnavailable as error:
            print(f"Skipping build {selected_id}: {error}")
            continue
        require(outputs["sha"] == sha, "Release tag changed during build selection; start validation again.")
        return {"run_id": selected_id, **outputs}
    raise ValueError(
        f"No successful distribution build with an available artifact matches v{version} ({sha}). "
        "Select an explicit build run ID or obtain a new successful build for this commit."
    )


def validate_metadata(data, version):
    metadata = BytesParser().parsebytes(data)
    require(metadata.get_all("Name") == ["nemoguardrails"], "Distribution package name is incorrect.")
    require(metadata.get_all("Version") == [version], "Distribution version does not match the release.")


def validate_distributions(directory, version):
    validate_inputs(version)
    wheel_name = f"nemoguardrails-{version}-py3-none-any.whl"
    sdist_name = f"nemoguardrails-{version}.tar.gz"
    require(
        {p.name for p in directory.iterdir()} == {wheel_name, sdist_name},
        "Artifact must contain exactly the expected wheel and source archive.",
    )
    for path in directory.iterdir():
        require(path.is_file() and not path.is_symlink(), "Distributions must be regular files.")
    with ZipFile(directory / wheel_name) as wheel:
        metadata_name = f"nemoguardrails-{version}.dist-info/METADATA"
        require(wheel.namelist().count(metadata_name) == 1, "Wheel must contain one expected METADATA file.")
        validate_metadata(wheel.read(metadata_name), version)
    with tarfile.open(directory / sdist_name) as sdist:
        metadata_name = f"nemoguardrails-{version}/PKG-INFO"
        members = [m for m in sdist.getmembers() if m.name == metadata_name]
        require(len(members) == 1 and members[0].isfile(), "Source archive must contain one regular PKG-INFO file.")
        metadata_file = sdist.extractfile(members[0])
        require(metadata_file is not None, "Source archive metadata is missing.")
        with metadata_file:
            validate_metadata(metadata_file.read(), version)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "distributions", "tag"))
    args = parser.parse_args()
    version = os.environ["RELEASE_VERSION"]
    validate_inputs(version)
    if args.mode == "distributions":
        validate_distributions(Path("dist"), version)
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as summary:
            summary.write("\nDistribution metadata validated:\n\n")
            for path in sorted(Path("dist").iterdir()):
                summary.write(f"- `{path.name}`\n")
        return

    repository = os.environ["GITHUB_REPOSITORY"]
    if args.mode == "tag":
        require(
            tag_commit(repository, version) == os.environ["RELEASE_SHA"],
            "Release tag changed after validation; refusing to publish.",
        )
        return

    outputs = select_build(repository, version, os.environ.get("BUILD_RUN_ID", ""))
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
        for key, value in outputs.items():
            output.write(f"{key}={value}\n")
    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as summary:
        summary.write(f"## PyPI release {version}\n\n")
        summary.write(f"- Tag: `{outputs['tag']}`\n- Commit: `{outputs['sha']}`\n")
        summary.write(f"- Build: https://github.com/{repository}/actions/runs/{outputs['run_id']}\n")
        summary.write(f"- Artifact ID: `{outputs['artifact_id']}`\n")


if __name__ == "__main__":
    main()
