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

"""Generate the manifest-backed rail installation requirements page."""

import argparse
import shlex
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional

from packaging.markers import Marker

from nemoguardrails.manifests import RailManifest, all_rail_manifests, build_installation_plan
from nemoguardrails.manifests.installation import InstallationPackage, InstallationResource

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_ROOT = PROJECT_ROOT / "docs" / "configure-rails" / "guardrail-catalog"
DOCUMENT_PATH = CATALOG_ROOT / "rail-installation-requirements.mdx"
BLOCK_START = "<!-- BEGIN GENERATED RAIL REQUIREMENTS -->"
BLOCK_END = "<!-- END GENERATED RAIL REQUIREMENTS -->"


def _description(description: Optional[str]) -> str:
    return f": {' '.join(description.split())}" if description else ""


def _provenance(rail_names: tuple[str, ...], selected_count: int) -> str:
    if selected_count == 1:
        return ""
    names = ", ".join(f"`{rail_name}`" for rail_name in rail_names)
    return f"; rails: {names}"


def _install_command(packages: Iterable[InstallationPackage]) -> str:
    requirements = []
    for package in packages:
        requirement = package.distribution + (package.version or "")
        if package.marker is not None:
            requirement = f"{requirement}; {Marker(package.marker)}"
        requirements.append(shlex.quote(requirement))
    return "pip install " + " ".join(requirements)


def _render_packages(title: str, packages: tuple[InstallationPackage, ...], *, selected_count: int = 1) -> list[str]:
    if not packages:
        return []
    lines = [f"**{title}**", ""]
    for package in packages:
        provenance = _provenance(package.rail_names, selected_count)
        provenance_suffix = f" ({provenance.removeprefix('; ')})" if provenance else ""
        lines.append(f"- `{package.requirement}`{provenance_suffix}{_description(package.description)}")
    lines.extend(("", "```bash", _install_command(packages), "```", ""))
    return lines


def _render_resources(title: str, resources: tuple[InstallationResource, ...], *, selected_count: int = 1) -> list[str]:
    if not resources:
        return []
    lines = [f"**{title}**", ""]
    for resource in resources:
        qualifier = "required" if resource.required else "optional"
        provenance = _provenance(resource.rail_names, selected_count)
        lines.append(f"- `{resource.name}` ({qualifier}{provenance}){_description(resource.description)}")
    lines.append("")
    return lines


def _documentation_url(manifest: RailManifest) -> Optional[str]:
    path = manifest.metadata.docs_url
    if path is None or not path.startswith("docs/configure-rails/") or not path.endswith(".mdx"):
        return None
    relative = path.removeprefix("docs/configure-rails/").removesuffix(".mdx")
    return f"/configure-guardrails/{relative}"


def _render_manifest(manifest: RailManifest) -> list[str]:
    display_name = manifest.metadata.display_name or manifest.name
    documentation_url = _documentation_url(manifest)
    heading = f"## {display_name} (`{manifest.name}`)"
    if documentation_url is not None:
        heading = f"## [{display_name}]({documentation_url}) (`{manifest.name}`)"
    lines = [heading, ""]
    plan = build_installation_plan((manifest,))
    required = tuple(package for package in plan.python_packages if package.required)
    optional = tuple(package for package in plan.python_packages if not package.required)
    lines.extend(_render_packages("Required Python packages", required))
    lines.extend(_render_packages("Optional Python packages", optional))
    lines.extend(_render_resources("Environment variables", plan.environment_variables))
    lines.extend(_render_resources("Services", plan.services))
    lines.extend(_render_resources("Models", plan.models))
    if not any((plan.python_packages, plan.environment_variables, plan.services, plan.models)):
        lines.extend(("No additional installation resources are declared.", ""))
    return lines


def _rail_subject(rail_names: tuple[str, ...]) -> str:
    names = [f"`{rail_name}`" for rail_name in rail_names]
    if len(names) == 1:
        return f"The {names[0]} rail"
    if len(names) == 2:
        return f"The {names[0]} and {names[1]} rails"
    return f"The {', '.join(names[:-1])}, and {names[-1]} rails"


def render_page_requirements(manifests: Iterable[RailManifest]) -> str:
    selected = tuple(sorted(manifests, key=lambda manifest: manifest.name))
    if not selected:
        raise ValueError("At least one rail manifest is required to render a catalog requirements block.")
    plan = build_installation_plan(selected)
    lines = [BLOCK_START, "", "## Installation requirements", ""]
    subject = _rail_subject(plan.rail_names)
    verb = "declares" if len(plan.rail_names) == 1 else "declare"
    if not any((plan.python_packages, plan.environment_variables, plan.services, plan.models)):
        verb = "does" if len(plan.rail_names) == 1 else "do"
        lines.extend((f"{subject} {verb} not declare additional installation resources.", ""))
    else:
        lines.extend((f"{subject} {verb} the following installation resources.", ""))
        required = tuple(package for package in plan.python_packages if package.required)
        optional = tuple(package for package in plan.python_packages if not package.required)
        selected_count = len(selected)
        lines.extend(_render_packages("Required Python packages", required, selected_count=selected_count))
        lines.extend(_render_packages("Optional Python packages", optional, selected_count=selected_count))
        lines.extend(
            _render_resources("Environment variables", plan.environment_variables, selected_count=selected_count)
        )
        lines.extend(_render_resources("Services", plan.services, selected_count=selected_count))
        lines.extend(_render_resources("Models", plan.models, selected_count=selected_count))
        rail_options = " ".join(f"--rail {shlex.quote(rail_name)}" for rail_name in plan.rail_names)
        lines.extend(
            (
                "Print the requirements from your installed version:",
                "",
                "```bash",
                f"nemoguardrails rails requirements {rail_options}",
                "```",
                "",
            )
        )
    lines.append(BLOCK_END)
    return "\n".join(lines)


def update_page_requirements(document: str, manifests: Iterable[RailManifest]) -> str:
    block = render_page_requirements(manifests)
    start_count = document.count(BLOCK_START)
    end_count = document.count(BLOCK_END)
    if start_count != end_count or start_count > 1:
        raise ValueError("Catalog documents must contain either one complete generated requirements block or none.")
    if start_count == 1:
        start = document.index(BLOCK_START)
        end = document.index(BLOCK_END, start) + len(BLOCK_END)
        prefix = document[:start].rstrip()
        suffix = document[end:].lstrip()
        document = f"{prefix}\n\n{suffix}".rstrip() + "\n"

    frontmatter_end = document.find("\n---\n", len("---\n"))
    content_start = frontmatter_end + len("\n---\n") if frontmatter_end >= 0 else 0
    while content_start < len(document) and document[content_start] == "\n":
        content_start += 1
    paragraph_end = document.find("\n\n", content_start)
    insertion = paragraph_end if paragraph_end >= 0 else len(document)
    prefix = document[:insertion].rstrip()
    suffix = document[insertion:].lstrip()
    return f"{prefix}\n\n{block}\n\n{suffix}".rstrip() + "\n"


def _manifest_document_path(manifest: RailManifest) -> Path:
    docs_url = manifest.metadata.docs_url
    if docs_url is None:
        raise ValueError(f"Built-in rail {manifest.name!r} does not declare metadata.docs_url.")
    if docs_url.startswith(("http://", "https://")):
        raise ValueError(f"Built-in rail {manifest.name!r} must reference a local catalog document.")
    path = (PROJECT_ROOT / docs_url).resolve()
    try:
        path.relative_to(CATALOG_ROOT)
    except ValueError as error:
        raise ValueError(f"Built-in rail {manifest.name!r} references a document outside the catalog.") from error
    return path


def _manifests_by_document(manifests: Iterable[RailManifest]) -> Dict[Path, tuple[RailManifest, ...]]:
    grouped = defaultdict(list)
    for manifest in manifests:
        grouped[_manifest_document_path(manifest)].append(manifest)
    return {
        path: tuple(sorted(group, key=lambda manifest: manifest.name))
        for path, group in sorted(grouped.items(), key=lambda item: str(item[0]))
    }


def expected_catalog_documents(manifests: Iterable[RailManifest]) -> Dict[Path, str]:
    documents = {}
    for path, document_manifests in _manifests_by_document(manifests).items():
        if not path.is_file():
            names = ", ".join(manifest.name for manifest in document_manifests)
            raise FileNotFoundError(f"Catalog document {path} for rails {names} does not exist.")
        documents[path] = update_page_requirements(path.read_text(encoding="utf-8"), document_manifests)
    return documents


def catalog_documents_are_current(manifests: Iterable[RailManifest]) -> bool:
    try:
        expected = expected_catalog_documents(manifests)
    except (FileNotFoundError, ValueError):
        return False
    return all(path.read_text(encoding="utf-8") == document for path, document in expected.items())


def write_catalog_documents(manifests: Iterable[RailManifest]) -> None:
    for path, document in expected_catalog_documents(manifests).items():
        path.write_text(document, encoding="utf-8")


def render_document(manifests: Iterable[RailManifest]) -> str:
    selected = tuple(sorted(manifests, key=lambda manifest: manifest.name))
    lines = [
        "---",
        "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
        "# SPDX-License-Identifier: Apache-2.0",
        'title: "Rail Installation Requirements"',
        'sidebar-title: "Installation Requirements"',
        'description: "Manifest-generated installation requirements for the built-in guardrail catalog."',
        "content:",
        '  type: "reference"',
        "---",
        "",
        "<!-- Generated by scripts/generate_rail_requirements_docs.py. Do not edit directly. -->",
        "",
        "This page is generated from the requirements declared by each built-in rail manifest. "
        "Install only the packages required by the rails you use. Package markers are preserved so `pip` can apply "
        "the requirement to compatible Python environments.",
        "",
        "To print requirements for a rail or a complete configuration, use:",
        "",
        "```bash",
        "nemoguardrails rails requirements --rail sensitive_data_detection",
        "nemoguardrails rails requirements --config ./config",
        "```",
        "",
        "The command prints installation guidance but never installs packages. Environment-variable values are never "
        "read or displayed. Models and services can require additional setup. Each linked catalog page repeats the "
        "requirements for that rail next to its configuration instructions.",
        "",
    ]
    for manifest in selected:
        lines.extend(_render_manifest(manifest))
    return "\n".join(lines).rstrip() + "\n"


def expected_document() -> str:
    return render_document(all_rail_manifests().values())


def write_document(path: Path = DOCUMENT_PATH) -> None:
    path.write_text(expected_document(), encoding="utf-8")


def document_is_current(path: Path = DOCUMENT_PATH) -> bool:
    return path.exists() and path.read_text(encoding="utf-8") == expected_document()


def documents_are_current() -> bool:
    manifests = tuple(all_rail_manifests().values())
    return document_is_current() and catalog_documents_are_current(manifests)


def write_documents() -> None:
    manifests = tuple(all_rail_manifests().values())
    DOCUMENT_PATH.write_text(render_document(manifests), encoding="utf-8")
    write_catalog_documents(manifests)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail when the generated document is stale.")
    args = parser.parse_args()
    if args.check:
        if not documents_are_current():
            raise SystemExit("Rail installation requirements are stale; run make generate-rail-requirements.")
        return
    write_documents()


if __name__ == "__main__":
    main()
