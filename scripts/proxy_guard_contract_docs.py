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

"""Render bound guard contracts as navigable Fern reference pages."""

from __future__ import annotations

import html
import json
from collections.abc import Iterator

import yaml

EXTENSION = "x-nemo-guardrails"
FIELD_POLICY_GUIDE = "/run-guardrailed-inference/using-proxy/guard-contracts#read-the-field-policy"


def policy_badge(schema: dict) -> str:
    """Render field classification separately from presence and nullability."""
    policy = schema.get(EXTENSION, {})
    classification = "disabled" if policy.get("gate") == "disabled" else policy.get("classification")
    intents = {"guarded": "success", "constrained": "warning", "opaque": "note", "disabled": "danger"}
    if classification not in intents:
        return ""
    classes = f"guardrails-proxy-policy guardrails-proxy-policy-{classification}"
    if schema.get("type") == "null":
        classes += " guardrails-proxy-null-only"
    return (
        f'<span className="{classes}"><a href="{FIELD_POLICY_GUIDE}">'
        f'<Badge intent="{intents[classification]}" minimal>{classification.title()}</Badge></a></span>'
    )


def _code(value: str) -> str:
    return "`" + value.replace("|", "\\|").replace("`", "&#96;") + "`"


def _json(value: object) -> str:
    return _code(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _pointer_part(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _schema_nodes(schema: dict, path: str, pointer: str) -> Iterator[tuple[str, str, dict]]:
    yield path, pointer, schema
    for name, child in schema.get("properties", {}).items():
        yield from _schema_nodes(child, f"{path}.{name}", f"{pointer}/properties/{_pointer_part(name)}")
    if "items" in schema:
        yield from _schema_nodes(schema["items"], path + "[]", pointer + "/items")
    for keyword in ("oneOf", "anyOf", "allOf"):
        for index, child in enumerate(schema.get(keyword, [])):
            yield from _schema_nodes(child, f"{path} ({keyword} branch {index + 1})", f"{pointer}/{keyword}/{index}")


def _type_label(schema: dict) -> str:
    if "type" in schema:
        return schema["type"]
    for keyword in ("oneOf", "anyOf", "allOf"):
        if keyword in schema:
            alternatives = schema[keyword]
            if keyword in {"oneOf", "anyOf"} and len(alternatives) == 2 and {"type": "null"} in alternatives:
                return " or ".join(_type_label(child) for child in alternatives)
            return f"{keyword}: {len(alternatives)} alternatives"
    return "any"


def _readable_rules(schema: dict) -> str:
    """Summarize common assertions; the exact export retains all annotations."""
    policy = schema.get(EXTENSION, {})
    if policy.get("gate") == "disabled":
        return "Omit or null."
    if policy.get("classification") == "opaque":
        return "Provider requirements apply."
    for keyword in ("oneOf", "anyOf", "allOf"):
        if keyword in schema:
            branches = [child for child in schema[keyword] if child != {"type": "null"}]
            if keyword in {"oneOf", "anyOf"} and len(branches) == 1 and len(schema[keyword]) == 2:
                rule = _readable_rules(branches[0])
                return "If non-null, " + rule[0].lower() + rule[1:]
            return "See branch rules below."
    rules = []
    if "const" in schema:
        rules.append(f"Must be {_json(schema['const'])}.")
    if "enum" in schema:
        rules.append("Allowed: " + ", ".join(_json(value) for value in schema["enum"]) + ".")
    for minimum, maximum, unit in (("minItems", "maxItems", "items"), ("minLength", "maxLength", "characters")):
        low, high = schema.get(minimum), schema.get(maximum)
        if low is not None and low == high:
            rules.append(f"Exactly {low} {unit[:-1] if low == 1 else unit}.")
        else:
            if low is not None:
                rules.append(f"At least {low} {unit[:-1] if low == 1 else unit}.")
            if high is not None:
                rules.append(f"At most {high} {unit[:-1] if high == 1 else unit}.")
    low, high = schema.get("minimum"), schema.get("maximum")
    if low is not None and low == high:
        rules.append(f"Must be {_json(low)}.")
    else:
        if low is not None:
            rules.append(f"Minimum {_json(low)}.")
        if high is not None:
            rules.append(f"Maximum {_json(high)}.")
    if "pattern" in schema:
        rules.append(f"Pattern: {_code(schema['pattern'])}.")
    if "properties" in schema or "items" in schema:
        rules.append("See nested fields.")
    if not rules and policy.get("subject"):
        rules.append("Text selected for checks.")
    return " ".join(rules) or "See exact contract."


def _object_table(schema: dict, pointer: str) -> str:
    lines = [
        '<div className="guardrails-proxy-contract-table">',
        "",
        "| Field | Type | Presence | Policy | Rules |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, child in schema.get("properties", {}).items():
        field_pointer = f"{pointer}/properties/{_pointer_part(name)}"
        label = html.escape(name).replace("_", "_<wbr />")
        field = f'<code title="{html.escape(field_pointer, quote=True)}">{label}</code>'
        presence = "Required" if name in schema.get("required", []) else "Optional"
        lines.append(
            f"| {field} | {_code(_type_label(child))} | {presence} | {policy_badge(child)} | {_readable_rules(child)} |"
        )
    lines.extend(["", "</div>"])
    policy = schema.get(EXTENSION, {})
    if policy:
        lines.extend(
            [
                "",
                '<Accordion title="Object policy">',
                "",
                "```yaml",
                yaml.safe_dump(policy, sort_keys=True).rstrip(),
                "```",
                "",
                "</Accordion>",
            ]
        )
    return "\n".join(lines)


def _payload_section(schema: dict, section: str) -> str:
    lines = [f"## {section.title()}", ""]
    for path, pointer, node in _schema_nodes(schema, "$", f"#/{section}"):
        if "properties" not in node:
            continue
        table = _object_table(node, pointer)
        if path == "$":
            lines.append(table)
        else:
            lines.extend([f'<Accordion title="{html.escape(path, quote=True)}">', "", table, "", "</Accordion>"])
        lines.append("")
    return "\n".join(lines)


def _stream_section(stream: dict | None) -> str:
    lines = ["## Stream", ""]
    if stream is None:
        return "\n".join([*lines, "This binding has no stream declaration.", ""])
    lines.extend(
        [
            "Event roles and missing-text rules come from the bound classifier. "
            "See [stream declaration meanings](/run-guardrailed-inference/using-proxy/guard-contracts#read-stream-declarations) "
            "and [runtime completion and errors](/run-guardrailed-inference/using-proxy/providers-and-streaming#completion-and-errors).",
            "",
            "| Shape | Role | Text subjects | Missing text | Selection |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for index, schema in enumerate(stream.get("oneOf", [])):
        event = schema[EXTENSION]["event"]
        subjects = [
            _code(path)
            for path, _, node in _schema_nodes(schema, "$", f"#/stream/oneOf/{index}")
            if node.get(EXTENSION, {}).get("subject", {}).get("kind") == "text"
        ]
        for variant in event["variants"]:
            selection = {key: value for key, value in variant.items() if key != "shape"}
            missing_text = {key: value for key, value in event.items() if key.startswith("missing_text")}
            lines.append(
                f"| {_code(variant['shape'])} | {_code(event['classification'])} | "
                f"{', '.join(subjects) or '—'} | {_json(missing_text) if missing_text else '—'} | {_json(selection)} |"
            )
    lines.extend(
        [
            "",
            "### Transport declarations",
            "",
            "```yaml",
            yaml.safe_dump(stream.get(EXTENSION, {}), sort_keys=True).rstrip(),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def render_contract_page(contract: dict, slug: str) -> str:
    """Present policy scopes and include the unmodified export for exact inspection."""
    endpoint = contract["integration"]["endpoint"]
    title = endpoint["operation_label"] + " Guard Contract"
    header = yaml.safe_dump(
        {
            "title": title,
            "description": "Request, response, and stream policies exported from the bound native OpenAI operation.",
        },
        sort_keys=False,
    ).rstrip()
    lines = [
        "---",
        header,
        "---",
        "",
        "{/* Generated by scripts/export_proxy_openapi.py. Do not edit directly. */}",
        "",
        f"[API reference](/reference/proxy-api/{slug}) · [Field-policy meanings]({FIELD_POLICY_GUIDE})",
        "",
        f"Operation: {_code(contract['operationId'])}. Payload profile: {_code(contract['profile'])}. "
        f"Contract format: {_code(contract['version'])}.",
        "",
        "[Request](#request) · [Response](#response) · [Stream](#stream) · [Exact contract](#exact-contract)",
        "",
        "These declarations describe the native binding's guard boundary, not a complete provider schema. "
        "Rails configuration selects the checks that run. See [supported content]"
        "(/run-guardrailed-inference/using-proxy/providers-and-streaming#coverage-of-the-shipped-profiles).",
        "",
        "Response and stream inspection declarations apply when output inspection is enabled. "
        "Guarded request validation still applies when checks are disabled. See [disabled checks and pass-through]"
        "(/run-guardrailed-inference/using-proxy/providers-and-streaming#disabled-checks-and-pass-through).",
        "",
        "Required presence is local to each containing object and schema branch. "
        "Nested accordions preserve those scopes; branch labels identify schema alternatives, not payload members. "
        "Hover a field name for its exact schema pointer. Defaults are not inserted into forwarded traffic.",
        "",
        _payload_section(contract["request"], "request"),
        _payload_section(contract["response"], "response"),
        _stream_section(contract.get("stream")),
        "## Exact contract",
        "",
        "This is the unmodified export from the bound Python endpoint. "
        "Stateful stream hooks, enabled Rails checks, and arbitrary Python validation are not serialized.",
        "",
        '<Accordion title="Exact exported contract">',
        "",
        "```yaml",
        yaml.safe_dump(contract, sort_keys=True, allow_unicode=True, width=100).rstrip(),
        "```",
        "",
        "</Accordion>",
        "",
    ]
    return "\n".join(lines)
