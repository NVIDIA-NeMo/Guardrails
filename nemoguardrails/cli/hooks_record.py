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

"""Commands for the hooks server's decision record, and for asking it about model calls from a file."""

import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import typer

from nemoguardrails.server.hooks import decision_record

app = typer.Typer(help="Show the hooks server's decision record, or check that its chain holds.")


@app.command("show")
def show(
    record: Path = typer.Argument(..., help="The decision record file (hooks_server.decision_record)."),
    run: Optional[str] = typer.Option(None, "--run", help="Only the decisions of this run id."),
    outcome: Optional[str] = typer.Option(None, "--outcome", help="Only this outcome: allow, deny, stop, quarantine."),
    kind: Optional[str] = typer.Option(None, "--kind", help="Only this kind: agent_start, tool_call, model_call."),
    agent: Optional[str] = typer.Option(None, "--agent", help="Only the decisions about this agent name."),
):
    """Print the recorded decisions, one JSON object per line, oldest first."""
    try:
        entries = decision_record.read(record)
    except decision_record.DecisionRecordError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    for entry in decision_record.select(entries, kind=kind, outcome=outcome, run_id=run, agent_name=agent):
        typer.echo(json.dumps(entry, sort_keys=True))


@app.command("check")
def check(
    record: Path = typer.Argument(..., help="The decision record file (hooks_server.decision_record)."),
    head: Optional[str] = typer.Option(None, "--head", help="The hash the last entry must have, kept elsewhere."),
):
    """Check that every entry is as written and chains to the one before. Exits 0 when the chain holds, 1 if not."""
    result = decision_record.check(record, head=head)
    typer.echo(json.dumps(result))
    raise typer.Exit(0 if result["intact"] else 1)


def decide(
    package: Path = typer.Argument(..., exists=True, file_okay=False, help="The agent package folder."),
    action: Path = typer.Option(..., "--action", exists=True, help="A JSON list of the calls to decide."),
    url: str = typer.Option("http://127.0.0.1:8100", "--url", help="The hooks server."),
):
    """Ask a running hooks server about each model call in a file, as the agent of PACKAGE would during its run.

    Each action is an object with `kind` `call_model`, `model`, and optionally `id` and `run` (the run id). The agent
    is named by the package's `agent.json` and its digest is the sha256 of `oadl.lock.json`, as NeMo Fabric names
    it. Prints one JSON object; exits 0 when every call is allowed, 2 when one is denied, 1 on an error."""
    agent_name = json.loads((package / "agent.json").read_text(encoding="utf-8"))["name"]
    agent_digest = "sha256:" + hashlib.sha256((package / "oadl.lock.json").read_bytes()).hexdigest()
    actions = json.loads(action.read_text(encoding="utf-8"))
    decisions, reasons = [], []
    for item in actions:
        if item.get("kind") != "call_model":
            decisions.append(
                {
                    "id": item.get("id"),
                    "subject": None,
                    "decision": "not_decided",
                    "reason": f"kind {item.get('kind')!r} is not a model call",
                    "rule": None,
                }
            )
            continue
        body = {
            "hook_event_name": "ModelCall",
            "agent_name": agent_name,
            "agent_digest": agent_digest,
            "model": item["model"],
            **({"id": item["id"]} if item.get("id") else {}),
        }
        headers = {
            "content-type": "application/json",
            "x-nemo-agent-name": agent_name,
            "x-nemo-agent-digest": agent_digest,
            **({"x-nemo-run-id": item["run"]} if item.get("run") else {}),
        }
        request = urllib.request.Request(f"{url.rstrip('/')}/v1/hooks/model-call", json.dumps(body).encode(), headers)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                answer = json.loads(response.read())
        except (urllib.error.URLError, ValueError) as exc:
            typer.secho(f"the hooks server at {url} did not answer: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(1)
        decisions.append(
            {
                "id": item.get("id"),
                "subject": item["model"],
                "decision": answer["decision"],
                "reason": answer.get("reason"),
                "rule": answer.get("rule"),
            }
        )
        if answer["decision"] != "allow":
            reasons.append(f"call_model {item['model']}: {answer.get('reason')}")
    denied = any(item["decision"] == "deny" for item in decisions)
    typer.echo(
        json.dumps(
            {
                "decision": "deny" if denied else "allow",
                "reasons": reasons,
                "agent_name": agent_name,
                "agent_digest": agent_digest,
                "decisions": decisions,
            },
            indent=2,
        )
    )
    raise typer.Exit(2 if denied else 0)
