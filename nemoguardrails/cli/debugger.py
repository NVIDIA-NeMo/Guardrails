# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import shlex
from typing import TYPE_CHECKING, Dict, Optional, cast

import typer
from rich.table import Table
from rich.tree import Tree

from nemoguardrails.colang.v2_x.lang.colang_ast import Spec, SpecOp, SpecType
from nemoguardrails.colang.v2_x.runtime.flows import (
    FlowConfig,
    FlowState,
    InteractionLoopType,
    State,
)
from nemoguardrails.colang.v2_x.runtime.runtime import RuntimeV2_x
from nemoguardrails.colang.v2_x.runtime.statemachine import is_active_flow
from nemoguardrails.utils import console

if TYPE_CHECKING:
    from nemoguardrails.cli.chat import ChatState

runtime: Optional[RuntimeV2_x] = None
state: Optional[State] = None
chat_state: Optional["ChatState"] = None

app = typer.Typer(name="!!!", no_args_is_help=True, add_completion=False)


def set_chat_state(_chat_state: "ChatState"):
    """Register the chat state that will be used by the debugger."""
    global chat_state
    chat_state = _chat_state


def set_runtime(_runtime: RuntimeV2_x):
    """Registers the runtime that will be used by the debugger."""
    global runtime
    runtime = _runtime


def set_output_state(_state: State):
    """Registers the state that will be used by the debugger."""
    global state
    state = _state


@app.command()
def restart():
    """Restart the current Colang script."""
    if chat_state is not None:
        chat_state.state = None
        chat_state.input_events = []
        chat_state.first_time = True


@app.command()
def pause():
    """Pause current interaction."""
    if chat_state is not None:
        chat_state.paused = True


@app.command()
def resume():
    """Pause current interaction."""
    if chat_state is not None:
        chat_state.paused = False


@app.command()
def flow(
    flow_name: str = typer.Argument(help="Name of flow or uid of a flow instance."),
):
    """Shows all details about a flow or flow instance."""
    assert state

    if state is not None:
        if flow_name in state.flow_configs:
            flow_config = state.flow_configs[flow_name]
            console.print(flow_config)
        else:
            matches = [(uid, item) for uid, item in state.flow_states.items() if flow_name in uid]
            if matches:
                flow_instance = matches[0][1]
                console.print(flow_instance.__dict__)
            else:
                console.print(f"Flow '{flow_name}' does not exist.")
                return
    else:
        console.print("No state available.")


@app.command()
def flows(
    all: bool = typer.Option(default=False, help="Show all flows (including inactive)."),
    order_by_name: bool = typer.Option(
        default=False,
        help="Order flows by flow name, otherwise its ordered by event processing priority.",
    ),
):
    """Shows a table with all (active) flows ordered in terms of there interaction loop priority and name."""
    if state is None:
        raise RuntimeError("No state available")

    table = Table(header_style="bold magenta")

    table.add_column("ID", style="dim", width=9)
    table.add_column("Flow Name")
    table.add_column("Loop (Priority | Type | Id)")
    table.add_column("Flow Instances")
    table.add_column("Source")

    def get_loop_info(flow_config: FlowConfig) -> str:
        if flow_config.loop_type == InteractionLoopType.NAMED:
            return (
                f"{flow_config.loop_priority} │ "
                + flow_config.loop_type.value.capitalize()
                + f" │ '{flow_config.loop_id}'"
            )
        else:
            return f"{flow_config.loop_priority} │ " + flow_config.loop_type.value

    rows = []
    for flow_id, flow_config in state.flow_configs.items():
        source = flow_config.source_file
        if source and "nemoguardrails" in source:
            source = source.rsplit("nemoguardrails", 1)[1]

        if not all:
            # Show only active flows
            active_instances = []
            if flow_id in state.flow_id_states:
                for flow_instance in state.flow_id_states[flow_id]:
                    if is_active_flow(flow_instance):
                        active_instances.append(flow_instance.uid.split(")")[1][:5])
                if active_instances:
                    rows.append(
                        [
                            flow_id,
                            get_loop_info(state.flow_configs[flow_id]),
                            ",".join(active_instances),
                            source,
                        ]
                    )
        else:
            instances = []
            if flow_id in state.flow_id_states:
                instances = [i.uid.split(")")[1][:5] for i in state.flow_id_states[flow_id]]
            rows.append(
                [
                    flow_id,
                    get_loop_info(state.flow_configs[flow_id]),
                    ",".join(instances),
                    source,
                ]
            )

    if order_by_name:
        rows.sort(key=lambda x: x[0])
    else:
        flow_configs: Dict[str, FlowConfig] = state.flow_configs
        rows.sort(key=lambda x: (-flow_configs[x[0]].loop_priority, x[0]))

    for i, row in enumerate(rows):
        table.add_row(f"{i + 1}", *row)

    console.print(table)


@app.command()
def tree(
    all: bool = typer.Option(
        default=False,
        help="Show all active flow instances (including inactive with `--all`).",
    ),
):
    """Lists the tree of all active flows."""
    if state is None or "main" not in state.flow_id_states:
        raise RuntimeError("No main flow available.")

    main_flow = state.flow_id_states["main"][0]

    root = Tree("main")
    queue = [[main_flow, root]]

    while queue:
        flow_state: FlowState
        node: Tree
        flow_state, node = queue.pop(0)
        flow_config = state.flow_configs[flow_state.flow_id]
        elements = flow_config.elements

        for child_uid in flow_state.child_flow_uids:
            child_flow_config = state.flow_configs[state.flow_states[child_uid].flow_id]
            child_flow_state = state.flow_states[child_uid]

            # Check if flow is inactive but parent instance of activate instances of same flow
            is_inactive_parent_instance: bool = False
            if not is_active_flow(child_flow_state):
                for child_instance_uid in child_flow_state.child_flow_uids:
                    child_instance_flow_state = state.flow_states[child_instance_uid]
                    if (
                        is_active_flow(child_instance_flow_state)
                        and child_instance_flow_state.flow_id == child_flow_state.flow_id
                    ):
                        is_inactive_parent_instance = True
                        break

            if not is_inactive_parent_instance and not all and not is_active_flow(child_flow_state):
                continue

            child_uid_short = child_uid.split(")")[1][0:3] + "..."
            parameter_values = ""
            for param in child_flow_config.parameters:
                value = child_flow_state.context[param.name]
                parameter_values += f" `{value}`"

            # We also want to figure out if the flow is actually waiting on this child
            waiting_on = False

            for _, head in flow_state.active_heads.items():
                head_element = elements[head.position]

                if isinstance(head_element, SpecOp):
                    head_element_spec_op = cast(SpecOp, head_element)
                    if head_element_spec_op.op == "match":
                        # Convert Spec to Spec object if it's a Dict
                        spec: Spec = (
                            head_element_spec_op.spec
                            if isinstance(head_element_spec_op.spec, Spec)
                            else Spec(**cast(Dict, head_element_spec_op.spec))
                        )

                        if spec.spec_type and spec.spec_type == SpecType.REFERENCE and spec.var_name:
                            var_name = spec.var_name
                            var = flow_state.context.get(var_name)

                            if var == child_flow_state:
                                waiting_on = True

            child_flow_label = (
                ("[green]>[/] " if waiting_on else "")
                + child_flow_state.flow_id
                + parameter_values
                + " ("
                + child_uid_short
                + " ,"
                + child_flow_state.status.value
                + ")"
            )

            if not is_active_flow(child_flow_state):
                child_flow_label = "[dim]" + child_flow_label + "[/]"
            if is_inactive_parent_instance:
                child_flow_label = "[" + child_flow_label + "]"

            child_node = node.add(child_flow_label)
            queue.append([child_flow_state, child_node])

    console.print(root)


def run_command(command: str):
    try:
        if command.strip() == "" or command == "help":
            command = "--help"

        app(shlex.split(command))
    except SystemExit:
        # Prevent stopping the app
        pass
