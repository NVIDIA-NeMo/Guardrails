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

import copy
import re
from typing import Dict, List, Optional, Tuple, Union

from nemoguardrails.colang.v2_x.lang.colang_ast import (
    Abort,
    Assignment,
    BeginScope,
    Break,
    CatchPatternFailure,
    Continue,
    Element,
    ElementType,
    EndScope,
    ForkHead,
    Goto,
    If,
    Label,
    MergeHeads,
    Spec,
    SpecOp,
    SpecType,
    WaitForHeads,
    When,
    While,
)
from nemoguardrails.colang.v2_x.runtime.errors import ColangSyntaxError
from nemoguardrails.colang.v2_x.runtime.flows import FlowConfig, InternalEvents
from nemoguardrails.colang.v2_x.runtime.utils import escape_special_string_characters
from nemoguardrails.utils import new_var_uuid


def expand_elements(
    elements: List[ElementType],
    flow_configs: Dict[str, FlowConfig],
    continue_break_labels: Optional[Tuple[str, str]] = None,
) -> List[ElementType]:
    """Iterates through all elements and expands/replaces them according to the rules."""
    elements_changed = True
    while elements_changed:
        elements_changed = False
        new_elements: List[ElementType] = []
        for element in elements:
            try:
                expanded_elements: List[ElementType] = []
                if isinstance(element, SpecOp):
                    if element.op == "send":
                        expanded_elements = _expand_send_element(element)
                    elif element.op == "match":
                        expanded_elements = _expand_match_element(element)
                    elif element.op == "start":
                        expanded_elements = _expand_start_element(element)
                    elif element.op == "stop":
                        expanded_elements = _expand_stop_element(element)
                    elif element.op == "activate":
                        expanded_elements = _expand_activate_element(element)
                    elif element.op == "deactivate":
                        expanded_elements = _expand_deactivate_element(element)
                    elif element.op == "await":
                        expanded_elements = _expand_await_element(element)
                elif isinstance(element, Assignment):
                    expanded_elements = _expand_assignment_stmt_element(element)
                elif isinstance(element, While):
                    expanded_elements = _expand_while_stmt_element(element, flow_configs)
                elif isinstance(element, If):
                    expanded_elements = _expand_if_element(element, flow_configs)
                    elements_changed = True  # Makes sure to update continue/break elements
                elif isinstance(element, When):
                    expanded_elements = _expand_when_stmt_element(element, flow_configs)
                    elements_changed = True  # Makes sure to update continue/break elements
                elif isinstance(element, Continue):
                    if element.label is None and continue_break_labels is not None:
                        element.label = continue_break_labels[0]
                elif isinstance(element, Break):
                    if element.label is None and continue_break_labels is not None:
                        element.label = continue_break_labels[1]

                if len(expanded_elements) > 0:
                    # Map new elements to source
                    for expanded_element in expanded_elements:
                        if isinstance(expanded_element, Element) and isinstance(element, Element):
                            expanded_element._source = element._source
                    # Add new elements
                    new_elements.extend(expanded_elements)
                    elements_changed = True
                else:
                    new_elements.extend([element])

            except Exception as e:
                error = "Error"
                if e.args[0]:
                    error = e.args[0]

                if hasattr(element, "_source") and element._source:
                    # TODO: Resolve source line to Colang file level
                    raise ColangSyntaxError(error + f" on source line {element._source.line}")
                else:
                    raise ColangSyntaxError(error)

        elements = new_elements
    return elements


def _expand_element_group(element: SpecOp) -> List[ElementType]:
    # TODO: Simplify for a single or group (we don't need head forking)
    new_elements: List[ElementType] = []

    normalized_group = normalize_element_groups(element.spec)

    if len(normalized_group["elements"]) == 1:
        # Only one and-group
        for and_group in normalized_group["elements"]:
            for group_element in and_group["elements"]:
                new_elements.append(
                    SpecOp(
                        op=element.op,
                        spec=group_element,
                    )
                )
    else:
        # Multiple and-groups
        fork_uid: str = new_var_uuid()
        fork_element = ForkHead(fork_uid=fork_uid)
        group_label_elements: List[Label] = []
        failure_label_name = f"failure_label_{new_var_uuid()}"
        failure_label_element = Label(name=failure_label_name)
        end_label_name = f"end_label_{new_var_uuid()}"
        goto_end_element = Goto(label=end_label_name)
        end_label_element = Label(name=end_label_name)

        for group_idx, and_group in enumerate(normalized_group["elements"]):
            group_label_name = f"group_{group_idx}_{new_var_uuid()}"
            group_label_elements.append(Label(name=group_label_name))
            fork_element.labels.append(group_label_name)

        # Generate new element sequence
        new_elements.append(CatchPatternFailure(label=failure_label_name))
        new_elements.append(fork_element)
        for group_idx, and_group in enumerate(normalized_group["elements"]):
            new_elements.append(group_label_elements[group_idx])
            for group_element in and_group["elements"]:
                new_elements.append(
                    SpecOp(
                        op=element.op,
                        spec=group_element,
                    )
                )
            new_elements.append(goto_end_element)
        new_elements.append(failure_label_element)
        new_elements.append(WaitForHeads(number=len(normalized_group["elements"])))
        new_elements.append(MergeHeads(fork_uid=fork_uid))
        new_elements.append(CatchPatternFailure(label=None))
        new_elements.append(Abort())
        new_elements.append(end_label_element)
        new_elements.append(MergeHeads(fork_uid=fork_uid))
        new_elements.append(CatchPatternFailure(label=None))

    return new_elements


def _expand_start_element(
    element: SpecOp,
) -> List[ElementType]:
    new_elements: List[ElementType] = []
    if isinstance(element.spec, Spec):
        # Single element
        if element.spec.spec_type == SpecType.FLOW and element.spec.members is None:
            # It's a flow
            # $_instance_<uid> = (<flow_id>)<uid>
            instance_uid_variable_name = f"_instance_uid_{new_var_uuid()}"
            new_elements.append(
                Assignment(
                    key=instance_uid_variable_name,
                    expression=f"'({element.spec.name}){{uid()}}'",
                )
            )
            # send StartFlow(flow_id=<flow_id>, flow_instance_uid=$_instance_<uid>)
            element.spec.arguments.update(
                {
                    "flow_id": f"'{element.spec.name}'",
                    "flow_instance_uid": f"'{{${instance_uid_variable_name}}}'",
                }
            )
            new_elements.append(
                SpecOp(
                    op="send",
                    spec=Spec(
                        name=InternalEvents.START_FLOW,
                        arguments=element.spec.arguments,
                        spec_type=SpecType.EVENT,
                    ),
                )
            )
            # match FlowStarted(...) as $_flow_event_ref
            flow_event_ref_uid = f"_flow_event_ref_{new_var_uuid()}"
            new_elements.append(
                SpecOp(
                    op="match",
                    spec=Spec(
                        name=InternalEvents.FLOW_STARTED,
                        arguments=element.spec.arguments,
                        ref=_create_ref_ast_dict_helper(flow_event_ref_uid),
                        spec_type=SpecType.EVENT,
                    ),
                    info={"internal": True},
                )
            )
            # $flow_ref = $_flow_event_ref.flow
            element_ref = element.spec.ref
            if element_ref is None:
                flow_ref_uid = f"_flow_ref_{new_var_uuid()}"
                element_ref = _create_ref_ast_dict_helper(flow_ref_uid)
            assert isinstance(element_ref, dict)
            new_elements.append(
                Assignment(
                    key=element_ref["elements"][0]["elements"][0].lstrip("$"),
                    expression=f"${flow_event_ref_uid}.flow",
                )
            )
        elif element.spec.spec_type == SpecType.ACTION:
            # It's an UMIM action
            element_ref = element.spec.ref
            if element_ref is None:
                action_event_ref_uid = f"_action_ref_{new_var_uuid()}"
                element_ref = _create_ref_ast_dict_helper(action_event_ref_uid)
                element.spec.ref = element_ref
            assert isinstance(element_ref, dict)
            new_elements.append(
                SpecOp(
                    op="_new_action_instance",
                    spec=element.spec,
                )
            )
            new_elements.append(
                SpecOp(
                    op="send",
                    spec=Spec(
                        name=element.spec.name,
                        arguments=element.spec.arguments,
                        members=_create_member_ast_dict_helper("Start", {}),
                        var_name=element_ref["elements"][0]["elements"][0].lstrip("$"),
                        spec_type=SpecType.EVENT,
                    ),
                )
            )
        else:
            raise ColangSyntaxError(f"'await' keyword cannot be used on '{element.spec.spec_type}'")
    else:
        # Element group
        new_elements = _expand_element_group(element)

    return new_elements


def _expand_stop_element(
    element: SpecOp,
) -> List[ElementType]:
    new_elements: List[ElementType] = []
    if isinstance(element.spec, Spec):
        # Single element
        raise NotImplementedError()
        # if (
        #     element.spec.spec_type == SpecType.REFERENCE
        #     and element.spec.members is None
        # ):
        #     # It's a reference to a flow or action
        #     new_elements.append(
        #         SpecOp(
        #             op="send",
        #             spec=Spec(
        #                 name=InternalEvents.STOP_FLOW,
        #                 arguments=element.spec.arguments,
        #                 spec_type=SpecType.EVENT,
        #             ),
        #         )
        #     )
        # else:
        #     raise ColangSyntaxError(
        #         f"'stop' keyword cannot yet be used on '{element.spec.spec_type}'"
        #     )
    else:
        # Element group
        new_elements = _expand_element_group(element)

    return new_elements


def _expand_send_element(
    element: SpecOp,
) -> List[ElementType]:
    new_elements: List[ElementType] = []
    if isinstance(element.spec, Spec):
        # Single send element
        if element.spec.spec_type != SpecType.EVENT and element.spec.members is None:
            raise ColangSyntaxError(f"Cannot send a non-event type: '{element.spec.spec_type}'")
    elif isinstance(element.spec, dict):
        # Element group
        new_elements = _expand_element_group(element)

    return new_elements


def _expand_match_element(
    element: SpecOp,
) -> List[ElementType]:
    new_elements: List[ElementType] = []
    if isinstance(element.spec, Spec):
        # Single match element
        if element.spec.spec_type == SpecType.FLOW and element.spec.members is None:
            # It's a flow
            raise ColangSyntaxError(f"Keyword `match` cannot be used with flows (flow `{element.spec.name}`)")
            # element_ref = element.spec.ref
            # if element_ref is None:
            #     element_ref = _create_ref_ast_dict_helper(
            #         f"_flow_event_ref_{new_var_uid()}"
            #     )
            # assert isinstance(element_ref, dict)

            # arguments = {"flow_id": f"'{element.spec.name}'"}
            # for arg in element.spec.arguments:
            #     arguments.update({arg: element.spec.arguments[arg]})k

            # new_elements.append(
            #     SpecOp(
            #         op="match",
            #         spec=Spec(
            #             name=InternalEvents.FLOW_FINISHED,
            #             arguments=arguments,
            #             ref=element_ref,
            #             spec_type=SpecType.EVENT,
            #         ),
            #         return_var_name=element.return_var_name,
            #     )
            # )
            # if element.return_var_name is not None:
            #     new_elements.append(
            #         Assignment(
            #             key=element.return_var_name,
            #             expression=f"${element_ref['elements'][0]['elements'][0]}.arguments.return_value",
            #         )
            #     )
        elif element.spec.spec_type == SpecType.EVENT or element.spec.members is not None:
            # It's an event
            if element.return_var_name is not None:
                element_ref = element.spec.ref
                if element_ref is None:
                    element_ref = _create_ref_ast_dict_helper(f"_event_ref_{new_var_uuid()}")
                assert isinstance(element_ref, dict)

                return_var_name = element.return_var_name

                element.spec.ref = element_ref
                element.return_var_name = None

                new_elements.append(element)
                new_elements.append(
                    Assignment(
                        key=return_var_name,
                        expression=f"${element_ref['elements'][0]['elements'][0]}.arguments.return_value",
                    )
                )
        else:
            raise ColangSyntaxError(f"Unsupported spec type: '{element.spec.spec_type}'")

    elif isinstance(element.spec, dict):
        # Element group
        normalized_group = normalize_element_groups(element.spec)

        if len(normalized_group["elements"]) == 1:
            # Single and-group
            if len(normalized_group["elements"][0]["elements"]) == 1:
                # and-group with a single element
                new_elements.append(
                    SpecOp(
                        op=element.op,
                        spec=normalized_group["elements"][0]["elements"][0],
                    )
                )
            else:
                # and-group with multiple elements
                fork_uid: str = new_var_uuid()
                fork_element = ForkHead(fork_uid=fork_uid)
                event_label_elements: List[Label] = []
                failure_label_name = f"failure_label_{new_var_uuid()}"
                failure_label_element = Label(name=failure_label_name)
                end_label_name = f"end_label_{new_var_uuid()}"
                goto_end_element = Goto(label=end_label_name)
                end_label_element = Label(name=end_label_name)

                and_group = normalized_group["elements"][0]
                for element_idx, _ in enumerate(and_group["elements"]):
                    element_label_name = f"event_{element_idx}_{new_var_uuid()}"
                    event_label_elements.append(Label(name=element_label_name))
                    fork_element.labels.append(element_label_name)

                new_elements.append(CatchPatternFailure(label=failure_label_name))
                new_elements.append(fork_element)

                for idx, element in enumerate(and_group["elements"]):
                    new_elements.append(event_label_elements[idx])
                    new_elements.append(
                        SpecOp(
                            op="match",
                            spec=element,
                        )
                    )
                    new_elements.append(goto_end_element)

                new_elements.append(failure_label_element)
                new_elements.append(MergeHeads(fork_uid=fork_uid))
                new_elements.append(CatchPatternFailure(label=None))
                new_elements.append(Abort())

                new_elements.append(end_label_element)
                new_elements.append(WaitForHeads(number=len(and_group["elements"])))
                new_elements.append(MergeHeads(fork_uid=fork_uid))
                new_elements.append(CatchPatternFailure(label=None))

        else:
            # Multiple and-groups combined by or
            fork_uid: str = new_var_uuid()
            fork_element = ForkHead(fork_uid=fork_uid)
            group_label_elements: List[Label] = []
            failure_label_name = f"failure_label_{new_var_uuid()}"
            failure_label_element = Label(name=failure_label_name)
            end_label_name = f"end_label_{new_var_uuid()}"
            goto_end_element = Goto(label=end_label_name)
            end_label_element = Label(name=end_label_name)

            or_group = normalized_group["elements"]
            for group_idx, _ in enumerate(or_group):
                group_label_name = f"group_{group_idx}_{new_var_uuid()}"
                group_label_elements.append(Label(name=group_label_name))
                fork_element.labels.append(group_label_name)

            new_elements.append(CatchPatternFailure(label=failure_label_name))
            new_elements.append(fork_element)

            for idx, group_element in enumerate(or_group):
                new_elements.append(group_label_elements[idx])
                new_elements.append(
                    SpecOp(
                        op="match",
                        spec=group_element,
                    )
                )
                new_elements.append(goto_end_element)

            new_elements.append(failure_label_element)
            new_elements.append(WaitForHeads(number=len(or_group)))
            new_elements.append(MergeHeads(fork_uid=fork_uid))
            new_elements.append(CatchPatternFailure(label=None))
            new_elements.append(Abort())

            new_elements.append(end_label_element)
            new_elements.append(MergeHeads(fork_uid=fork_uid))
            new_elements.append(CatchPatternFailure(label=None))

    else:
        raise ColangSyntaxError(f"Unknown element type '{type(element.spec)}'")

    return new_elements


def _expand_await_element(
    element: SpecOp,
) -> List[ElementType]:
    new_elements: List[ElementType] = []
    if isinstance(element.spec, Spec):
        # Single element
        if (
            element.spec.spec_type == SpecType.FLOW or element.spec.spec_type == SpecType.ACTION
        ) and element.spec.members is None:
            # It's a flow or an UMIM action
            element_ref = element.spec.ref
            if element_ref is None:
                element_ref = _create_ref_ast_dict_helper(f"_ref_{new_var_uuid()}")
            assert isinstance(element_ref, dict)

            element.spec.ref = element_ref
            new_elements.append(
                SpecOp(
                    op="start",
                    spec=element.spec,
                )
            )
            new_elements.append(
                SpecOp(
                    op="match",
                    spec=Spec(
                        var_name=element_ref["elements"][0]["elements"][0].lstrip("$"),
                        members=_create_member_ast_dict_helper("Finished", {}),
                        spec_type=SpecType.REFERENCE,
                    ),
                    return_var_name=element.return_var_name,
                )
            )
        else:
            raise ColangSyntaxError(f"Unsupported spec type '{type(element.spec)}', element '{element.spec.name}'")
    else:
        # Element group
        normalized_group = normalize_element_groups(element.spec)

        fork_uid: str = new_var_uuid()
        fork_element = ForkHead(fork_uid=fork_uid)
        group_label_elements: List[Label] = []
        scope_name = f"scope_{new_var_uuid()}"
        begin_scope_element = BeginScope(name=scope_name)
        end_scope_element = EndScope(name=scope_name)
        start_elements: List[List[SpecOp]] = []
        match_elements: List[List[Spec]] = []
        assignment_elements: List[List[Assignment]] = []
        failure_label_name = f"failure_label_{new_var_uuid()}"
        failure_label_element = Label(name=failure_label_name)
        end_label_name = f"end_label_{new_var_uuid()}"
        goto_end_element = Goto(label=end_label_name)
        end_label_element = Label(name=end_label_name)

        for group_idx, and_group in enumerate(normalized_group["elements"]):
            group_label_name = f"group_{group_idx}_{new_var_uuid()}"
            group_label_elements.append(Label(name=group_label_name))

            fork_element.labels.append(group_label_name)
            start_elements.append([])
            match_elements.append([])
            assignment_elements.append([])
            for group_element in and_group["elements"]:
                group_element_copy = copy.deepcopy(group_element)
                temp_element_ref = f"_ref_{new_var_uuid()}"

                group_element_copy.ref = _create_ref_ast_dict_helper(temp_element_ref)
                start_elements[-1].append(
                    SpecOp(
                        op="start",
                        spec=group_element_copy,
                    )
                )
                match_elements[-1].append(
                    Spec(
                        var_name=temp_element_ref,
                        members=_create_member_ast_dict_helper("Finished", {}),
                        spec_type=SpecType.REFERENCE,
                    )
                )
                if group_element.ref:
                    assignment_elements[-1].append(
                        Assignment(
                            key=group_element.ref["elements"][0]["elements"][0].lstrip("$"),
                            expression=f"${temp_element_ref}",
                        )
                    )

        # Generate new element sequence
        if len(normalized_group["elements"]) == 1:
            # Single and-group
            and_group = normalized_group["elements"][0]
            for idx, _ in enumerate(and_group["elements"]):
                new_elements.append(start_elements[0][idx])
            match_group = {"_type": "spec_and", "elements": match_elements[0]}
            new_elements.append(SpecOp(op="match", spec=match_group))
            for assignment_element in assignment_elements[0]:
                new_elements.append(assignment_element)

        else:
            # Multiple and-groups
            new_elements.append(begin_scope_element)
            new_elements.append(CatchPatternFailure(label=failure_label_name))
            new_elements.append(fork_element)
            for group_idx, and_group in enumerate(normalized_group["elements"]):
                new_elements.append(group_label_elements[group_idx])
                for idx, _ in enumerate(and_group["elements"]):
                    new_elements.append(start_elements[group_idx][idx])
                match_group = {
                    "_type": "spec_and",
                    "elements": match_elements[group_idx],
                }
                new_elements.append(SpecOp(op="match", spec=match_group))
                for assignment_element in assignment_elements[group_idx]:
                    new_elements.append(assignment_element)
                new_elements.append(goto_end_element)
            new_elements.append(failure_label_element)
            new_elements.append(WaitForHeads(number=len(normalized_group["elements"])))
            new_elements.append(CatchPatternFailure(label=None))
            new_elements.append(end_scope_element)
            new_elements.append(Abort())
            new_elements.append(end_label_element)
            new_elements.append(MergeHeads(fork_uid=fork_uid))
            new_elements.append(CatchPatternFailure(label=None))
            new_elements.append(end_scope_element)

    return new_elements


def _expand_activate_element(
    element: SpecOp,
) -> List[ElementType]:
    new_elements: List[ElementType] = []
    if isinstance(element.spec, Spec):
        # Single match element
        if element.spec.spec_type == SpecType.FLOW and element.spec.members is None:
            # It's a flow
            # $_instance_<uid> = (<flow_id>)<uid>
            instance_uid_variable_name = f"_instance_uid_{new_var_uuid()}"
            new_elements.append(
                Assignment(
                    key=instance_uid_variable_name,
                    expression=f"'({element.spec.name}){{uid()}}'",
                )
            )
            # send StartFlow(flow_id=<flow_id>, flow_instance_uid=$_instance_<uid>)
            match_arguments = dict(element.spec.arguments)
            match_arguments.update(
                {
                    "flow_id": f"'{element.spec.name}'",
                    "flow_instance_uid": f"'{{${instance_uid_variable_name}}}'",
                }
            )
            start_arguments = dict(match_arguments)
            start_arguments.update(
                {
                    "activated": "True",
                }
            )
            new_elements.append(
                SpecOp(
                    op="send",
                    spec=Spec(
                        name=InternalEvents.START_FLOW,
                        arguments=start_arguments,
                        spec_type=SpecType.EVENT,
                    ),
                )
            )
            # match FlowStarted(...)
            # flow_event_ref_uid = f"_flow_event_ref_{new_var_uid()}"
            new_elements.append(
                SpecOp(
                    op="match",
                    spec=Spec(
                        name=InternalEvents.FLOW_STARTED,
                        arguments=match_arguments,
                        # ref=_create_ref_ast_dict_helper(flow_event_ref_uid),
                        spec_type=SpecType.EVENT,
                    ),
                    info={"internal": True},
                )
            )
        else:
            # It's an UMIM event
            raise ColangSyntaxError(
                f"Only flows can be activated but not '{element.spec.spec_type}', element '{element.spec.name}'"
            )
    elif isinstance(element.spec, dict):
        # Multiple match elements
        normalized_group = normalize_element_groups(element.spec)
        if len(normalized_group["elements"]) > 1:
            raise NotImplementedError("Activating 'or' groups not supported yet!")
        for group_element in normalized_group["elements"][0]["elements"]:
            new_elements.append(
                SpecOp(
                    op="activate",
                    spec=group_element,
                )
            )

    return new_elements


def _expand_deactivate_element(
    element: SpecOp,
) -> List[ElementType]:
    new_elements: List[ElementType] = []
    if isinstance(element.spec, Spec):
        # Single element
        if element.spec.spec_type == SpecType.FLOW and element.spec.members is None:
            # It's a flow
            # send StopFlow(flow_id=<flow_id>)
            new_elements.append(
                SpecOp(
                    op="send",
                    spec=Spec(
                        name=InternalEvents.STOP_FLOW,
                        arguments={
                            "flow_id": f"'{element.spec.name}'",
                            "deactivate": True,
                            **element.spec.arguments,
                        },
                        spec_type=SpecType.EVENT,
                    ),
                )
            )
        else:
            # It's an UMIM event
            raise ColangSyntaxError(
                f"Only flows can be deactivated but not '{element.spec.spec_type}', element '{element.spec.name}'"
            )
    elif isinstance(element.spec, dict):
        # Multiple elements
        normalized_group = normalize_element_groups(element.spec)
        if len(normalized_group["elements"]) > 1:
            raise NotImplementedError("Deactivating 'or' groups not supported yet!")
        for group_element in normalized_group["elements"][0]["elements"]:
            new_elements.append(
                SpecOp(
                    op="deactivate",
                    spec=group_element,
                )
            )

    return new_elements


def _expand_assignment_stmt_element(element: Assignment) -> List[ElementType]:
    new_elements: List[ElementType] = []

    # Check if the expression is an NLD instruction
    nld_instruction_pattern = r'\.\.\.\s*("""|\'\'\')((?:\\\1|(?!\1)[\s\S])*?)\1|\.\.\.\s*("|\')((?:\\\3|(?!\3).)*?)\3'
    match = re.search(nld_instruction_pattern, element.expression)

    if match:
        # Replace the assignment with the GenerateValueAction system action
        instruction = escape_special_string_characters(match.group(2) or match.group(4))
        new_elements.append(
            SpecOp(
                op="await",
                spec=Spec(
                    name="GenerateValueAction",
                    spec_type=SpecType.ACTION,
                    arguments={
                        "var_name": f'"{element.key}"',
                        "instructions": f'"{instruction}"',
                    },
                ),
                return_var_name=element.key,
            )
        )

    return new_elements


def _expand_while_stmt_element(element: While, flow_configs: Dict[str, FlowConfig]) -> List[ElementType]:
    new_elements: List[ElementType] = []

    label_uid = new_var_uuid()
    begin_label = Label(name=f"_while_begin_{label_uid}")
    end_label = Label(name=f"_while_end_{label_uid}")
    goto_end = Goto(
        label=end_label.name,
        expression=f"not ({element.expression})",
    )
    goto_begin = Goto(
        label=begin_label.name,
        expression="True",
    )
    body_elements = expand_elements(element.elements, flow_configs, (begin_label.name, end_label.name))

    new_elements = [begin_label, goto_end]
    new_elements.extend(body_elements)
    new_elements.extend([goto_begin, end_label])

    return new_elements


def _expand_if_element(element: If, flow_configs: Dict[str, FlowConfig]) -> List[ElementType]:
    elements: List[ElementType] = []

    if_else_body_label_name = f"if_else_body_label_{new_var_uuid()}"
    if_end_label_name = f"if_end_label_{new_var_uuid()}"

    # TODO: optimize for cases when the else section is missing
    elements.append(
        Goto(
            expression=f"not({element.expression})",
            label=(if_end_label_name if not element.else_elements else if_else_body_label_name),
        )
    )
    elements.extend(expand_elements(element.then_elements, flow_configs))

    if element.else_elements:
        elements.append(Goto(label=if_end_label_name))
        elements.append(Label(name=if_else_body_label_name))
        elements.extend(expand_elements(element.else_elements, flow_configs))

    elements.append(Label(name=if_end_label_name))

    return elements


def _expand_when_stmt_element(element: When, flow_configs: Dict[str, FlowConfig]) -> List[ElementType]:
    stmt_uid = new_var_uuid()

    init_case_label_names: List[str] = []
    cases_fork_uid: str = new_var_uuid()
    cases_fork_head_element = ForkHead(fork_uid=cases_fork_uid)
    groups_fork_head_elements: List[ForkHead] = []
    failure_case_label_names: List[str] = []
    scope_label_name = f"scope_{stmt_uid}"
    group_label_names: List[List[str]] = []
    group_start_elements: List[List[List[Spec]]] = []
    group_match_elements: List[List[List[Spec]]] = []
    group_assignment_elements: List[List[List[Assignment]]] = []
    case_label_names: List[str] = []
    else_label_name = f"when_else_label_{stmt_uid}"
    else_statement_label_name = f"when_else_statement_label_{stmt_uid}"
    end_label_name = f"when_end_label_{stmt_uid}"

    for case_idx, case_element in enumerate(element.when_specs):
        case_uid = str(chr(ord("a") + case_idx))
        init_case_label_names.append(f"init_case_{case_uid}_label_{stmt_uid}")
        cases_fork_head_element.labels.append(init_case_label_names[case_idx])
        failure_case_label_names.append(f"failure_case_{case_uid}_label_{stmt_uid}")
        case_label_names.append(f"case_{case_uid}_label_{stmt_uid}")
        groups_fork_head_elements.append(ForkHead(fork_uid=new_var_uuid()))

        case_element_dict: dict
        if isinstance(case_element, Spec):
            # Single element
            case_element_dict = {
                "_type": "spec_and",
                "elements": [case_element],
            }
        elif isinstance(case_element, dict):
            case_element_dict = case_element
        else:
            raise ColangSyntaxError(f"Unexpected type: '{type(case_element)}'")

        normalized_group = normalize_element_groups(case_element_dict)

        group_label_names.append([])
        group_start_elements.append([])
        group_match_elements.append([])
        group_assignment_elements.append([])
        for group_idx, and_group in enumerate(normalized_group["elements"]):
            group_label_names[case_idx].append(f"group_{case_uid}_{group_idx}_label_{stmt_uid}")
            groups_fork_head_elements[case_idx].labels.append(group_label_names[case_idx][group_idx])

            group_start_elements[case_idx].append([])
            group_match_elements[case_idx].append([])
            group_assignment_elements[case_idx].append([])
            for group_element in and_group["elements"]:
                match_element = copy.deepcopy(group_element)
                ref_uid = None
                temp_ref_uid: str
                if (
                    group_element.spec_type == SpecType.FLOW or group_element.spec_type == SpecType.ACTION
                ) and group_element.members is None:
                    # Add start element
                    temp_ref_uid = f"_ref_{new_var_uuid()}"
                    if group_element.ref is not None:
                        ref_uid = group_element.ref["elements"][0]["elements"][0].lstrip("$")
                    group_element.ref = _create_ref_ast_dict_helper(temp_ref_uid)
                    group_start_elements[case_idx][group_idx].append(group_element)

                    match_element.name = None
                    match_element.var_name = temp_ref_uid
                    match_element.members = _create_member_ast_dict_helper("Finished", {})
                    match_element.ref = None
                    match_element.spec_type = SpecType.REFERENCE

                    # Add assignment element
                    if ref_uid:
                        assignment_element = Assignment(
                            key=ref_uid,
                            expression=f"${temp_ref_uid}",
                        )
                        group_assignment_elements[case_idx][group_idx].append(assignment_element)

                # Add match element
                group_match_elements[case_idx][group_idx].append(match_element)

    new_elements: List[ElementType] = []
    new_elements.append(BeginScope(name=scope_label_name))
    new_elements.append(cases_fork_head_element)
    for case_idx, case_element in enumerate(element.when_specs):
        # Case init groups
        new_elements.append(Label(name=init_case_label_names[case_idx]))
        new_elements.append(CatchPatternFailure(label=failure_case_label_names[case_idx]))
        new_elements.append(groups_fork_head_elements[case_idx])

        # And-group element groups
        for group_idx, group_label_name in enumerate(group_label_names[case_idx]):
            new_elements.append(Label(name=group_label_name))

            if group_start_elements[case_idx][group_idx]:
                new_elements.append(
                    SpecOp(
                        op="start",
                        spec={
                            "_type": "spec_and",
                            "elements": group_start_elements[case_idx][group_idx],
                        },
                    )
                    # TODO: Replace above with this once refactored
                    # SpecOp(
                    #     op="start",
                    #     spec=SpecAnd(
                    #         elements=group_start_elements[case_idx][group_idx]
                    #     ),
                    # )
                )
            new_elements.append(
                SpecOp(
                    op="match",
                    spec={
                        "_type": "spec_and",
                        "elements": group_match_elements[case_idx][group_idx],
                    },
                )
                # TODO: Replace above with this once refactored
                # SpecOp(
                #     op="match",
                #     spec=SpecAnd(elements=group_match_elements[case_idx][group_idx]),
                # )
            )

            if group_start_elements[case_idx][group_idx]:
                for assignment_element in group_assignment_elements[case_idx][group_idx]:
                    new_elements.append(assignment_element)

            new_elements.append(Goto(label=case_label_names[case_idx]))

            # Case groups
            new_elements.append(Label(name=case_label_names[case_idx]))
            new_elements.append(MergeHeads(fork_uid=cases_fork_uid))
            new_elements.append(CatchPatternFailure(label=None))
            new_elements.append(EndScope(name=scope_label_name))
            new_elements.extend(expand_elements(element.then_elements[case_idx], flow_configs))
            new_elements.append(Goto(label=end_label_name))

            # Failure case groups
            new_elements.append(Label(name=failure_case_label_names[case_idx]))
            new_elements.append(WaitForHeads(number=len(group_label_names[case_idx])))
            new_elements.append(CatchPatternFailure(label=None))
            new_elements.append(Goto(label=else_label_name))

        # Else group
        new_elements.append(Label(name=else_label_name))
        new_elements.append(WaitForHeads(number=len(group_label_names)))
        if element.else_elements is None:
            new_elements.append(Abort())
        else:
            new_elements.append(Goto(label=else_statement_label_name))

            new_elements.append(Label(name=else_statement_label_name))
            new_elements.extend(expand_elements(element.else_elements, flow_configs))

        # End label
        new_elements.append(Label(name=end_label_name))

    return new_elements


def normalize_element_groups(group: Union[Spec, dict]) -> dict:
    """
    Normalize groups to the disjunctive normal form (DNF),
    resulting in a single or group that contains multiple and groups.
    """

    if isinstance(group, Spec):
        group = {"_type": "spec_and", "elements": [group]}

    if group["_type"] == "spec_or":
        return flatten_or_group(
            {
                "_type": "spec_or",
                "elements": [
                    (
                        normalize_element_groups(elem)
                        if isinstance(elem, dict)
                        else {"_type": "spec_and", "elements": [elem]}
                    )
                    for elem in group["elements"]
                ],
            }
        )
    elif group["_type"] == "spec_and":
        results = [{"_type": "spec_and", "elements": []}]
        for elem in group["elements"]:
            normalized = (
                normalize_element_groups(elem)
                if isinstance(elem, dict)
                else {
                    "_type": "spec_or",
                    "elements": [{"_type": "spec_and", "elements": [elem]}],
                }
            )

            # Distribute using the property: A and (B or C) = (A and B) or (A and C)
            new_results = []
            for res_elem in results:
                for norm_elem in normalized["elements"]:
                    new_elem = {
                        "_type": "spec_and",
                        "elements": res_elem["elements"] + norm_elem["elements"],
                    }
                    new_results.append(new_elem)
            results = new_results

        # Remove duplicate elements from groups
        # for idx, and_group in enumerate(results):
        #     results[idx] = uniquify_element_group(and_group)

        # TODO: Remove duplicated and groups
        return flatten_or_group({"_type": "spec_or", "elements": results})

    return {}


def flatten_or_group(group: dict):
    """Flattens a group that has multiple or levels to a single one."""
    new_elements = []
    for elem in group["elements"]:
        if isinstance(elem, dict) and elem["_type"] == "spec_or":
            new_elements.extend(elem["elements"])
        else:
            new_elements.append(elem)
    return {"_type": "spec_or", "elements": new_elements}


def _create_ref_ast_dict_helper(ref_name: str) -> dict:
    return {
        "_type": "capture_ref",
        "elements": [{"_type": "var_name", "elements": [f"{ref_name}"]}],
    }


def _create_member_ast_dict_helper(name: str, arguments: dict) -> list:
    return [
        {
            "_type": "spec",
            "_source": None,
            "name": name,
            "arguments": arguments,
            "members": None,
            "var_name": None,
        }
    ]


# def uniquify_element_group(group: dict) -> dict:
#     """Remove all duplicate elements from group."""
#     unique_elements: Dict[Tuple[int, Spec]] = {}
#     for element in group["elements"]:
#         unique_elements.setdefault(element.hash(), element)
#     new_group = group.copy()
#     new_group["elements"] = [e for e in unique_elements.values()]
#     return new_group


# def convert_to_single_and_element_group(group: dict) -> dict:
#     """Convert element group into a single 'and' group with unique elements."""
#     unique_elements: Dict[Tuple[int, Spec]] = {}
#     for and_group in group["elements"]:
#         for element in and_group["elements"]:
#             # Makes sure that we add the same element only once
#             unique_elements.update({element.hash(): element})
#     return {
#         "_type": "spec_or",
#         "elements": [
#             {
#                 "_type": "spec_and",
#                 "elements": [elem for elem in unique_elements.values()],
#             }
#         ],
#     }
