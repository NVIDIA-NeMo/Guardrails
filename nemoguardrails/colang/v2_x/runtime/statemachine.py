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
import logging
import random
import re
from collections import deque
from datetime import datetime, timedelta
from functools import partial
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

from nemoguardrails.colang.v2_x.lang.colang_ast import (
    Abort,
    Assignment,
    BeginScope,
    Break,
    CatchPatternFailure,
    Continue,
    ElementType,
    EndScope,
    ForkHead,
    Global,
    Goto,
    Label,
    Log,
    MergeHeads,
    Print,
    Priority,
    Return,
    Spec,
    SpecOp,
    SpecType,
    WaitForHeads,
)
from nemoguardrails.colang.v2_x.lang.expansion import expand_elements
from nemoguardrails.colang.v2_x.runtime.errors import (
    ColangRuntimeError,
    ColangSyntaxError,
    ColangValueError,
)
from nemoguardrails.colang.v2_x.runtime.eval import (
    ComparisonExpression,
    eval_expression,
)
from nemoguardrails.colang.v2_x.runtime.flows import (
    Action,
    ActionEvent,
    ActionStatus,
    Event,
    FlowConfig,
    FlowHead,
    FlowHeadStatus,
    FlowState,
    FlowStatus,
    InteractionLoopType,
    InternalEvent,
    InternalEvents,
    State,
)
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.utils import console, new_event_dict, new_readable_uuid, new_uuid

log = logging.getLogger(__name__)


def initialize_state(state: State) -> None:
    """
    Initialize the state to make it ready for the story start.
    """

    state.internal_events = deque()

    assert "main" in state.flow_configs, "No main flow found!"

    state.flow_states = dict()

    try:
        # TODO: Think about where to put this
        for flow_config in state.flow_configs.values():
            initialize_flow(state, flow_config)
    except Exception as e:
        if e.args[0]:
            raise ColangSyntaxError(e.args[0] + f" in flow `{flow_config.id}` ({flow_config.source_file})")
        else:
            raise ColangSyntaxError() from e

    # Create main flow state first
    main_flow_config = state.flow_configs["main"]
    main_flow = add_new_flow_instance(
        state,
        create_flow_instance(main_flow_config, new_readable_uuid("main"), "0", {}),
    )
    main_flow.activated = 1
    if main_flow_config.loop_id is None:
        main_flow.loop_id = new_readable_uuid("main")
    else:
        main_flow.loop_id = main_flow_config.loop_id
    state.main_flow_state = main_flow


def initialize_flow(state: State, flow_config: FlowConfig) -> None:
    """Initialize a flow before it can be used and instantiated."""
    # Transform and resolve flow configuration element notation (actions, flows, ...)
    flow_config.elements = expand_elements(flow_config.elements, state.flow_configs)

    # Extract all the label elements
    for idx, element in enumerate(flow_config.elements):
        if isinstance(element, Label):
            flow_config.element_labels.update({element["name"]: idx})


def create_flow_instance(
    flow_config: FlowConfig,
    flow_instance_uid: str,
    flow_hierarchy_position: str,
    event_arguments: Dict[str, Any],
) -> FlowState:
    """Create a new flow instance that can be added."""
    loop_uid: Optional[str] = None
    if flow_config.loop_type == InteractionLoopType.NEW:
        loop_uid = new_uuid()
    elif flow_config.loop_type == InteractionLoopType.NAMED:
        assert flow_config.loop_id is not None
        loop_uid = flow_config.loop_id
    # For type InteractionLoopType.PARENT we keep it None to infer loop_id at run_time from parent

    head_uid = new_uuid()
    flow_state = FlowState(
        uid=flow_instance_uid,
        flow_id=flow_config.id,
        loop_id=loop_uid,
        hierarchy_position=flow_hierarchy_position,
        heads={
            head_uid: FlowHead(
                uid=head_uid,
                flow_state_uid=flow_instance_uid,
                matching_scores=[],
            )
        },
    )

    if "context" in event_arguments:
        if flow_config.parameters:
            raise ColangRuntimeError(f"Context cannot be shared to flows with parameters: '{flow_config.id}'")
        # Replace local context with context from parent flow (shared flow context)
        flow_state.context = event_arguments["context"]

    # Add all the flow parameters
    for idx, param in enumerate(flow_config.parameters):
        if param.name in event_arguments:
            val = event_arguments[param.name]
        else:
            val = eval_expression(param.default_value_expr, {}) if param.default_value_expr else None
        flow_state.arguments[param.name] = val
        flow_state.context.update(
            {
                param.name: val,
            }
        )

    # Add the positional flow parameter identifiers
    for idx, param in enumerate(flow_config.parameters):
        positional_param = f"${idx}"
        if positional_param in event_arguments:
            val = event_arguments[positional_param]
            flow_state.arguments[param.name] = val
            flow_state.arguments[positional_param] = val

    # Add all flow return members
    for idx, member in enumerate(flow_config.return_members):
        flow_state.context.update(
            {
                member.name: (eval_expression(member.default_value_expr, {}) if member.default_value_expr else None),
            }
        )

    return flow_state


def add_new_flow_instance(state: State, flow_state: FlowState) -> FlowState:
    """Add a new flow instance to the current state."""
    # Update state structures
    state.flow_states.update({flow_state.uid: flow_state})
    if flow_state.flow_id in state.flow_id_states:
        state.flow_id_states[flow_state.flow_id].append(flow_state)
    else:
        state.flow_id_states.update({flow_state.flow_id: [flow_state]})

    flow_head = next(iter(flow_state.heads.values()))
    flow_head.position_changed_callback = partial(_flow_head_changed, state, flow_state)
    flow_head.status_changed_callback = partial(_flow_head_changed, state, flow_state)
    _flow_head_changed(state, flow_state, flow_head)

    return flow_state


def _create_event_reference(state: State, flow_state: FlowState, element: SpecOp, event: Event) -> dict:
    assert isinstance(element.spec, Spec) and element.spec.ref and isinstance(element.spec.ref, dict)
    reference_name = element.spec.ref["elements"][0]["elements"][0].lstrip("$")
    new_event = get_event_from_element(state, flow_state, element)
    new_event.arguments.update(event.arguments)

    if isinstance(new_event, InternalEvent):
        flow_state_uid = event.arguments.get("source_flow_instance_uid", None)
        if flow_state_uid is not None:
            new_event.flow = state.flow_states[flow_state_uid]
    elif isinstance(new_event, ActionEvent):
        event = cast(ActionEvent, event)
        new_event.action_uid = event.action_uid
        if event.action_uid is not None:
            if event.action_uid not in state.actions:
                user_action = Action.from_event(event)
                assert user_action is not None
                state.actions.update({event.action_uid: user_action})
                new_event.action = user_action
            else:
                new_event.action = state.actions[event.action_uid]
    return {reference_name: new_event}


def _context_log(flow_state: FlowState) -> str:
    return str(
        [
            {key: value}
            for key, value in flow_state.context.items()
            if not isinstance(value, InternalEvent) and not isinstance(value, FlowState)
        ]
    )


def run_to_completion(state: State, external_event: Union[dict, Event]) -> State:
    """
    Compute the next state of the flow-driven system.
    """
    log.info("[bold violet]-> External Event[/]: %s", external_event)

    # Convert to event type
    converted_external_event: Event
    if isinstance(external_event, dict):
        if "Action" in external_event["type"]:
            converted_external_event = ActionEvent.from_umim_event(external_event)
        else:
            converted_external_event = Event.from_umim_event(external_event)
    elif isinstance(external_event, Event):
        converted_external_event = external_event

    # Initialize the new state
    state.internal_events = deque([converted_external_event])
    state.outgoing_events.clear()

    _clean_up_state(state)

    actionable_heads: List[FlowHead] = []
    merging_heads: List[FlowHead] = []

    # Main processing loop
    heads_are_advancing = True
    heads_are_merging = True
    while heads_are_advancing:
        while heads_are_merging:
            while state.internal_events:
                event = state.internal_events.popleft()
                log.info("Process internal event: %s", event)

                # Find all active interaction loops
                active_interaction_loops = set()
                for flow_state in state.flow_states.values():
                    if is_listening_flow(flow_state):
                        active_interaction_loops.add(flow_state.loop_id)

                # TODO: Check if we should rather should do this after the event matching step
                # or even skip the event processing
                if event.name == "ContextUpdate":
                    # Update the context
                    if "data" in event.arguments and isinstance(event.arguments, dict):
                        state.context.update(event.arguments["data"])

                handled_event_loops = _process_internal_events_without_default_matchers(state, event)

                head_candidates = _get_all_head_candidates(state, event)

                heads_matching: List[FlowHead] = []
                heads_not_matching: List[FlowHead] = []
                heads_failing: List[FlowHead] = []

                # Iterate over all potential head candidates and check if we have an event match
                for flow_state_uid, head_uid in head_candidates:
                    flow_state = state.flow_states[flow_state_uid]
                    head = flow_state.heads[head_uid]
                    element = get_element_from_head(state, head)
                    if element is not None and is_match_op_element(element):
                        matching_score = _compute_event_matching_score(state, flow_state, head, event)

                        if matching_score > 0.0:
                            # Successful event match
                            head.matching_scores = event.matching_scores.copy()
                            head.matching_scores.append(matching_score)

                            heads_matching.append(head)
                            if event.name == InternalEvents.START_FLOW:
                                handled_event_loops.add("all_loops")
                            else:
                                assert flow_state.loop_id
                                handled_event_loops.add(flow_state.loop_id)
                            log.info(
                                "Matching head :: %s context=%s",
                                head,
                                _context_log(flow_state),
                            )
                        elif matching_score < 0.0:
                            # Event match mismatch
                            heads_failing.append(head)
                            log.info(
                                "Matching head failed: %s context=%s",
                                head,
                                _context_log(flow_state),
                            )
                        else:
                            # No match nor mismatch
                            heads_not_matching.append(head)

                # Create internal events for unhandled events for every independent interaction loop
                unhandled_event_loops = active_interaction_loops - handled_event_loops
                if (
                    "all_loops" not in handled_event_loops
                    and len(unhandled_event_loops) > 0
                    and event.name != InternalEvents.UNHANDLED_EVENT
                ):
                    arguments = event.arguments.copy()
                    arguments.update({"event": event.name, "loop_ids": unhandled_event_loops})
                    internal_event = create_internal_event(
                        InternalEvents.UNHANDLED_EVENT, arguments, event.matching_scores
                    )
                    _push_internal_event(state, internal_event)

                # Sort matching heads to prioritize more specific matches over the others
                heads_matching = sorted(heads_matching, key=lambda x: x.matching_scores, reverse=True)

                _handle_event_matching(state, event, heads_matching)

                if isinstance(event, ActionEvent):
                    # Update actions status in all active flows by current action event
                    _update_action_status_by_event(state, event)

                # Abort all flows with a mismatch
                for head in heads_failing:
                    if head.catch_pattern_failure_label:
                        head.position = get_flow_config_from_head(state, head).element_labels[
                            head.catch_pattern_failure_label[-1]
                        ]
                        heads_matching.append(head)
                    else:
                        flow_state = get_flow_state_from_head(state, head)
                        _abort_flow(state, flow_state, [])

                # Advance front of all matching heads to actionable or match statements
                for new_head in _advance_head_front(state, heads_matching):
                    if new_head not in actionable_heads:
                        actionable_heads.append(new_head)

            # Separate merging from actionable heads and remove inactive heads
            merging_heads = [head for head in actionable_heads if head.status == FlowHeadStatus.MERGING]
            actionable_heads = [head for head in actionable_heads if head.status == FlowHeadStatus.ACTIVE]

            # Advance all merging heads and create potential new internal events
            actionable_heads.extend(_advance_head_front(state, merging_heads))

            heads_are_merging = len(merging_heads) > 0

        # All internal events are processed and flow heads are either on an action or a match statements
        log.debug("All internal event processed -> advance actionable heads:")

        # Remove heads from stopped or finished flows
        actionable_heads = [
            head
            for head in actionable_heads
            if is_active_flow(get_flow_state_from_head(state, head)) and head.status == FlowHeadStatus.ACTIVE
        ]

        advancing_heads = _resolve_action_conflicts(state, actionable_heads)

        heads_are_advancing = len(advancing_heads) > 0
        actionable_heads = _advance_head_front(state, advancing_heads)
        heads_are_merging = True

    return state


def _clean_up_state(state: State) -> None:
    """Perform a clean up of the state to avoid growing memory footprint."""
    # Clear all matching scores
    for flow_state in state.flow_states.values():
        for head in flow_state.heads.values():
            head.matching_scores.clear()

    # Remove all old flow states based on last status update to limit their number
    # TODO: Refactor, we need to have reference based clean up approach
    states_to_be_removed = []
    for flow_state in state.flow_states.values():
        if (
            _is_done_flow(flow_state)
            and (datetime.now() - flow_state.status_updated) > timedelta(seconds=5)
            and flow_state.activated == 0
        ):
            states_to_be_removed.append(flow_state.uid)
    for flow_state_uid in states_to_be_removed:
        flow_state = state.flow_states[flow_state_uid]
        if (
            flow_state.parent_uid
            and flow_state.parent_uid in state.flow_states
            and flow_state_uid in state.flow_states[flow_state.parent_uid].child_flow_uids
        ):
            state.flow_states[flow_state.parent_uid].child_flow_uids.remove(flow_state_uid)
        flow_states = state.flow_id_states[state.flow_states[flow_state_uid].flow_id]
        flow_states.remove(flow_state)
        del state.flow_states[flow_state_uid]

    # Remove all actions that are no longer referenced
    # TODO: Refactor to use no more ids to simplify memory management
    new_action_dict: Dict[str, Action] = {}
    for flow_state in state.flow_states.values():
        for action_uid in flow_state.action_uids:
            if action_uid not in new_action_dict:
                new_action_dict.update({action_uid: state.actions[action_uid]})
    state.actions = new_action_dict


def _process_internal_events_without_default_matchers(state: State, event: Event) -> Set[str]:
    """
    Process internal events that have no default matchers in flows yet.
    Return a set of all the event loop ids that handled the event.
    """
    handled_event_loops = set()
    if event.name == InternalEvents.START_FLOW:
        # Start new flow state instance if flow exists
        flow_id = event.arguments["flow_id"]
        if flow_id in state.flow_configs and flow_id != "main":
            started_instance = None
            if event.arguments.get("activated", None) and flow_id in state.flow_id_states:
                # The flow was already activated
                assert isinstance(event, InternalEvent)
                started_instance = _get_reference_activated_flow_instance(state, event)

            is_activated_child_flow = flow_id == state.flow_states[event.arguments["source_flow_instance_uid"]].flow_id
            if started_instance and not is_activated_child_flow:
                # Activate a flow that already has been activated

                started_instance.activated = started_instance.activated + 1

                # We add activated flows still as child flows to keep track for termination
                parent_flow = state.flow_states[event.arguments["source_flow_instance_uid"]]
                parent_flow.child_flow_uids.append(started_instance.uid)

                # Send started event to inform calling flow that activated flow was (has been) started
                started_event = started_instance.started_event(
                    event.matching_scores,
                    {"flow_instance_uid": event.arguments["flow_instance_uid"]},
                )
                _push_internal_event(
                    state,
                    started_event,
                )
                handled_event_loops.add("all_loops")
            else:
                # Start a new instance of an activated flow

                if started_instance and is_activated_child_flow:
                    # Create instance as a child of the activated reference flow
                    event.arguments["source_flow_instance_uid"] = started_instance.uid

                add_new_flow_instance(
                    state,
                    create_flow_instance(
                        state.flow_configs[flow_id],
                        event.arguments["flow_instance_uid"],
                        event.arguments["flow_hierarchy_position"],
                        event.arguments,
                    ),
                )

    elif event.name == InternalEvents.FINISH_FLOW:
        # Extract flow parameters
        arguments = dict(event.arguments)

        if "flow_instance_uid" in event.arguments:
            flow_instance_uid = event.arguments["flow_instance_uid"]
            if flow_instance_uid in state.flow_states:
                flow_state = state.flow_states[event.arguments["flow_instance_uid"]]
                if not is_inactive_flow(flow_state):
                    _finish_flow(
                        state=state,
                        flow_state=flow_state,
                        matching_scores=event.matching_scores,
                        context_update=arguments.get("context_update", None),
                    )
                    assert flow_state.loop_id
                    handled_event_loops.add(flow_state.loop_id)
        elif "flow_id" in event.arguments:
            flow_id = arguments.pop("flow_id", None)
            deactivate = arguments.pop("deactivate", False)
            context_update = arguments.pop("context_update", None)
            arguments.pop("source_flow_instance_uid", None)
            arguments.pop("source_head_uid", None)
            if flow_id in state.flow_id_states:
                for flow_state in state.flow_id_states[flow_id]:
                    if arguments.items() <= flow_state.arguments.items():
                        _finish_flow(
                            state=state,
                            flow_state=flow_state,
                            matching_scores=event.matching_scores,
                            deactivate_flow=deactivate,
                            context_update=context_update,
                        )
                        assert flow_state.loop_id
                        handled_event_loops.add(flow_state.loop_id)
    elif event.name == InternalEvents.STOP_FLOW:
        arguments = dict(event.arguments)

        if "flow_instance_uid" in event.arguments:
            flow_instance_uid = event.arguments["flow_instance_uid"]
            if flow_instance_uid in state.flow_states:
                flow_state = state.flow_states[flow_instance_uid]
                if not is_inactive_flow(flow_state):
                    _abort_flow(
                        state=state,
                        flow_state=flow_state,
                        matching_scores=event.matching_scores,
                        deactivate_flow=flow_state.activated > 0,
                        context_update=arguments.get("context_update", None),
                    )
                    assert flow_state.loop_id
                    handled_event_loops.add(flow_state.loop_id)
        elif "flow_id" in event.arguments:
            # Extract flow parameters

            flow_id = arguments.pop("flow_id", None)
            deactivate = arguments.pop("deactivate", False)
            arguments.pop("source_flow_instance_uid", None)
            arguments.pop("source_head_uid", None)
            context_update = arguments.pop("context_update", None)
            if flow_id in state.flow_id_states:
                for flow_state in state.flow_id_states[flow_id]:
                    if arguments.items() <= flow_state.arguments.items():
                        _abort_flow(
                            state=state,
                            flow_state=flow_state,
                            matching_scores=event.matching_scores,
                            deactivate_flow=deactivate,
                            context_update=context_update,
                        )
                        assert flow_state.loop_id
                        handled_event_loops.add(flow_state.loop_id)
        # TODO: Add support for all flow instances of same flow with "flow_id"
    # elif event.name == "ResumeFlow":
    #     pass
    # elif event.name == "PauseFlow":
    #     pass
    elif (
        event.name == InternalEvents.BOT_INTENT_LOG
        or event.name == InternalEvents.USER_INTENT_LOG
        or event.name == InternalEvents.BOT_ACTION_LOG
        or event.name == InternalEvents.USER_ACTION_LOG
    ):
        # We also record the flow finished events in the history
        state.last_events.append(event)
        handled_event_loops.add("all_loops")

    return handled_event_loops


def _get_reference_activated_flow_instance(state: State, event: InternalEvent) -> Optional[FlowState]:
    # Find reference instance for the provided flow
    flow_id = event.arguments["flow_id"]
    for activated_flow in state.flow_id_states[flow_id]:
        # Check if it is not a reference instance
        if (
            activated_flow.activated == 0
            or activated_flow.parent_uid not in state.flow_states
            or (
                activated_flow.parent_uid
                and activated_flow.flow_id == state.flow_states[activated_flow.parent_uid].flow_id
            )
        ):
            continue

        # Check that the reference instance has exactly the same parameters
        matching_parameters: bool = True
        for idx, arg in enumerate(state.flow_configs[flow_id].parameters):
            val = activated_flow.arguments[arg.name]
            # Named flow parameters
            matched = arg.name in event.arguments and val == event.arguments[arg.name]
            # Positional flow parameters
            matched |= f"${idx}" in event.arguments and val == event.arguments[f"${idx}"]
            # Default flow parameters
            matched |= (
                arg.name not in event.arguments
                and f"${idx}" not in event.arguments
                and arg.default_value_expr is not None
                and val == eval_expression(arg.default_value_expr, {})
            )

            if not matched:
                matching_parameters = False
                break

        if matching_parameters:
            return activated_flow

    return None


def _get_all_head_candidates(state: State, event: Event) -> List[Tuple[str, str]]:
    """
    Find all heads that are on a potential match with the event.
    Returns those heads in a flow hierarchical order.
    """
    # Find all heads of flows where the event is relevant
    head_candidates = state.event_matching_heads.get(event.name, []).copy()

    # TODO: We still need to check for those events since they could fail
    # Let's implement that by an explicit keyword for mismatching, e.g. 'not'
    if event.name == InternalEvents.FLOW_FINISHED:
        head_candidates.extend(state.event_matching_heads.get(InternalEvents.FLOW_STARTED, []))
        head_candidates.extend(state.event_matching_heads.get(InternalEvents.FLOW_FAILED, []))
    elif event.name == InternalEvents.FLOW_FAILED:
        head_candidates.extend(state.event_matching_heads.get(InternalEvents.FLOW_STARTED, []))
        head_candidates.extend(state.event_matching_heads.get(InternalEvents.FLOW_FINISHED, []))

    # Ensure that event order is related to interaction loop priority and secondly the flow hierarchy
    sorted_head_candidates = sorted(
        head_candidates,
        key=lambda s: (
            -1 * state.flow_configs[state.flow_states[s[0]].flow_id].loop_priority,
            state.flow_states[s[0]].hierarchy_position,
        ),
    )

    return sorted_head_candidates


def _handle_event_matching(state: State, event: Event, heads_matching: List[FlowHead]) -> None:
    for head in heads_matching:
        element = get_element_from_head(state, head)
        flow_state = get_flow_state_from_head(state, head)

        # Create a potential reference from the match
        if (
            element is not None
            and isinstance(element, SpecOp)
            and isinstance(element.spec, Spec)
            and element.spec.ref is not None
        ):
            flow_state.context.update(_create_event_reference(state, flow_state, element, event))

        if (
            event.name == InternalEvents.START_FLOW
            and event.arguments["flow_id"] == flow_state.flow_id
            and head.position == 0
        ):
            _start_flow(state, flow_state, event.arguments)
        elif event.name == InternalEvents.FLOW_STARTED:
            # Add started flow to active scopes
            # TODO: Make this independent from matching to FlowStarted event since otherwise it could be added elsewhere
            for scope_uid in head.scope_uids:
                if scope_uid in flow_state.scopes:
                    flow_state.scopes[scope_uid][0].append(event.arguments["source_flow_instance_uid"])
        # elif event.name == InternalEvents.FINISH_FLOW:
        #     _finish_flow(new_state, flow_state)
        # TODO: Introduce default matching statements with heads for all flows
        # elif event.name == InternalEvents.ABORT_FLOW:
        #     _abort_flow(new_state, flow_state)
        # elif event.name == "ResumeFlow":
        #     pass
        # elif event.name == "PauseFlow":
        #     pass


def _resolve_action_conflicts(state: State, actionable_heads: List[FlowHead]) -> List[FlowHead]:
    """Resolve all conflicting action conflicts from actionable heads."""

    # Check for potential conflicts between actionable heads
    advancing_heads: List[FlowHead] = []
    if len(actionable_heads) == 1:
        # If we have only one actionable head there is no conflict
        advancing_heads = actionable_heads
        _generate_action_event_from_actionable_element(state, list(actionable_heads)[0])
    elif len(actionable_heads) > 1:
        # Group all actionable heads by their flows interaction loop
        head_groups: Dict[str, List[FlowHead]] = {}
        for head in actionable_heads:
            flow_state = get_flow_state_from_head(state, head)
            assert flow_state.loop_id
            if flow_state.loop_id in head_groups:
                head_groups[flow_state.loop_id].append(head)
            else:
                head_groups.update({flow_state.loop_id: [head]})

        for group in head_groups.values():
            max_length = max(len(head.matching_scores) for head in group)
            ordered_heads = sorted(
                group,
                key=lambda head: head.matching_scores + [1.0] * (max_length - len(head.matching_scores)),
                reverse=True,
            )
            # Check if we have heads with the exact same matching scores and pick one at random (or-group)
            equal_heads_index = next(
                (i for i, h in enumerate(ordered_heads) if h.matching_scores != ordered_heads[0].matching_scores),
                len(ordered_heads),
            )
            picked_head = random.choice(ordered_heads[:equal_heads_index])
            winning_element = get_flow_config_from_head(state, picked_head).elements[picked_head.position]
            assert isinstance(winning_element, SpecOp)
            flow_state = get_flow_state_from_head(state, picked_head)
            winning_event = get_event_from_element(state, flow_state, winning_element)
            log.info(
                "Winning action at head: %s scores=%s",
                picked_head,
                picked_head.matching_scores,
            )

            advancing_heads.append(picked_head)
            _generate_action_event_from_actionable_element(state, picked_head)
            for head in ordered_heads:
                if head == picked_head:
                    continue
                competing_element = get_flow_config_from_head(state, head).elements[head.position]
                assert isinstance(competing_element, SpecOp)
                competing_flow_state = get_flow_state_from_head(state, head)
                competing_event = get_event_from_element(state, competing_flow_state, competing_element)
                if winning_event.is_equal(competing_event):
                    if (
                        isinstance(winning_event, ActionEvent)
                        and winning_event.action_uid
                        and isinstance(competing_event, ActionEvent)
                        and competing_event.action_uid
                    ):
                        # All heads that are on the exact same action as the winning head
                        # need to replace their action references with the winning heads action reference
                        for (
                            key,
                            context_variable,
                        ) in competing_flow_state.context.items():
                            if (
                                isinstance(context_variable, Action)
                                and context_variable.uid == competing_event.action_uid
                            ):
                                action = state.actions[winning_event.action_uid]
                                action.flow_scope_count += 1
                                competing_flow_state.context[key] = action
                        index = competing_flow_state.action_uids.index(competing_event.action_uid)
                        # Adding _action_uid to avoid formatting flipping by black.
                        _action_uid = winning_event.action_uid
                        competing_flow_state.action_uids[index] = _action_uid
                        del state.actions[competing_event.action_uid]

                    advancing_heads.append(head)
                    log.info(
                        "Co-winning action at head: %s scores=%s",
                        head,
                        head.matching_scores,
                    )
                elif head.catch_pattern_failure_label:
                    # If a head defines a pattern failure catch label,
                    # it will forward the head to the label rather the aborting the flow
                    head.position = get_flow_config_from_head(state, head).element_labels[
                        head.catch_pattern_failure_label[-1]
                    ]
                    advancing_heads.append(head)
                    log.info(
                        "Caught loosing action head: %s scores=%s",
                        head,
                        head.matching_scores,
                    )
                else:
                    # Loosing heads will abort the flow
                    flow_state = get_flow_state_from_head(state, head)
                    log.info(
                        "Loosing action at head: %s scores=%s",
                        head,
                        head.matching_scores,
                    )
                    _abort_flow(state, flow_state, head.matching_scores)

    return advancing_heads


def _advance_head_front(state: State, heads: List[FlowHead]) -> List[FlowHead]:
    """
    Advance all provided heads to the next blocking elements (actionable, matching, head merge)
    and returns all heads on actionable elements.
    """
    actionable_heads: List[FlowHead] = []
    for head in heads:
        log.debug("Advancing head: %s flow_state_uid: %s", head, head.flow_state_uid)
        flow_state = get_flow_state_from_head(state, head)
        flow_config = get_flow_config_from_head(state, head)

        if head.status == FlowHeadStatus.INACTIVE or not is_listening_flow(flow_state):
            continue
        elif head.status == FlowHeadStatus.MERGING and len(state.internal_events) > 0:
            # We only advance merging heads if all internal events were processed
            actionable_heads.append(head)
            continue
        elif head.status == FlowHeadStatus.ACTIVE:
            head.position += 1

        if flow_state.status == FlowStatus.WAITING:
            flow_state.status = FlowStatus.STARTING

        flow_finished = False
        flow_aborted = False
        try:
            new_heads = slide(state, flow_state, flow_config, head)

            # Advance all new heads created by a head fork
            if len(new_heads) > 0:
                for new_head in _advance_head_front(state, new_heads):
                    if new_head not in actionable_heads:
                        actionable_heads.append(new_head)

            # Add merging heads to the actionable heads since they need to be advanced in the next iteration
            if head.status == FlowHeadStatus.MERGING:
                actionable_heads.append(head)

            if head.position >= len(flow_config.elements):
                if flow_state.status == FlowStatus.STOPPING:
                    flow_aborted = True
                else:
                    flow_finished = True

            all_heads_are_waiting = False
            if not flow_finished and not flow_aborted:
                # Check if all flow heads are waiting at a 'match' or a 'wait_for_heads' element
                all_heads_are_waiting = True
                for temp_head in flow_state.active_heads.values():
                    element = flow_config.elements[temp_head.position]
                    if not isinstance(element, WaitForHeads) and (
                        not is_match_op_element(element) or (isinstance(element, SpecOp) and "internal" in element.info)
                    ):
                        all_heads_are_waiting = False
                        break

            if flow_finished or all_heads_are_waiting:
                if flow_state.status == FlowStatus.STARTING:
                    flow_state.status = FlowStatus.STARTED
                    event = flow_state.started_event(head.matching_scores)
                    _push_internal_event(state, event)

                    # Avoid an activated flow that was just started from finishing
                    # since this would end in an infinite loop
                    if flow_finished and flow_state.activated > 0:
                        flow_finished = False
                        head.status = FlowHeadStatus.INACTIVE
            elif not flow_aborted:
                elem = get_element_from_head(state, head)
                if elem and is_action_op_element(elem):
                    actionable_heads.append(head)
        except Exception as e:
            # In case there were any runtime error the flow will be aborted (fail)
            source_line = "unknown"
            element = flow_config.elements[head.position]
            if hasattr(element, "_source") and element._source:
                source_line = str(element._source.line)
            log.warning(
                "Flow '%s' failed on line %s (%s) due to Colang runtime exception: %s",
                flow_state.flow_id,
                source_line,
                flow_config.source_file,
                e,
                exc_info=True,
            )
            colang_error_event = Event(
                name="ColangError",
                arguments={
                    "type": str(type(e).__name__),
                    "error": str(e),
                },
            )
            _push_internal_event(state, colang_error_event)
            flow_aborted = True

        if flow_finished:
            _finish_flow(state, flow_state, head.matching_scores)
            log.debug("Flow finished: %s with last element", head.flow_state_uid)
        elif flow_aborted:
            _abort_flow(state, flow_state, head.matching_scores)
            log.debug("Flow aborted: %s by 'abort' statement", head.flow_state_uid)

    # Make sure that all actionable heads still exist in flows, otherwise remove them
    actionable_heads = [
        head for head in actionable_heads if head in state.flow_states[head.flow_state_uid].active_heads.values()
    ]

    return actionable_heads


def slide(state: State, flow_state: FlowState, flow_config: FlowConfig, head: FlowHead) -> List[FlowHead]:
    """Try to slide a flow with the provided head."""
    new_heads: List[FlowHead] = []

    while True:
        # if we reached the end, we stop
        if head.position >= len(flow_config.elements) or head.status == FlowHeadStatus.INACTIVE:
            break

        element = flow_config.elements[head.position]
        log.debug("--Sliding element: '%s'", element)

        if isinstance(element, SpecOp):
            if element.op == "send":
                event = get_event_from_element(state, flow_state, element)

                if event.name not in InternalEvents.ALL:
                    # It's an action event and we need to stop
                    break

                # Add source flow information to event
                event.arguments.update(
                    {
                        "source_flow_instance_uid": head.flow_state_uid,
                        "source_head_uid": head.uid,
                    }
                )

                if event.name == InternalEvents.START_FLOW:
                    # Add flow hierarchy information to event
                    event.arguments.update(
                        {
                            "flow_hierarchy_position": flow_state.hierarchy_position + f".{head.position}",
                        }
                    )

                new_event = create_internal_event(event.name, event.arguments, head.matching_scores)
                _push_internal_event(state, new_event)
                head.position += 1

            elif element.op == "_new_action_instance":
                assert isinstance(element.spec, Spec)
                assert element.spec.spec_type == SpecType.ACTION, "Only actions ca be instantiated!"

                evaluated_arguments = _evaluate_arguments(element.spec.arguments, _get_eval_context(state, flow_state))
                assert element.spec.name
                action = Action(
                    name=element.spec.name,
                    arguments=evaluated_arguments,
                    flow_uid=head.flow_state_uid,
                )
                state.actions.update({action.uid: action})
                flow_state.action_uids.append(action.uid)
                for scope_uid in head.scope_uids:
                    flow_state.scopes[scope_uid][1].append(action.uid)
                assert isinstance(element.spec.ref, dict)
                reference_name = element.spec.ref["elements"][0]["elements"][0].lstrip("$")
                flow_state.context.update({reference_name: action})
                head.position += 1
            else:
                # Not a sliding element
                break

        elif isinstance(element, Label):
            if element.name == "start_new_flow_instance":
                if flow_state.status is not FlowStatus.STARTED:
                    log.warning(
                        "Did not restart flow '%s' at"
                        " label 'start_new_flow_instance' since this would have created an infinite loop!",
                        flow_state.flow_id,
                    )
                else:
                    new_event = flow_state.start_event(head.matching_scores)
                    new_event.arguments["source_flow_instance_uid"] = flow_state.uid
                    _push_left_internal_event(state, new_event)
                    flow_state.new_instance_started = True
            head.position += 1

        elif isinstance(element, Goto):
            if eval_expression(element.expression, _get_eval_context(state, flow_state)):
                if element.label in flow_config.element_labels:
                    head.position = flow_config.element_labels[element.label] + 1
                else:
                    # Advance just to next element for an invalid label
                    log.warning("Invalid label `%s`.", element.label)
                    head.position += 1
            else:
                head.position += 1

        elif isinstance(element, ForkHead):
            # We deactivate current head (parent of new heads)
            head.status = FlowHeadStatus.INACTIVE
            # Register fork uid for later head merge
            flow_state.head_fork_uids[element.fork_uid] = head.uid
            # We create the new child heads
            for _idx, label in enumerate(element.labels):
                parent_fork_head_uid = new_uuid()
                pos = flow_config.element_labels[label]
                new_head = FlowHead(
                    uid=parent_fork_head_uid,
                    flow_state_uid=flow_state.uid,
                    matching_scores=head.matching_scores.copy(),
                    catch_pattern_failure_label=head.catch_pattern_failure_label.copy(),
                    scope_uids=head.scope_uids.copy(),
                )
                new_head.position_changed_callback = partial(_flow_head_changed, state, flow_state)
                new_head.status_changed_callback = partial(_flow_head_changed, state, flow_state)

                flow_state.heads[parent_fork_head_uid] = new_head
                head.child_head_uids.append(new_head.uid)
                new_heads.append(new_head)

                # Trigger the registered flow_head_changed callback function
                new_head.position = pos

            log.debug("Head forked: %s", element.labels)

            break

        elif isinstance(element, MergeHeads):
            if head.status == FlowHeadStatus.ACTIVE:
                # Change status of head to allow for other forked heads to process before merging
                head.status = FlowHeadStatus.MERGING
                break
            elif head.status == FlowHeadStatus.MERGING:
                # Compose a list of all head uids and there children that should be merged
                merging_head_uids: List[str] = []
                scope_uids: List[str] = []
                parent_fork_head_uid = flow_state.head_fork_uids[element.fork_uid]
                parent_fork_head = flow_state.heads[parent_fork_head_uid]
                # TODO: Make sure that child head uids are kept up-to-date to remove this check
                if parent_fork_head_uid in flow_state.heads:
                    merging_head_uids.extend(flow_state.heads[parent_fork_head_uid].get_child_head_uids(state))
                    # Merge scope uids from heads
                    # TODO: Should we really merge them or would it be better to close those scopes instead?
                    for child_heads in parent_fork_head.child_head_uids:
                        scope_uids.extend(
                            [
                                scope_uid
                                for scope_uid in flow_state.heads[child_heads].scope_uids
                                if scope_uid not in scope_uids
                            ]
                        )

                # Check that all of the other heads that are on a merging statements
                # do also target to merge at the same fork uid
                for head_uid in merging_head_uids:
                    if head_uid != head.uid:
                        other_head = flow_state.heads[head_uid]
                        if other_head.status == FlowHeadStatus.MERGING:
                            merge_element = cast(MergeHeads, flow_config.elements[other_head.position])
                            if element.fork_uid != merge_element.fork_uid:
                                # If we still have heads that can be merged independently let's wait
                                break

                # Now we are sure that all other related heads had the chance to process
                # Let's resolve competing heads and merge them with the winner

                # Extract all heads that arrived at a merge statement
                merging_heads = [
                    flow_state.heads[head_uid]
                    for head_uid in merging_head_uids
                    if flow_state.heads[head_uid].status == FlowHeadStatus.MERGING
                ]

                picked_head = head
                if len(merging_heads) > 1:
                    # Order the heads in terms of matching scores
                    max_length = max(len(head.matching_scores) for head in merging_heads)
                    ordered_heads = sorted(
                        merging_heads,
                        key=lambda head: head.matching_scores + [1.0] * (max_length - len(head.matching_scores)),
                        reverse=True,
                    )
                    # Check if we have heads with the exact same matching scores and pick one at random
                    equal_heads_index = next(
                        (
                            i
                            for i, h in enumerate(ordered_heads)
                            if h.matching_scores != ordered_heads[0].matching_scores
                        ),
                        len(ordered_heads),
                    )
                    picked_head = random.choice(ordered_heads[:equal_heads_index])

                head.status = FlowHeadStatus.INACTIVE

                if picked_head == head:
                    # We continue only at the position of the winning head with the head that initiated the fork
                    parent_fork_head.position = head.position
                    parent_fork_head.status = FlowHeadStatus.ACTIVE
                    parent_fork_head.scope_uids = scope_uids
                    parent_fork_head.matching_scores = head.matching_scores
                    parent_fork_head.catch_pattern_failure_label = head.catch_pattern_failure_label
                    parent_fork_head.child_head_uids.clear()
                    new_heads.append(parent_fork_head)

                    # Remove all the merged heads
                    for head_uid in merging_head_uids:
                        flow_state.heads[head_uid].status = FlowHeadStatus.INACTIVE
                        del flow_state.heads[head_uid]
                        if head_uid in flow_state.head_fork_uids:
                            del flow_state.head_fork_uids[head_uid]

                    del flow_state.head_fork_uids[element.fork_uid]

        elif isinstance(element, WaitForHeads):
            # Check if enough heads are on this element to continue
            waiting_heads = [h for h in flow_state.active_heads.values() if h.position == head.position]
            if len(waiting_heads) >= element.number:
                # TODO: Refactoring the merging/waiting for heads so that the clean up is clean
                # Remove all waiting head except for the current
                # for waiting_head in waiting_heads:
                #     if waiting_head.uid != head.uid:
                #         del flow_state.heads[waiting_head.uid]

                head.position += 1
            else:
                break

        elif isinstance(element, Assignment):
            # We need to first evaluate the expression
            expr_val = eval_expression(element.expression, _get_eval_context(state, flow_state))
            if f"_global_{element.key}" in flow_state.context:
                state.context.update({element.key: expr_val})
            else:
                flow_state.context.update({element.key: expr_val})
            head.position += 1

        elif isinstance(element, Return):
            value = None
            if element.expression:
                value = eval_expression(element.expression, _get_eval_context(state, flow_state))
            flow_state.context.update({"_return_value": value})
            head.position = len(flow_config.elements)

        elif isinstance(element, Abort):
            if head.catch_pattern_failure_label:
                head.position = flow_config.element_labels[head.catch_pattern_failure_label[-1]] + 1
            else:
                flow_state.status = FlowStatus.STOPPING
                head.position = len(flow_config.elements)

        elif isinstance(element, Continue) or isinstance(element, Break):
            if element.label is None:
                head.position += 1
            else:
                head.position = flow_config.element_labels[element.label] + 1

        elif isinstance(element, Log):
            log.info(
                "Colang Log %s :: %s",
                flow_state.uid,
                eval_expression(element.info, _get_eval_context(state, flow_state)),
            )
            head.position += 1

        elif isinstance(element, Print):
            console.print(eval_expression(element.info, _get_eval_context(state, flow_state)))
            head.position += 1

        elif isinstance(element, Priority):
            priority = eval_expression(element.priority_expr, _get_eval_context(state, flow_state))
            if not isinstance(priority, float) or priority < 0.0 or priority > 1.0:
                raise ColangValueError("priority must be a float number between 0.0 and 1.0!")
            flow_state.priority = priority
            head.position += 1

        elif isinstance(element, Global):
            var_name = element.name.lstrip("$")
            flow_state.context[f"_global_{var_name}"] = None
            if var_name not in state.context:
                state.context[var_name] = None
            head.position += 1

        elif isinstance(element, CatchPatternFailure):
            if element.label is None:
                head.catch_pattern_failure_label.pop(-1)
            else:
                head.catch_pattern_failure_label.append(element.label)
            head.position += 1

        elif isinstance(element, BeginScope):
            if element.name in head.scope_uids:
                raise ColangRuntimeError(f"Scope with name {element.name} already opened in this head!")
            head.scope_uids.append(element.name)
            if element.name not in flow_state.scopes:
                flow_state.scopes.update({element.name: ([], [])})
            head.position += 1

        elif isinstance(element, EndScope):
            if element.name not in flow_state.scopes:
                raise ColangRuntimeError(f"Scope with name {element.name} does not exist!")
            # Remove scope and stop all started flows/actions in scope
            flow_uids, action_uids = flow_state.scopes.pop(element.name)
            for flow_uid in flow_uids:
                # TODO: This should not be needed if states would be cleaned-up correctly
                if flow_uid in state.flow_states:
                    child_flow_state = state.flow_states[flow_uid]
                    if is_listening_flow(child_flow_state):
                        _abort_flow(state, child_flow_state, head.matching_scores)
            for action_uid in action_uids:
                action = state.actions[action_uid]
                if action.status == ActionStatus.STARTING or action.status == ActionStatus.STARTED:
                    action.flow_scope_count -= 1
                    if action.flow_scope_count == 0:
                        action_event = action.stop_event({})
                        action.status = ActionStatus.STOPPING
                        _generate_umim_event(state, action_event)

            # Remove scope from all heads
            for h in flow_state.heads.values():
                if element.name in h.scope_uids:
                    h.scope_uids.remove(element.name)

            head.position += 1

        else:
            # Ignore unknown element
            head.position += 1

    # If we got this far, it means we had a match and the flow advanced
    return new_heads


def _start_flow(state: State, flow_state: FlowState, event_arguments: dict) -> None:
    if state.main_flow_state is None or flow_state.uid != state.main_flow_state.uid:
        # Link to parent flow
        parent_flow_uid = event_arguments["source_flow_instance_uid"]
        parent_flow = state.flow_states[parent_flow_uid]
        flow_state.parent_uid = parent_flow_uid
        parent_flow.child_flow_uids.append(flow_state.uid)
        flow_state.parent_head_uid = event_arguments["source_head_uid"]

        loop_id = state.flow_configs[flow_state.flow_id].loop_id
        if loop_id is not None:
            if loop_id == "NEW":
                flow_state.loop_id = new_uuid()
            else:
                flow_state.loop_id = loop_id
        else:
            flow_state.loop_id = parent_flow.loop_id

        flow_state.activated = event_arguments.get("activated", 0)
        if flow_state.activated is True:
            flow_state.activated = 1

        # Update context with event/flow parameters
        # TODO: Check if we really need all arguments int the context
        # flow_state.context.update(event_arguments)
        # Inherit parent context
        # context = event_arguments.get("context", None)
        # if context:
        #     flow_state.context = context
        # Resolve positional flow parameters to their actual name in the flow
        last_idx = -1
        for idx, arg in enumerate(flow_state.arguments):
            pos_arg = f"${idx}"
            last_idx = idx
            if pos_arg in event_arguments:
                flow_state.context[arg] = event_arguments[pos_arg]
            else:
                break
        # Check if more parameters were provided than the flow takes
        if f"${last_idx + 1}" in event_arguments:
            raise ColangRuntimeError(f"To many parameters provided in start of flow '{flow_state.flow_id}'")


def _abort_flow(
    state: State,
    flow_state: FlowState,
    matching_scores: List[float],
    deactivate_flow: bool = False,
    context_update: Optional[dict] = None,
) -> None:
    """Abort a flow instance and all its active child flows and decrement number of references of activated flow."""

    if deactivate_flow and _is_reference_activated_flow(state, flow_state):
        # It's a reference activated flow
        flow_state.activated = flow_state.activated - 1
        if flow_state.activated == 0:
            # Abort all activated child flows
            for child_flow_uid in list(flow_state.child_flow_uids):
                child_flow = state.flow_states[child_flow_uid]
                if child_flow.flow_id == flow_state.flow_id:
                    _abort_flow(state, child_flow, matching_scores, True)
                    child_flow.activated = 0

            log.info(
                "Flow deactivated: %s",
                _get_readable_flow_state_hierarchy(state, flow_state.uid),
            )
        else:
            return

    if not is_listening_flow(flow_state) and flow_state.status != FlowStatus.STOPPING:
        # Skip the rest for all inactive flows
        return

    # Abort/deactivate all running child flows
    for child_flow_uid in list(flow_state.child_flow_uids):
        # TODO (cschueller): check why this was the case
        if child_flow_uid not in state.flow_states:
            continue

        child_flow_state = state.flow_states[child_flow_uid]
        if not _is_child_activated_flow(state, child_flow_state):
            _abort_flow(state, child_flow_state, matching_scores, True)

    # Abort all started actions that have not finished yet
    for action_uid in flow_state.action_uids:
        action = state.actions[action_uid]
        if action.status == ActionStatus.STARTING or action.status == ActionStatus.STARTED:
            action.flow_scope_count -= 1
            if action.flow_scope_count == 0:
                action_event = action.stop_event({})
                action.status = ActionStatus.STOPPING
                _generate_umim_event(state, action_event)

    # Cleanup all head from flow
    for head in flow_state.heads.values():
        _remove_head_from_event_matching_structures(state, flow_state, head)
    flow_state.heads.clear()

    # Remove flow uid from parents children list
    if flow_state.activated == 0 and flow_state.parent_uid and flow_state.parent_uid in state.flow_states:
        state.flow_states[flow_state.parent_uid].child_flow_uids.remove(flow_state.uid)

    flow_state.status = FlowStatus.STOPPED

    # Update context if needed
    if context_update:
        flow_state.context.update(context_update)

    # Generate FlowFailed event
    event = flow_state.failed_event(matching_scores)
    _push_internal_event(state, event)

    log.info(
        "Flow aborted/failed: '%s'",
        _get_readable_flow_state_hierarchy(state, flow_state.uid),
    )

    # Restart the flow if it is an activated flow
    if not deactivate_flow and flow_state.activated > 0 and not flow_state.new_instance_started:
        event = flow_state.start_event(matching_scores)
        if flow_state.parent_uid and state.flow_states[flow_state.parent_uid].flow_id == flow_state.flow_id:
            event.arguments.update({"source_flow_instance_uid": flow_state.parent_uid})
        else:
            event.arguments.update({"source_flow_instance_uid": flow_state.uid})
        _push_left_internal_event(state, event)
        flow_state.new_instance_started = True


def _finish_flow(
    state: State,
    flow_state: FlowState,
    matching_scores: List[float],
    deactivate_flow: bool = False,
    context_update: Optional[dict] = None,
) -> None:
    """Finish a flow instance and all its active child flows and decrement number of references of activated flow."""

    if deactivate_flow and _is_reference_activated_flow(state, flow_state):
        # It's a reference activated flow
        flow_state.activated = flow_state.activated - 1
        if flow_state.activated == 0:
            # Abort all activated child flows
            for child_flow_uid in list(flow_state.child_flow_uids):
                child_flow = state.flow_states[child_flow_uid]
                if child_flow.flow_id == flow_state.flow_id:
                    _abort_flow(state, child_flow, matching_scores, True)
                    child_flow.activated = 0
            log.info(
                "Flow deactivated: %s",
                _get_readable_flow_state_hierarchy(state, flow_state.uid),
            )
        else:
            return

    if not is_listening_flow(flow_state):
        # Skip the rest for all inactive flows
        return

    # Abort/deactivate all running child flows
    for child_flow_uid in list(flow_state.child_flow_uids):
        # TODO (cschueller): check why this was the case
        if child_flow_uid not in state.flow_states:
            continue

        child_flow_state = state.flow_states[child_flow_uid]
        if not _is_child_activated_flow(state, child_flow_state):
            _abort_flow(state, child_flow_state, matching_scores, True)

    # Abort all started actions that have not finished yet
    for action_uid in flow_state.action_uids:
        action = state.actions[action_uid]
        if action.status == ActionStatus.STARTING or action.status == ActionStatus.STARTED:
            action.flow_scope_count -= 1
            if action.flow_scope_count == 0:
                action_event = action.stop_event({})
                action.status = ActionStatus.STOPPING
                _generate_umim_event(state, action_event)

    # Cleanup all head from flow
    for head in flow_state.heads.values():
        _remove_head_from_event_matching_structures(state, flow_state, head)
    flow_state.heads.clear()

    # If it is the main flow restart it
    # TODO: Refactor this to use event based mechanics (START_FLOW)
    if flow_state.flow_id == "main":
        # Find an active head
        head_uid = new_uuid()
        new_head = FlowHead(
            uid=head_uid,
            flow_state_uid=flow_state.uid,
            matching_scores=[],
        )
        new_head.position_changed_callback = partial(_flow_head_changed, state, flow_state)
        new_head.status_changed_callback = partial(_flow_head_changed, state, flow_state)
        _flow_head_changed(state, flow_state, new_head)
        flow_state.heads = {head_uid: new_head}
        flow_state.status = FlowStatus.WAITING
        log.info("Main flow finished and restarting...")
        return

    flow_state.status = FlowStatus.FINISHED

    # Remove flow uid from parents children list
    if flow_state.activated == 0 and flow_state.parent_uid and flow_state.parent_uid in state.flow_states:
        state.flow_states[flow_state.parent_uid].child_flow_uids.remove(flow_state.uid)

    # Update context if needed
    if context_update:
        flow_state.context.update(context_update)

    # Generate FlowFinished event
    event = flow_state.finished_event(matching_scores)
    _push_internal_event(state, event)

    _log_action_or_intents(state, flow_state, matching_scores)

    log.info(
        "Flow finished: '%s' context=%s",
        _get_readable_flow_state_hierarchy(state, flow_state.uid),
        _context_log(flow_state),
    )

    # Restart the flow if it is an activated flow
    if not deactivate_flow and flow_state.activated > 0 and not flow_state.new_instance_started:
        event = flow_state.start_event(matching_scores)
        if flow_state.parent_uid and state.flow_states[flow_state.parent_uid].flow_id == flow_state.flow_id:
            event.arguments.update({"source_flow_instance_uid": flow_state.parent_uid})
        else:
            event.arguments.update({"source_flow_instance_uid": flow_state.uid})
        _push_left_internal_event(state, event)
        flow_state.new_instance_started = True


def _log_action_or_intents(state: State, flow_state: FlowState, matching_scores: List[float]) -> None:
    # Check if it was an user/bot intent/action flow and generate internal events
    # TODO: Let's refactor that once we have the new llm prompting
    event_type: Optional[str] = None
    flow_config = state.flow_configs[flow_state.flow_id]
    meta_tag_parameters = None
    if flow_config.has_meta_tag("user_intent"):
        meta_tag_parameters = flow_config.meta_tag("user_intent")
        event_type = InternalEvents.USER_INTENT_LOG
    elif flow_config.has_meta_tag("bot_intent"):
        meta_tag_parameters = flow_config.meta_tag("bot_intent")
        event_type = InternalEvents.BOT_INTENT_LOG
    elif flow_config.has_meta_tag("user_action"):
        meta_tag_parameters = flow_config.meta_tag("user_action")
        event_type = InternalEvents.USER_ACTION_LOG
    elif flow_config.has_meta_tag("bot_action"):
        meta_tag_parameters = flow_config.meta_tag("bot_action")
        event_type = InternalEvents.BOT_ACTION_LOG

    if isinstance(meta_tag_parameters, str):
        meta_tag_parameters = eval_expression(
            '"' + meta_tag_parameters.replace('"', '\\"') + '"',
            _get_eval_context(state, flow_state),
        )

    if event_type == InternalEvents.USER_INTENT_LOG or event_type == InternalEvents.BOT_INTENT_LOG:
        if isinstance(meta_tag_parameters, str):
            name = meta_tag_parameters
            parameter = None
        else:
            # TODO: Generalize to multi flow parameters
            name = (
                flow_state.flow_id
                if not flow_state.flow_id.startswith("_dynamic_") or len(flow_state.flow_id) < 18
                else flow_state.flow_id[18:]
            )
            parameter = flow_state.arguments.get("$0", None)

        event = create_internal_event(
            event_type,
            {
                "flow_id": name,
                "parameter": parameter,
            },
            matching_scores,
        )

        _push_internal_event(state, event)

    elif event_type == InternalEvents.USER_ACTION_LOG or event_type == InternalEvents.BOT_ACTION_LOG:
        hierarchy = _get_flow_state_hierarchy(state, flow_state.uid)
        # Find next intent in hierarchy
        # TODO: Generalize to multi intents
        intent = None
        for flow_state_uid in reversed(hierarchy):
            parent_flow_state = state.flow_states[flow_state_uid]
            intent_flow_config = state.flow_configs[parent_flow_state.flow_id]
            if intent_flow_config.has_meta_tag("bot_intent"):
                intent = intent_flow_config.meta_tag("bot_intent")
            elif intent_flow_config.has_meta_tag("user_intent"):
                intent = intent_flow_config.meta_tag("user_intent")
            elif "_bot_intent" in parent_flow_state.context:
                intent = parent_flow_state.context["_bot_intent"]
            elif "_user_intent" in parent_flow_state.context:
                intent = parent_flow_state.context["_user_intent"]

            if isinstance(intent, str):
                intent = eval_expression(
                    '"' + intent.replace('"', '\\"') + '"',
                    _get_eval_context(state, parent_flow_state),
                )
                break
            elif isinstance(intent, bool):
                intent = intent_flow_config.id
                break

        # Create event based on meta tag
        if isinstance(meta_tag_parameters, str):
            name = meta_tag_parameters
            parameter = None
        else:
            # TODO: Generalize to multi flow parameters
            name = flow_state.flow_id
            parameter = flow_state.arguments.get("$0", None)

        event = create_internal_event(
            event_type,
            {
                "flow_id": name,
                "parameter": parameter,
                "intent_flow_id": intent,
            },
            matching_scores,
        )

        _push_internal_event(state, event)


def _flow_head_changed(state: State, flow_state: FlowState, head: FlowHead) -> None:
    """
    Callback function that is registered to head position/status changes
    and will update acceleration data structures.
    """
    _remove_head_from_event_matching_structures(state, flow_state, head)
    element = get_element_from_head(state, head)
    if (
        element is not None
        and head.status is not FlowHeadStatus.INACTIVE
        and is_listening_flow(flow_state)
        and is_match_op_element(element)
    ):
        _add_head_to_event_matching_structures(state, flow_state, head)


def _add_head_to_event_matching_structures(state: State, flow_state: FlowState, head: FlowHead) -> None:
    flow_config = state.flow_configs[flow_state.flow_id]
    element = flow_config.elements[head.position]
    assert isinstance(element, SpecOp)
    ref_event_name = get_event_name_from_element(state, flow_state, element)
    heads = state.event_matching_heads.get(ref_event_name, None)
    if heads is None:
        state.event_matching_heads.update({ref_event_name: [(flow_state.uid, head.uid)]})
    else:
        heads.append((flow_state.uid, head.uid))
    state.event_matching_heads_reverse_map.update({flow_state.uid + head.uid: ref_event_name})


def _remove_head_from_event_matching_structures(state: State, flow_state: FlowState, head: FlowHead) -> bool:
    event_name = state.event_matching_heads_reverse_map.get(flow_state.uid + head.uid, None)
    if event_name is not None:
        state.event_matching_heads[event_name].remove((flow_state.uid, head.uid))
        state.event_matching_heads_reverse_map.pop(flow_state.uid + head.uid)
        return True
    return False


def _update_action_status_by_event(state: State, event: ActionEvent) -> None:
    for flow_state in state.flow_states.values():
        if not is_listening_flow(flow_state):
            # Don't process flows that are not active
            continue

        for action_uid in flow_state.action_uids:
            # TODO: Make sure that the state.action are deleted so we don't need this check
            if action_uid in state.actions:
                action = state.actions[action_uid]
                if action.status != ActionStatus.FINISHED:
                    action.process_event(event)


def is_listening_flow(flow_state: FlowState) -> bool:
    """True if flow is started or waiting to be started."""
    return (
        flow_state.status == FlowStatus.WAITING
        or flow_state.status == FlowStatus.STARTED
        or flow_state.status == FlowStatus.STARTING
    )


def is_active_flow(flow_state: FlowState) -> bool:
    """True if flow has started."""
    return flow_state.status == FlowStatus.STARTED or flow_state.status == FlowStatus.STARTING


def is_inactive_flow(flow_state: FlowState) -> bool:
    """True if flow is not started."""
    return (
        flow_state.status == FlowStatus.WAITING
        or flow_state.status == FlowStatus.STOPPED
        or flow_state.status == FlowStatus.FINISHED
    )


def _is_done_flow(flow_state: FlowState) -> bool:
    return flow_state.status == FlowStatus.STOPPED or flow_state.status == FlowStatus.FINISHED


def _generate_umim_event(state: State, event: Event) -> Dict[str, Any]:
    umim_event = create_umim_event(event, event.arguments, state.rails_config)
    state.outgoing_events.append(umim_event)
    log.info("[bold violet]<- Action[/]: %s", event)

    # Update the status of relevant actions by event
    if isinstance(event, ActionEvent):
        _update_action_status_by_event(state, event)

    return umim_event


def _push_internal_event(state: State, event: Event) -> None:
    state.internal_events.append(event)
    log.debug("Created internal event: %s", event)


def _push_left_internal_event(state: State, event: InternalEvent) -> None:
    state.internal_events.appendleft(event)
    log.debug("Created internal event: %s", event)


def get_element_from_head(state: State, head: FlowHead) -> Optional[ElementType]:
    """Returns the element at the flow head position"""
    flow_config = get_flow_config_from_head(state, head)
    if head.position >= 0 and head.position < len(flow_config.elements):
        return flow_config.elements[head.position]
    else:
        return None


def get_flow_config_from_head(state: State, head: FlowHead) -> FlowConfig:
    """Return the flow config of the flow of the head"""
    return state.flow_configs[get_flow_state_from_head(state, head).flow_id]


def get_flow_state_from_head(state: State, head: FlowHead) -> FlowState:
    """Return the flow state of the flow head"""
    return state.flow_states[head.flow_state_uid]


def is_action_op_element(element: ElementType) -> bool:
    """Check if the given element is actionable."""
    return (
        isinstance(element, SpecOp)
        and element.op == "send"
        and isinstance(element.spec, Spec)
        and element.spec.name not in InternalEvents.ALL
    )


def is_match_op_element(element: ElementType) -> bool:
    """Check if the given element is a match statement."""
    return isinstance(element, SpecOp) and element.op == "match"


def _evaluate_arguments(arguments: dict, context: dict) -> dict:
    return dict([(key, eval_expression(arguments[key], context)) for key in arguments])


def _get_readable_flow_state_hierarchy(state: State, flow_state_uid: str) -> str:
    hierarchy = _get_flow_state_hierarchy(state, flow_state_uid)
    result = ""
    for flow_state_uid in hierarchy:
        result += flow_state_uid + "/"
    result.rstrip("/")
    return result


def _get_flow_state_hierarchy(state: State, flow_state_uid: str) -> List[str]:
    if flow_state_uid not in state.flow_states:
        return []
    flow_state = state.flow_states[flow_state_uid]
    if flow_state.parent_uid is None:
        return [flow_state.uid]
    else:
        result = _get_flow_state_hierarchy(state, flow_state.parent_uid)
        result.append(flow_state.uid)
        return result


def _compute_event_matching_score(state: State, flow_state: FlowState, head: FlowHead, event: Event) -> float:
    """Check if the element matches with given event."""
    element = get_element_from_head(state, head)
    assert element is not None and isinstance(element, SpecOp) and is_match_op_element(element), (
        f"Element '{element}' is not a match element!"
    )

    ref_event = get_event_from_element(state, flow_state, element)
    if not isinstance(ref_event, type(event)):
        return 0.0

    return _compute_event_comparison_score(state, event, ref_event, flow_state.priority)


def _compute_event_comparison_score(
    state: State, event: Event, ref_event: Event, priority: Optional[float] = None
) -> float:
    """Check if the given element matches the given event.

    Factors that determine the final score:
    - match event parameter specificity
    - flow priority [0.0-1.0]
    - definition order of flow

    Returns:
        1.0: Exact match (all parameters match)
        < 1.0: Fuzzy match (some parameters are missing, but all the others match)
        0.0: No match
        -1.0: Mismatch -> Event will fail the current match
    """

    # Compute matching score based on event argument matching
    match_score: float = 1.0
    if event.name == InternalEvents.START_FLOW and ref_event.name == InternalEvents.START_FLOW:
        match_score = _compute_arguments_dict_matching_score(event.arguments, ref_event.arguments)

        if "flow_id" not in ref_event.arguments:
            match_score *= 0.9
        else:
            match_score = float(
                ref_event.name == InternalEvents.START_FLOW
                and ref_event.arguments["flow_id"] == event.arguments["flow_id"]
            )
    elif event.name in InternalEvents.ALL and ref_event.name in InternalEvents.ALL:
        assert isinstance(event, InternalEvent) and isinstance(ref_event, InternalEvent)
        if (
            "flow_id" in ref_event.arguments
            and "flow_id" in event.arguments
            and _compute_arguments_dict_matching_score(event.arguments["flow_id"], ref_event.arguments["flow_id"])
            != 1.0
        ) or (
            ref_event.flow is not None
            and "source_flow_instance_uid" in event.arguments
            and _compute_arguments_dict_matching_score(event.arguments["source_flow_instance_uid"], ref_event.flow.uid)
            != 1.0
        ):
            return 0.0

        match_score = _compute_arguments_dict_matching_score(event.arguments, ref_event.arguments)

        # TODO: Generalize this with mismatch using e.g. the 'not' keyword
        if match_score > 0.0:
            if "flow_instance_uid" in ref_event.arguments and (
                (ref_event.name == InternalEvents.FLOW_FINISHED and event.name == InternalEvents.FLOW_FAILED)
                or (ref_event.name == InternalEvents.FLOW_FAILED and event.name == InternalEvents.FLOW_FINISHED)
                or (
                    ref_event.name == InternalEvents.FLOW_STARTED
                    and (event.name == InternalEvents.FLOW_FINISHED or event.name == InternalEvents.FLOW_FAILED)
                )
            ):
                # Match failure
                return -1.0
            elif ref_event.name != event.name:
                # Match success
                return 0.0

    else:
        # Its an UMIM event
        if ref_event.name != event.name:
            return 0.0

        event_copy = copy.deepcopy(event)

        if hasattr(event, "action_uid") and hasattr(ref_event, "action_uid"):
            if ref_event.action_uid is not None and ref_event.action_uid != event.action_uid:
                return 0.0

            # TODO: Action event matches can also fail for certain events, e.g. match Started(), received Finished()

            if event.action_uid is not None and event.action_uid in state.actions:
                action_arguments = state.actions[event.action_uid].start_event_arguments
                event_copy.arguments["action_arguments"] = action_arguments

        match_score = _compute_arguments_dict_matching_score(event_copy.arguments, ref_event.arguments)

    # Take into account the priority of the flow
    if priority:
        match_score *= priority

    return match_score


def find_all_active_event_matchers(state: State, event: Optional[Event] = None) -> List[FlowHead]:
    """Return a list of all active heads that point to an event 'match' element."""
    event_matchers: List[FlowHead] = []
    for flow_state in state.flow_states.values():
        if not is_active_flow(flow_state) or not is_listening_flow(flow_state):
            continue

        flow_config = state.flow_configs[flow_state.flow_id]

        for head in flow_state.active_heads.values():
            if head.status != FlowHeadStatus.INACTIVE:
                element = flow_config.elements[head.position]
                if is_match_op_element(element):
                    element = cast(SpecOp, element)
                    if event:
                        element_event = get_event_from_element(state, flow_state, element)
                        score = _compute_event_comparison_score(
                            state,
                            element_event,
                            event,
                        )
                        if score > 0.0:
                            event_matchers.append(head)
                    else:
                        event_matchers.append(head)

    return event_matchers


def _compute_arguments_dict_matching_score(args: Any, ref_args: Any) -> float:
    # TODO: Find a better way of passing arguments to distinguish the ones that count for matching
    score = 1.0
    if isinstance(ref_args, re.Pattern) and (isinstance(args, str) or isinstance(args, int) or isinstance(args, float)):
        args = str(args)
        if not ref_args.search(args):
            return 0.0
    elif isinstance(ref_args, ComparisonExpression):
        return ref_args.compare(args)
    elif not isinstance(ref_args, type(args)):
        return 0.0
    elif isinstance(ref_args, dict):
        argument_filter = ["return_value", "activated", "source_flow_instance_uid"]
        if len(ref_args) > len(args):
            return 0.0
        for val in ref_args.keys():
            if val in argument_filter:
                continue
            elif val in args:
                score *= _compute_arguments_dict_matching_score(args[val], ref_args[val])
                if score == 0.0:
                    return 0.0
            else:
                return 0.0

        # Fuzzy match since number of arguments are not the same
        score *= 0.9 ** (len(args) - len(ref_args))
    elif isinstance(ref_args, list):
        if len(ref_args) > len(args):
            return 0.0
        ref_idx = 0
        idx = 0
        while ref_idx < len(ref_args) and idx < len(args):
            temp_score = _compute_arguments_dict_matching_score(args[idx], ref_args[ref_idx])
            if temp_score > 0.0:
                score *= temp_score
                ref_idx += 1
            idx += 1
        if ref_idx != len(ref_args):
            return 0.0
        # Fuzzy match since number of arguments are not the same
        score *= 0.9 ** (len(args) - len(ref_args))
    elif isinstance(ref_args, set):
        for ref_val in ref_args:
            temp_score = 0.0
            for val in args:
                temp_score = _compute_arguments_dict_matching_score(val, ref_val)
                if temp_score > 0.0:
                    break
            score *= temp_score
            if temp_score == 0.0:
                return 0.0
        # Fuzzy match since number of arguments are not the same
        score *= 0.9 ** (len(args) - len(ref_args))
    elif args != ref_args:
        return 0.0

    return score


def get_event_name_from_element(state: State, flow_state: FlowState, element: SpecOp) -> str:
    """
    Converts the element into the corresponding event name if possible.
    See also function get_event_from_element which is very similar but returns the full event including parameters.
    """

    assert isinstance(element.spec, Spec)
    element_spec: Spec = element.spec

    if element_spec["var_name"] is not None:
        # Case 1)
        variable_name = element_spec["var_name"]
        if variable_name not in flow_state.context:
            raise ColangRuntimeError((f"Unknown variable: '{variable_name}'!"))

        # Resolve variable and member attributes
        obj = flow_state.context[variable_name]
        member = None
        if element_spec.members is not None:
            for member in element_spec.members[:-1]:
                if isinstance(obj, dict):
                    if member.name not in obj:
                        raise ColangValueError(f"No attribute '{member.name}' in {obj}")
                    obj = obj[member.name]
                else:
                    if member.name is None or not hasattr(obj, member.name):
                        raise ColangValueError(f"No attribute '{member.name}' in {obj}")
                    obj = getattr(obj, member.name)
        if element_spec.members is not None:
            member = element_spec.members[-1]

        if isinstance(obj, Event):
            if element_spec.members is not None:
                raise ColangValueError("Events have no event attributes!")
            return obj.name
        elif member is not None and (isinstance(obj, Action) or isinstance(obj, FlowState)):
            if element_spec.members is None:
                raise ColangValueError("Missing event attributes!")
            event_name = member["name"]
            event = obj.get_event(event_name, {})
            return event.name
        else:
            raise ColangRuntimeError(f"Unsupported type '{type(obj)}'")

    elif element_spec.members is not None:
        if element_spec.spec_type == SpecType.FLOW:
            # Flow object
            assert element_spec.name
            flow_config = state.flow_configs[element_spec.name]
            temp_flow_state = create_flow_instance(flow_config, "", "", {})
            flow_event_name = element_spec.members[0]["name"]
            flow_event: InternalEvent = temp_flow_state.get_event(flow_event_name, {})
            del flow_event.arguments["source_flow_instance_uid"]
            del flow_event.arguments["flow_instance_uid"]
            return flow_event.name
        elif element_spec.spec_type == SpecType.ACTION:
            # Action object
            assert element_spec.name
            action = Action(element_spec.name, {}, flow_state.flow_id)
            event_name = element_spec.members[0]["name"]
            action_event: ActionEvent = action.get_event(event_name, {})
            return action_event.name
        else:
            raise ColangRuntimeError(f"Unsupported type '{element_spec.spec_type}'")
    else:
        assert element_spec.name
        return element_spec.name


def get_event_from_element(state: State, flow_state: FlowState, element: SpecOp) -> Event:
    """
    Converts the element into the corresponding event if possible.

    Cases:
    1) Event as member of an action or flow reference: send/match $ref.Finished(args) (This is action/flow specific)
    2) Event as member of an action or flow constructor: send/match UtteranceBotAction(args).Finished(args)
    3) Bare event: send/match UtteranceBotActionFinished(args)
    """

    assert isinstance(element.spec, Spec)
    element_spec: Spec = element.spec

    action: Action
    if element_spec["var_name"] is not None:
        # Case 1)
        variable_name = element_spec["var_name"]
        if variable_name not in flow_state.context:
            raise ColangRuntimeError((f"Unknown variable: '{variable_name}'!"))

        # Resolve variable and member attributes
        obj = flow_state.context[variable_name]
        member = None
        if element_spec.members is not None:
            for member in element_spec.members[:-1]:
                if isinstance(obj, dict):
                    if member.name not in obj:
                        raise ColangValueError(f"No attribute '{member.name}' in {obj}")
                    obj = obj[member.name]
                else:
                    if not member.name or not hasattr(obj, member.name):
                        raise ColangValueError(f"No attribute '{member.name}' in {obj}")
                    obj = getattr(obj, member.name)
        if element_spec.members is not None:
            member = element_spec.members[-1]

        if isinstance(obj, Event):
            if element_spec.members is not None:
                raise ColangValueError("Events have no event attributes!")
            return obj
        elif member is not None and (isinstance(obj, Action) or isinstance(obj, FlowState)):
            if element_spec.members is None:
                raise ColangValueError("Missing event attributes!")
            event_name = member["name"]
            event_arguments = member["arguments"]
            event_arguments = _evaluate_arguments(event_arguments, _get_eval_context(state, flow_state))
            event = obj.get_event(event_name, event_arguments)

            if isinstance(event, InternalEvent) and isinstance(obj, FlowState):
                event.flow = obj
            elif isinstance(event, ActionEvent):
                event.action_uid = obj.uid
                event.action = None

            return event
        else:
            raise ColangRuntimeError(f"Unsupported type '{type(obj)}'")

    elif element_spec.members is not None:
        # Case 2)
        assert element_spec.name
        if element_spec.spec_type == SpecType.FLOW:
            # Flow object
            flow_config = state.flow_configs[element_spec.name]
            temp_flow_state = create_flow_instance(flow_config, "", "", {})
            flow_event_name = element_spec.members[0]["name"]
            flow_event_arguments = element_spec.arguments
            flow_event_arguments.update(element_spec.members[0]["arguments"])
            flow_event_arguments = _evaluate_arguments(flow_event_arguments, _get_eval_context(state, flow_state))
            flow_event: InternalEvent = temp_flow_state.get_event(flow_event_name, flow_event_arguments)
            del flow_event.arguments["source_flow_instance_uid"]
            del flow_event.arguments["flow_instance_uid"]
            if element["op"] == "match":
                # Delete flow reference from event since it is only a helper object
                flow_event.flow = None
            return flow_event
        elif element_spec.spec_type == SpecType.ACTION:
            # Action object
            action_arguments = _evaluate_arguments(element_spec.arguments, _get_eval_context(state, flow_state))
            action = Action(element_spec.name, action_arguments, flow_state.flow_id)
            # TODO: refactor the following repetition of code (see above)
            event_name = element_spec.members[0]["name"]
            event_arguments = element_spec.members[0]["arguments"]
            event_arguments = _evaluate_arguments(event_arguments, _get_eval_context(state, flow_state))
            action_event: ActionEvent = action.get_event(event_name, event_arguments)
            if element["op"] == "match":
                # Delete action_uid from event since the action is only a helper object
                action_event.action_uid = None
            return action_event
    else:
        # Case 3)
        assert element_spec.name
        if element_spec.name.islower() or element_spec.name in InternalEvents.ALL:
            # Flow event
            event_arguments = _evaluate_arguments(element_spec.arguments, _get_eval_context(state, flow_state))
            flow_event = InternalEvent(name=element_spec.name, arguments=event_arguments)
            return flow_event
        elif "Action" in element_spec.name:
            # Action event
            event_arguments = _evaluate_arguments(element_spec.arguments, _get_eval_context(state, flow_state))
            action_event = ActionEvent(name=element_spec.name, arguments=event_arguments)
            return action_event
        else:
            # Event
            event_arguments = _evaluate_arguments(element_spec.arguments, _get_eval_context(state, flow_state))
            new_event = Event(name=element_spec.name, arguments=event_arguments)
            return new_event

    raise ColangRuntimeError("Unsupported case!")


def _generate_action_event_from_actionable_element(
    state: State,
    head: FlowHead,
) -> None:
    """Helper to create an outgoing event from the flow head element."""
    flow_state = get_flow_state_from_head(state, head)
    element = get_element_from_head(state, head)
    assert element is not None and is_action_op_element(element), (
        f"Cannot create an event from a non actionable flow element {element}!"
    )

    if isinstance(element, SpecOp) and element.op == "send":
        event = get_event_from_element(state, flow_state, element)
        umim_event = _generate_umim_event(state, event)
        if isinstance(event, ActionEvent):
            event.action_uid = umim_event["action_uid"]
            assert isinstance(element.spec, Spec)
            # Assign action event to optional reference
            if element.spec.ref and isinstance(element.spec.ref, dict):
                ref_name = element.spec.ref["elements"][0]["elements"][0].lstrip("$")
                flow_state.context.update({ref_name: event})

    # Extract the comment, if any
    # state.next_steps_comment = element.get("_source_mapping", {}).get("comment")


def create_internal_event(event_name: str, event_args: dict, matching_scores: List[float]) -> InternalEvent:
    """Returns an internal event for the provided event data"""
    event = InternalEvent(
        name=event_name,
        arguments=event_args,
        matching_scores=matching_scores,
    )
    return event


def create_umim_event(event: Event, event_args: Dict[str, Any], config: Optional[RailsConfig]) -> Dict[str, Any]:
    """Returns an outgoing UMIM event for the provided action data"""
    new_event_args = dict(event_args)
    new_event_args.setdefault("source_uid", config.event_source_uid if config else "NeMoGuardrails-Colang-2.x")
    if isinstance(event, ActionEvent) and event.action_uid is not None:
        if "action_uid" in new_event_args:
            event.action_uid = new_event_args["action_uid"]
            del new_event_args["action_uid"]
        return new_event_dict(event.name, action_uid=event.action_uid, **new_event_args)
    else:
        return new_event_dict(event.name, **new_event_args)


def _get_eval_context(state: State, flow_state: FlowState) -> dict:
    context = flow_state.context.copy()
    # Link global variables
    for var in flow_state.context.keys():
        if var.startswith("_global_"):
            context.update({var: state.context[var[8:]]})
    # Add state as _state
    context.update({"system": {"state": state, "config": state.rails_config}})
    context.update({"self": flow_state})
    return context


def _is_reference_activated_flow(state: State, flow_state: FlowState) -> bool:
    return (
        flow_state.activated > 0
        and flow_state.parent_uid is not None
        and flow_state.flow_id != state.flow_states[flow_state.parent_uid].flow_id
    )


def _is_child_activated_flow(state: State, flow_state: FlowState) -> bool:
    return (
        flow_state.activated > 0
        and flow_state.parent_uid is not None
        and flow_state.parent_uid in state.flow_states  # TODO: Figure out why this can fail sometimes
        and flow_state.flow_id == state.flow_states[flow_state.parent_uid].flow_id
    )
