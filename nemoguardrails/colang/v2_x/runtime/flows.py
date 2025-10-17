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

"""A simplified modeling of the CoFlows engine."""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, Union

from dataclasses_json import dataclass_json

from nemoguardrails.colang.v2_x.lang.colang_ast import (
    ElementType,
    FlowParamDef,
    FlowReturnMemberDef,
)
from nemoguardrails.colang.v2_x.runtime.errors import ColangSyntaxError
from nemoguardrails.utils import new_readable_uuid, new_uuid

log = logging.getLogger(__name__)

random_seed = int(time.time())


class InternalEvents:
    """All internal event types. This event will not appear in the event stream and have priority over them."""

    START_FLOW = "StartFlow"  # Starts a new flow instance
    FINISH_FLOW = "FinishFlow"  # Flow will be finished successfully
    STOP_FLOW = "StopFlow"  # Flow will be stopped and failed
    FLOW_STARTED = "FlowStarted"  # Flow has started (reached first official match statement or end)
    FLOW_FINISHED = "FlowFinished"  # Flow has finished successfully
    FLOW_FAILED = "FlowFailed"  # Flow has failed
    UNHANDLED_EVENT = "UnhandledEvent"  # For any unhandled event in a specific interaction loop we create an unhandled event

    # TODO: Check if we could convert them into just an internal list to track action/intents
    BOT_INTENT_LOG = "BotIntentLog"
    USER_INTENT_LOG = "UserIntentLog"
    BOT_ACTION_LOG = "BotActionLog"
    USER_ACTION_LOG = "UserActionLog"

    ALL = {
        START_FLOW,
        FINISH_FLOW,
        STOP_FLOW,
        FLOW_STARTED,
        FLOW_FINISHED,
        FLOW_FAILED,
        UNHANDLED_EVENT,
        BOT_INTENT_LOG,
        USER_INTENT_LOG,
        BOT_ACTION_LOG,
        USER_ACTION_LOG,
    }


@dataclass
class Event:
    """The base event class."""

    # Name of the event
    name: str

    # Context that contains all relevant event arguments
    arguments: dict

    # A list of matching scores from the event sequence triggered by an external event
    matching_scores: List[float] = field(default_factory=list)

    def is_equal(self, other: Event) -> bool:
        """Compares two events in terms of their name and arguments."""
        if isinstance(other, Event):
            return self.name == other.name and self.arguments == other.arguments
        return NotImplemented

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Event):
            return self.is_equal(other)
        return False

    def __str__(self) -> str:
        return f"[bold blue]{self.name}[/] {self.arguments}"

    @classmethod
    def from_umim_event(cls, event: dict) -> Event:
        """Creates an event from a flat dictionary."""
        new_event = Event(event["type"], {})
        new_event.arguments = dict(
            [(key, event[key]) for key in event if key not in ["type"]]
        )
        return new_event

    # Expose all event parameters as attributes of the event
    def __getattr__(self, name):
        if (
            name not in self.__dict__
            and "arguments" in self.__dict__
            and name in self.__dict__["arguments"]
        ):
            return self.__dict__["arguments"][name]
        else:
            return object.__getattribute__(self, "params")[name]


@dataclass
class InternalEvent(Event):
    """The internal event class."""

    # An internal event can belong to a flow
    flow: Optional[FlowState] = None


@dataclass
class ActionEvent(Event):
    """The action event class."""

    # The event can belong to an action
    action_uid: Optional[str] = None

    # This is the action reference to enable direct access via expressions
    # This needs to be consistent with the action_uid
    action: Optional[Action] = None

    @classmethod
    def from_umim_event(cls, event: dict) -> ActionEvent:
        """Creates an event from a flat dictionary."""
        new_event = ActionEvent(event["type"], {})
        new_event.arguments = dict(
            [(key, event[key]) for key in event if key not in ["type"]]
        )
        if "action_uid" in event:
            new_event.action_uid = event["action_uid"]
        return new_event


class ActionStatus(Enum):
    """The status of an action."""

    INITIALIZED = (
        "initialized"  # Action object created but StartAction event not yet sent
    )
    STARTING = "starting"  # StartAction event sent, waiting for ActionStarted event
    STARTED = "started"  # ActionStarted event received
    STOPPING = "stopping"  # StopAction event sent, waiting for ActionFinished event
    FINISHED = "finished"  # ActionFinished event received


class Action:
    """The action class groups and manages the action events."""

    # The action event name mapping
    _event_name_map = {
        "Started": "started_event",
        "Updated": "updated_event",
        "Finished": "finished_event",
        "Start": "start_event",
        "Change": "change_event",
        "Stop": "stop_event",
    }

    @classmethod
    def from_event(cls, event: ActionEvent) -> Optional[Action]:
        """Returns the action if event name conforms with UMIM convention."""
        assert event.action_uid is not None
        for name in cls._event_name_map:
            if name in event.name:
                action = Action(event.name.replace(name, ""), {})
                action.uid = event.action_uid
                action.status = (
                    ActionStatus.STARTED
                    if name != "Finished"
                    else ActionStatus.FINISHED
                )
                return action
        return None

    def __init__(
        self, name: str, arguments: Dict[str, Any], flow_uid: Optional[str] = None
    ) -> None:
        # The unique id of the action
        self.uid: str = new_uuid()

        # Name of the action
        self.name = name

        # An action belongs to the flow that it started
        self.flow_uid = flow_uid

        # State of the action
        self.status: ActionStatus = ActionStatus.INITIALIZED

        # Context that contains all relevant action parameters
        self.context: dict = {}

        # The arguments that will be used for the start event
        self.start_event_arguments = arguments

        # Number of flows where this action is still active
        self.flow_scope_count = 0

    def to_dict(self):
        return {
            "uid": self.uid,
            "name": self.name,
            "flow_uid": self.flow_uid,
            "status": self.status.name,
            "context": self.context,
            "start_event_arguments": self.start_event_arguments,
            "flow_scope_count": self.flow_scope_count,
        }

    @staticmethod
    def from_dict(d):
        action = Action(
            name=d["name"], arguments=d["start_event_arguments"], flow_uid=d["flow_uid"]
        )
        action.uid = d["uid"]
        action.status = ActionStatus[d["status"]]
        action.context = d["context"]
        action.flow_scope_count = d["flow_scope_count"]
        return action

    # Process an event
    def process_event(self, event: ActionEvent) -> None:
        """Processes event and updates action accordingly."""
        # TODO: This matching can easily break if action names are badly chosen
        if "Action" in event.name and event.action_uid == self.uid:
            if "ActionStarted" in event.name:
                self.context.update(event.arguments)
                self.status = ActionStatus.STARTED
            elif "ActionUpdated" in event.name:
                self.context.update(event.arguments)
            elif "ActionFinished" in event.name:
                self.context.update(event.arguments)
                self.status = ActionStatus.FINISHED
                self.flow_scope_count = 0
            elif "Start" in event.name:
                self.context.update(event.arguments)
                self.status = ActionStatus.STARTING
                self.flow_scope_count = 1
            elif "Stop" in event.name:
                self.context.update(event.arguments)
                self.status = ActionStatus.STOPPING

    def get_event(self, name: str, arguments: dict) -> ActionEvent:
        """Returns the corresponding action event."""
        if name.endswith("Updated"):
            if len(name) > 7:
                split_name = name.rsplit("Updated", 1)
                if split_name[0] == "":
                    raise ColangSyntaxError(f"Invalid action event {name}!")
                arguments.update({"event_parameter_name": split_name[0]})
                name = "Updated"
            else:
                arguments.update({"event_parameter_name": ""})
        if name not in Action._event_name_map:
            raise ColangSyntaxError(f"Invalid action event {name}!")
        func = getattr(self, Action._event_name_map[name])
        return func(arguments)

    # Action events to send
    def start_event(self, _args: dict) -> ActionEvent:
        """Starts the action. Takes no arguments."""
        return ActionEvent(
            name=f"Start{self.name}",
            arguments=self.start_event_arguments,
            action_uid=self.uid,
        )

    def change_event(self, args: dict) -> ActionEvent:
        """Changes a parameter of a started action."""
        return ActionEvent(
            name=f"Change{self.name}", arguments=args["arguments"], action_uid=self.uid
        )

    def stop_event(self, _args: dict) -> ActionEvent:
        """Stops a started action. Takes no arguments."""
        return ActionEvent(name=f"Stop{self.name}", arguments={}, action_uid=self.uid)

    # Action events to match
    def started_event(self, args: dict) -> ActionEvent:
        """Returns the Started action event."""
        arguments = args.copy()
        if self.start_event_arguments:
            arguments["action_arguments"] = self.start_event_arguments
        return ActionEvent(
            name=f"{self.name}Started", arguments=arguments, action_uid=self.uid
        )

    def updated_event(self, args: dict) -> ActionEvent:
        """Returns the Updated parameter action event."""
        name = f"{self.name}{args['event_parameter_name']}Updated"
        arguments = args.copy()
        del arguments["event_parameter_name"]
        if self.start_event_arguments:
            arguments["action_arguments"] = self.start_event_arguments
        return ActionEvent(
            name=name,
            arguments=arguments,
            action_uid=self.uid,
        )

    def finished_event(self, args: dict) -> ActionEvent:
        """Returns the Finished action event."""
        arguments = args.copy()
        if self.start_event_arguments:
            arguments["action_arguments"] = self.start_event_arguments
        return ActionEvent(
            name=f"{self.name}Finished", arguments=arguments, action_uid=self.uid
        )

    # Expose all action parameters as attributes
    def __getattr__(self, name):
        if (
            name not in self.__dict__
            and "context" in self.__dict__
            and name in self.__dict__["context"]
        ):
            return self.__dict__["context"][name]
        else:
            return object.__getattribute__(self, "params")[name]


class InteractionLoopType(Enum):
    """The type of the interaction loop."""

    NEW = "new"  # Every new instance of the flow will live in its own new loop
    PARENT = "parent"  # Every new instance of the flow will live in the same loop as its the parent
    NAMED = "named"  # Every new instance of the flow will live in the loop with the given name


@dataclass
class FlowConfig:
    """The configuration of a flow."""

    # A unique id of the flow.
    id: str

    # The sequence of elements that compose the flow.
    elements: List[ElementType]

    # The flow parameters
    parameters: List[FlowParamDef]

    # The decorators for the flow.
    # Maps the name of the applied decorators to the arguments.
    # If positional arguments are provided, then the "$0", "$1", ... are used as the keys.
    decorators: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # The flow return member variables
    return_members: List[FlowReturnMemberDef] = field(default_factory=list)

    # All the label element positions in the flow
    element_labels: Dict[str, int] = field(default_factory=dict)

    # The actual source code, if available
    source_code: Optional[str] = None

    # The name of the source code file
    source_file: Optional[str] = None

    @property
    def loop_id(self) -> Optional[str]:
        """Return the interaction loop id if set."""
        if "loop" in self.decorators:
            parameters = self.decorators["loop"]
            if "id" in parameters:
                return parameters["id"]
            elif "$0" in parameters:
                return parameters["$0"]
            else:
                log.warning(
                    "No loop id specified for @loop decorator for flow `%s`", self.id
                )
        return None

    @property
    def loop_priority(self) -> int:
        """Return the interaction loop priority (default: 0)."""
        if "loop" in self.decorators:
            parameters = self.decorators["loop"]
            if "priority" in parameters:
                return parameters["priority"]
            elif "$1" in parameters:
                return parameters["$1"]
        return 0

    @property
    def loop_type(self) -> InteractionLoopType:
        """Return the interaction loop type."""
        loop_id = self.loop_id
        if loop_id == "NEW":
            return InteractionLoopType.NEW
        elif loop_id is not None:
            return InteractionLoopType.NAMED
        else:
            return InteractionLoopType.PARENT

    @property
    def is_override(self) -> bool:
        """Return True if flow is marked as override."""
        return "override" in self.decorators

    def has_meta_tag(self, tag_name: str) -> bool:
        """Return True if flow is marked with given meta tag, e.g. `@meta(llm_exclude=True)`."""
        return "meta" in self.decorators and tag_name in self.decorators["meta"]

    def meta_tag(self, tag_name: str) -> Optional[Any]:
        """Return the parameter of the meta tag or None if it does not exist."""
        if not self.has_meta_tag(tag_name):
            return None
        else:
            return self.decorators["meta"][tag_name]


class FlowHeadStatus(Enum):
    """The status of a flow head."""

    ACTIVE = "active"  # The head is active and either waiting or progressing
    INACTIVE = "inactive"  # The head is no longer progressing (e.g. is a parent of an active child head)
    MERGING = "merging"  # The head arrived at a head merging element and will progress only in the next iteration


@dataclass
class FlowHead:
    """The flow head that points to a certain element in the flow"""

    # The unique id of a flow head
    uid: str

    # The flow of the head
    flow_state_uid: str

    # Matching score history of previous matches that resulted in this head to be advanced
    matching_scores: List[float]

    # List of all scopes that are relevant for the head
    # TODO: Check if scopes are really needed or if they could be replaced by the head forking/merging
    scope_uids: List[str] = field(default_factory=list)

    # If a flow head is forked it will create new child heads
    child_head_uids: List[str] = field(default_factory=list)

    # If set, a flow failure (abort) will be forwarded to the label, otherwise it will abort/fail the flow
    catch_pattern_failure_label: List[str] = field(default_factory=list)

    # Callback that can be registered to get informed about head position updates
    position_changed_callback: Optional[Callable[[FlowHead], None]] = None

    # Callback that can be registered to get informed about head status updates
    status_changed_callback: Optional[Callable[[FlowHead], None]] = None

    # The position of the flow element the head is pointing to
    _position: int = 0

    @property
    def position(self) -> int:
        """Return the current position of the head."""
        return self._position

    @position.setter
    def position(self, position: int) -> None:
        """Set the position of the head."""
        if position != self._position:
            self._position = position
            if self.position_changed_callback is not None:
                self.position_changed_callback(self)

    # Whether a head is active or not (a head fork will deactivate the parent head)
    _status: FlowHeadStatus = FlowHeadStatus.ACTIVE

    @property
    def status(self) -> FlowHeadStatus:
        """Return the current status of the head."""
        return self._status

    @status.setter
    def status(self, status: FlowHeadStatus) -> None:
        """Set the status of the head."""
        if status != self._status:
            self._status = status
            if self.status_changed_callback is not None:
                self.status_changed_callback(self)

    def get_child_head_uids(self, state: State) -> List[str]:
        """Return uids of all child heads (recursively)."""
        flow_state = state.flow_states[self.flow_state_uid]
        child_uids: List[str] = []
        for uid in self.child_head_uids:
            child_uids.append(uid)
            # TODO: Make sure that child head uids are kept up-to-date
            if uid in flow_state.heads:
                child_uids.extend(flow_state.heads[uid].get_child_head_uids(state))
        return child_uids

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, FlowHead):
            return self.uid == other.uid
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.uid)

    def __str__(self) -> str:
        return f"flow='{self.flow_state_uid.split(')',1)[0][1:]}' pos={self.position}"

    def __repr__(self) -> str:
        return f"FlowHead[uid={self.uid}, flow_state_uid={self.flow_state_uid}]"


class FlowStatus(Enum):
    """The status of a flow."""

    WAITING = "waiting"  # Waiting for the flow to start (at first match statement)
    STARTING = "starting"  # Flow has been started but head is not yet at the first match statement ('_match' excluded)
    STARTED = "started"  # Flow has started when head arrived at the first match statement ('_match' excluded)
    STOPPING = "stopping"  # Flow was stopped (e.g. by 'abort') but did not yet stop all child flows or actions
    STOPPED = "stopped"  # Flow has stopped/failed and all child flows and actions
    FINISHED = (
        "finished"  # Flow has finished and all child flows and actions were stopped
    )


# TODO: Rename just to "Flow" for better clarity, also all variables flow_state -> flow
@dataclass
class FlowState:
    """The state of a flow."""

    # The unique id of the flow instance
    uid: str

    # The name id of the flow
    flow_id: str

    # Interaction loop id
    loop_id: Optional[str]

    # An identifier that determines the exact position in the flow hierarchy tree.
    # E.g. in "0.1.0.4" each number represents the related start/activation element position in the related parent flow
    hierarchy_position: str

    # All the heads pointing to certain element positions in the flow.
    heads: Dict[str, FlowHead] = field(default_factory=dict)

    # All active/open scopes that contain a tuple of flow uids and action uids that were started within that scope
    scopes: Dict[str, Tuple[List[str], List[str]]] = field(default_factory=dict)

    # Relates head_fork_uids to corresponding child heads
    # Help structure to relate child heads to their parents by a predefined id
    head_fork_uids: Dict[str, str] = field(default_factory=dict)

    # All actions that were instantiated since the beginning of the flow
    action_uids: List[str] = field(default_factory=list)

    # The current set of variables in the flow state.
    context: dict = field(default_factory=dict)

    # The current priority of the flow instance that is used for action resolution.
    priority: float = 1.0

    # All the arguments of a flow (e.g. flow bot say $utterance -> arguments = ["$utterance"])
    arguments: Dict[str, Any] = field(default_factory=dict)

    # Parent flow id
    parent_uid: Optional[str] = None

    # Parent flow head id
    parent_head_uid: Optional[str] = None

    # The ids of all the child flows
    child_flow_uids: List[str] = field(default_factory=list)

    # The current state of the flow
    _status: FlowStatus = FlowStatus.WAITING

    # The datetime of the last status change of the flow
    status_updated: datetime = datetime.now()

    # An activated flow will restart immediately when finished. The integer counts the activations.
    activated: int = 0

    # True if a new instance was started either by restarting or
    # an early 'start_new_flow_instance' label
    new_instance_started: bool = False

    # The flow event name mapping
    _event_name_map: dict = field(init=False)

    @property
    def status(self) -> FlowStatus:
        """The status of the flow"""
        return self._status

    @status.setter
    def status(self, status: FlowStatus) -> None:
        self._status = status
        self.status_updated = datetime.now()

    @property
    def active_heads(self) -> Dict[str, FlowHead]:
        """All active heads of this flow."""
        return {
            id: h
            for (id, h) in self.heads.items()
            if h.status != FlowHeadStatus.INACTIVE
        }

    def __post_init__(self) -> None:
        self._event_name_map = {
            "Start": "start_event",
            "Stop": "stop_event",
            "Pause": "pause_event",
            "Resume": "resume_event",
            "Started": "started_event",
            "Paused": "paused_event",
            "Resumed": "resumed_event",
            "Finished": "finished_event",
            "Failed": "failed_event",
        }

    def get_event(
        self, name: str, arguments: dict, matching_scores: Optional[List[float]] = None
    ) -> InternalEvent:
        """Returns the corresponding action event."""
        assert name in self._event_name_map, f"Event '{name}' not available!"
        func = getattr(self, self._event_name_map[name])
        if not matching_scores:
            matching_scores = []
        return func(matching_scores, arguments)

    # Flow events to send
    def start_event(
        self, matching_scores: List[float], args: Optional[dict] = None
    ) -> InternalEvent:
        """Starts the flow. Takes no arguments."""
        arguments = {
            "flow_instance_uid": new_readable_uuid(self.flow_id),
            "flow_id": self.flow_id,
            "source_flow_instance_uid": self.parent_uid,
            "source_head_uid": self.parent_head_uid,
            "flow_hierarchy_position": self.hierarchy_position,
            "activated": self.activated,
        }
        arguments.update(self.arguments)
        if args:
            arguments.update(args)
        return InternalEvent(
            name=InternalEvents.START_FLOW,
            arguments=arguments,
            matching_scores=matching_scores,
        )

    def finish_event(self, matching_scores: List[float], _args: dict) -> InternalEvent:
        """Finishes the flow. Takes no arguments."""
        return InternalEvent(
            name=InternalEvents.FINISH_FLOW,
            arguments={"flow_id": self.flow_id, "flow_instance_uid": self.uid},
            matching_scores=matching_scores,
        )

    def stop_event(self, matching_scores: List[float], _args: dict) -> InternalEvent:
        """Stops the flow. Takes no arguments."""
        return InternalEvent(
            name="StopFlow",
            arguments={"flow_id": self.flow_id, "flow_instance_uid": self.uid},
            matching_scores=matching_scores,
        )

    def pause_event(self, matching_scores: List[float], _args: dict) -> InternalEvent:
        """Pauses the flow. Takes no arguments."""
        return InternalEvent(
            name="PauseFlow",
            arguments={"flow_id": self.flow_id, "flow_instance_uid": self.uid},
            matching_scores=matching_scores,
        )

    def resume_event(self, matching_scores: List[float], _args: dict) -> InternalEvent:
        """Resumes the flow. Takes no arguments."""
        return InternalEvent(
            name="ResumeFlow",
            arguments={"flow_id": self.flow_id, "flow_instance_uid": self.uid},
            matching_scores=matching_scores,
        )

    # Flow events to match
    def started_event(
        self, matching_scores: List[float], args: Optional[Dict[str, Any]] = None
    ) -> InternalEvent:
        """Returns the flow Started event."""
        return self._create_out_event(
            InternalEvents.FLOW_STARTED, matching_scores, args
        )

    # def paused_event(self, args: dict) -> FlowEvent:
    #     """Returns the flow Pause event."""
    #     return self._create_event(InternalEvents.FLOW_PAUSED, args)

    # def resumed_event(self, args: dict) -> FlowEvent:
    #     """Returns the flow Resumed event."""
    #     return self._create_event(InternalEvents.FLOW_RESUMED, args)

    def finished_event(
        self, matching_scores: List[float], args: Optional[Dict[str, Any]] = None
    ) -> InternalEvent:
        """Returns the flow Finished event."""
        if not args:
            args = {}
        if "_return_value" in self.context:
            args["return_value"] = self.context["_return_value"]
        return self._create_out_event(
            InternalEvents.FLOW_FINISHED, matching_scores, args
        )

    def failed_event(
        self, matching_scores: List[float], args: Optional[Dict[str, Any]] = None
    ) -> InternalEvent:
        """Returns the flow Failed event."""
        return self._create_out_event(InternalEvents.FLOW_FAILED, matching_scores, args)

    def _create_out_event(
        self,
        event_type: str,
        matching_scores: List[float],
        args: Optional[Dict[str, Any]],
    ) -> InternalEvent:
        arguments = {}
        arguments["source_flow_instance_uid"] = self.uid
        arguments["flow_instance_uid"] = self.uid
        arguments["flow_id"] = self.flow_id
        arguments.update(self.arguments)
        if args:
            arguments.update(args)
        return InternalEvent(event_type, arguments, matching_scores)

    def __repr__(self) -> str:
        return (
            f"FlowState[uid={self.uid}, flow_id={self.flow_id}, loop_id={self.loop_id}]"
        )

    # Expose all flow variables as attributes of the flow
    # TODO: Hide non public flow variables
    def __getattr__(self, name):
        if (
            name not in self.__dict__
            and "context" in self.__dict__
            and name in self.__dict__["context"]
        ):
            return self.__dict__["context"][name]
        else:
            return object.__getattribute__(self, "params")[name]


@dataclass_json
@dataclass
class State:
    """The state of a flow-driven system."""

    # The current set of flow instances with their uid as key.
    flow_states: Dict[str, FlowState]

    # The configuration of all the flows that are available.
    flow_configs: Dict[str, FlowConfig]

    # The full rails configuration object
    rails_config: Optional["RailsConfig"] = None

    # All actions that were instantiated in a flow that is still referenced somewhere
    actions: Dict[str, Action] = field(default_factory=dict)

    # Queue of internal events
    internal_events: Deque[Event] = field(default_factory=deque)

    # The main flow state
    main_flow_state: Optional[FlowState] = None

    # The global context that contains all flow variables defined as global
    context: Dict[str, Any] = field(default_factory=dict)

    # The resulting events of event-driven system
    outgoing_events: List[dict] = field(default_factory=list)

    # The most recent N events that have been processed. Will be capped at a
    # reasonable limit e.g. 500. The history is needed when prompting the LLM for example.
    # TODO: Clean this up to only use one type
    last_events: List[Union[dict, Event]] = field(default_factory=list)

    # The updates to the context that should be applied before the next step
    # TODO: This would be needed if we decide to implement assignments of global variables via context updates
    context_updates: dict = field(default_factory=dict)

    ########################
    # Helper data structures
    ########################

    # Helper dictionary that maps from flow_id (name) to all available flow states
    flow_id_states: Dict[str, List[FlowState]] = field(default_factory=dict)

    # Helper dictionary that maps active event matchers (by event names) to relevant heads (flow_state_uid, head_uid)
    event_matching_heads: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)

    # Helper dictionary that maps active heads (flow_state_uid, head_uid) to event matching names
    # The key is constructed as the concatenation of the two ids.
    event_matching_heads_reverse_map: Dict[str, str] = field(default_factory=dict)
