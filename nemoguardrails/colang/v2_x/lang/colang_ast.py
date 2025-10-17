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

"""The data types that are used when constructing the Abstract Syntax Tree after parsing."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Source:
    """Information about the source of an element."""

    line: int
    column: int
    start_pos: int
    end_pos: int


@dataclass_json
@dataclass
class FlowParamDef:
    """The definition for a flow parameter.

    Explicit typing is not yet supported.
    """

    name: str
    default_value_expr: Optional[str] = None


@dataclass_json
@dataclass
class FlowReturnMemberDef:
    """The definition for a flow return member.

    Explicit typing is not yet supported.
    """

    name: str
    default_value_expr: Optional[str] = None


@dataclass_json
@dataclass
class Element:
    """Base class for all elements in the AST."""

    _type: str
    _source: Optional[Source] = None

    def __getitem__(self, key):
        return getattr(self, key, None)

    def get(self, key, default_value=None):
        """Getter for backward compatibility with dict elements.

        TODO: to remove at some point.
        """
        return self[key] or default_value

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__hash__() == other.__hash__()
        return NotImplemented

    def hash(self):
        """Return the hash for the current object."""
        return hash(_make_hashable(self))


ElementType = Union[Element, dict]


def _make_hashable(obj: Any) -> Any:
    """Make all subtypes of Element hashable."""
    if isinstance(obj, dict):
        return tuple((k, _make_hashable(v)) for k, v in sorted(obj.items()))
    elif isinstance(obj, list):
        return tuple(_make_hashable(x) for x in obj)
    elif isinstance(obj, tuple):
        return tuple(_make_hashable(x) for x in obj)
    elif isinstance(obj, Element):
        return tuple((k, _make_hashable(v)) for k, v in sorted(vars(obj).items()))
    else:
        return obj


@dataclass_json
@dataclass
class Decorator(Element):
    """The definition for a flow decorator.

    Explicit typing is not yet supported.
    """

    name: str = ""
    parameters: dict = field(default_factory=dict)
    _type: str = "decorator"


@dataclass_json
@dataclass
class Import(Element):
    """The definition for an import statement.

    We support both "path mode" and "package" mode.

        import core
        import rag.advanced
        import "some-spec/some-sub-package"
    """

    path: Optional[str] = None
    package: Optional[str] = None
    _type: str = "import"


@dataclass_json
@dataclass
class Value(Element):
    """Element that contains a value."""

    value: Any = None
    _type: str = "value"


@dataclass_json
@dataclass
class Elements(Element):
    """Element that encapsulates multiple sub-elements."""

    elements: List[Element] = field(default_factory=list)
    _type: str = "value"


@dataclass_json
@dataclass
class Flow(Element):
    """Element that represents a flow."""

    name: str = ""
    decorators: List[Decorator] = field(default_factory=list)
    parameters: List[FlowParamDef] = field(default_factory=list)
    return_members: List[FlowReturnMemberDef] = field(default_factory=list)
    elements: List[Element] = field(default_factory=list)
    source_code: Optional[str] = None
    file_info: Dict[str, Any] = field(default_factory=dict)
    _type: str = "flow"


class SpecType(Enum):
    """The type of a spec."""

    EVENT = "event"
    ACTION = "action"
    FLOW = "flow"
    REFERENCE = "reference"


@dataclass_json
@dataclass
class Spec(Element):
    """Element that represents a spec.

    A spec can represent a flow, an action or an event. Currently, we will determine
    at runtime what the spec is.

    A spec is either specified directly through a name and arguments, or through a
    variable. In both cases, additional members can be accessed, e.g., .Finished().
    """

    name: Optional[str] = None
    spec_type: SpecType = SpecType.EVENT
    arguments: Dict[str, Any] = field(default_factory=dict)

    # Members on the spec, e.g. ActionX().Member()
    members: Optional[List["Spec"]] = None

    # Is set if it is a reference
    var_name: Optional[str] = None

    # The reference that should be captured.
    ref: Optional[Union[str, dict]] = field(default=None)

    _type: str = "spec"


@dataclass_json
@dataclass
class SpecAnd(Element):
    """A conjunction of specs."""

    specs: List[Spec] = field(default_factory=list)
    # TODO: Refactor to include this
    # elements: List[Spec] = field(default_factory=list)
    _type: str = "spec_and"


@dataclass_json
@dataclass
class SpecOr(Element):
    """A disjunction of spects."""

    specs: List[Spec] = field(default_factory=list)
    # TODO: Refactor to include this
    # elements: List[Spec] = field(default_factory=list)
    _type: str = "spec_or"


SpecElementType = Union[Spec, SpecAnd, SpecOr]


# TODO: Use this to replace the _type attribute in SpecOp
class SpecOpType(Enum):
    """All internal event types."""

    NONE = "none"
    SEND = "send"
    MATCH = "match"
    START = "start"
    STOP = "stop"
    AWAIT = "await"
    ACTIVATE = "activate"
    DEACTIVATE = "deactivate"


@dataclass_json
@dataclass
class SpecOp(Element):
    """An operation performed on a spec.

    Valid operations are: send, match, start, stop, await, activate, deactivate.
    """

    op: str = ""
    # TODO: refactor this removing dict and use SpecAnd and SpecOr instead
    spec: Union[Spec, dict] = field(default_factory=Spec)
    # spec: Union[Spec, SpecAnd, SpecOr] = Spec()

    # If the return value of the spec needs to be captured. The return value only makes sense
    # for await on flows and actions.
    # For compatibility, the return value in all other cases is the same value as the ref.
    # TODO: or should it just be None?
    return_var_name: Optional[str] = None

    # Attribute that can carry SpecOp specific information, currently only used for
    # 'match' operations that should not be considered for a flow to have started: info['internal'] = True
    info: dict = field(default_factory=dict)

    _type: str = "spec_op"


@dataclass_json
@dataclass
class If(Element):
    expression: str = ""
    then_elements: List[ElementType] = field(default_factory=list)
    else_elements: Optional[List[ElementType]] = None
    _type: str = "if"


@dataclass_json
@dataclass
class When(Element):
    when_specs: List[SpecElementType] = field(default_factory=list)
    then_elements: List[List[ElementType]] = field(default_factory=list)
    else_elements: Optional[List[ElementType]] = None
    _type: str = "when"


@dataclass_json
@dataclass
class While(Element):
    expression: str = ""
    elements: List[ElementType] = field(default_factory=list)
    _type: str = "while"


@dataclass_json
@dataclass
class Assignment(Element):
    key: str = ""
    expression: str = ""
    _type: str = "assignment"


@dataclass_json
@dataclass
class Continue(Element):
    label: Optional[str] = None
    _type: str = "continue"


@dataclass_json
@dataclass
class Abort(Element):
    _type: str = "abort"


@dataclass_json
@dataclass
class Break(Element):
    label: Optional[str] = None
    _type: str = "break"


@dataclass_json
@dataclass
class Return(Element):
    expression: Optional[str] = None
    _type: str = "return"


@dataclass_json
@dataclass
class Label(Element):
    name: str = ""
    _type: str = "label"


@dataclass_json
@dataclass
class Goto(Element):
    """Element for navigating to a label.

    If the expression is not True, just skips to the next element.
    """

    label: str = ""
    expression: str = "True"
    _type: str = "goto"


@dataclass_json
@dataclass
class Meta(Element):
    meta: dict = field(default_factory=dict)
    _type: str = "meta"


@dataclass_json
@dataclass
class ForkHead(Element):
    """Element to fork the current head into multiple heads.

    For each label a new head.
    """

    fork_uid: str = ""
    labels: List[str] = field(default_factory=list)
    _type: str = "_fork"


@dataclass_json
@dataclass
class MergeHeads(Element):
    """Merge heads based on uid from same flow.

    Only one head will advance from this element.
    """

    _type: str = "_merge"
    fork_uid: str = ""


@dataclass_json
@dataclass
class WaitForHeads(Element):
    """Wait for a number of heads.

    Once enough heads have reached this element they will be merged
    and only one head will advance.
    """

    number: int = 1
    _type: str = "_wait_for_heads"


@dataclass_json
@dataclass
class Priority(Element):
    """Set the priority of the flow in terms of action resolution."""

    priority_expr: str = "1.0"
    _type: str = "_priority"


@dataclass_json
@dataclass
class Global(Element):
    """Declare a global variable."""

    name: str = ""
    _type: str = "_global"


@dataclass_json
@dataclass
class Log(Element):
    """Log a message."""

    info: str = ""
    _type: str = "_log"


@dataclass_json
@dataclass
class Print(Element):
    """Print a message."""

    info: str = ""
    _type: str = "_print"


@dataclass_json
@dataclass
class CatchPatternFailure(Element):
    """Set the pattern failure catcher to catch a failing pattern and forward it to the specified label.
    If the label is not set the failure will be propagated to the parent flow.
    """

    label: Optional[str] = None
    _type: str = "_catch_pattern_failure"


@dataclass_json
@dataclass
class BeginScope(Element):
    """Defines the start of a new scope in a flow to keep track of all started actions/flows."""

    name: str = ""
    _type: str = "_begin_scope"


@dataclass_json
@dataclass
class EndScope(Element):
    """Defines the end of a scope in a flow that will stop all active actions/flows that have been started
    within that scope.
    """

    name: str = ""
    _type: str = "_end_scope"
