# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Protocol,
    Type,
    TypedDict,
    TypeVar,
    Union,
    cast,
)


class ActionMeta(TypedDict):
    name: str
    is_system_action: bool
    execute_async: bool
    output_mapping: Optional[Callable[[Any], bool]]


# Create a TypeVar to represent the decorated function or class
T = TypeVar("T", bound=Union[Callable[..., Any], Type[Any]])


def action(
    is_system_action: bool = False,
    name: Optional[str] = None,
    execute_async: bool = False,
    output_mapping: Optional[Callable[[Any], bool]] = None,
) -> Callable[[T], T]:
    """Decorator to mark a function or class as an action.

    Args:
        is_system_action (bool): Flag indicating if the action is a system action.
        name (str): The name to associate with the action.
        execute_async: Whether the function should be executed in async mode.
        output_mapping (Optional[Callable[[Any], bool]]): A function to interpret the action's result.
            It accepts the return value (e.g. the first element of a tuple) and return True if the output
            is not safe.

    Returns:
        callable: The decorated function or class.
    """

    def decorator(fn_or_cls: Union[Callable, Type]) -> Union[Callable, Type]:
        """Inner decorator function to add metadata to the action.

        Args:
            fn_or_cls: The function or class being decorated.
        """
        fn_or_cls_target = getattr(fn_or_cls, "__func__", fn_or_cls)

        # Action name is optional for the decorator, but mandatory for ActionMeta TypedDict
        action_name: str = cast(str, name or fn_or_cls.__name__)

        action_meta: ActionMeta = {
            "name": action_name,
            "is_system_action": is_system_action,
            "execute_async": execute_async,
            "output_mapping": output_mapping,
        }

        setattr(fn_or_cls_target, "action_meta", action_meta)
        return fn_or_cls

    return decorator  # pyright: ignore (TODO - resolve how the Actionable Protocol doesn't resolve the issue)


@dataclass
class ActionResult:
    """Data class representing the result of an action.

    Attributes:
        return_value (Optional[Any]): The value returned by the action.
        events (Optional[List[dict]]): The events to be added to the stream.
        context_updates (Optional[dict]): Updates made to the context by this action.
    """

    # The value returned by the action
    return_value: Optional[Any] = None

    # The events that should be added to the stream
    events: Optional[List[dict]] = None

    # The updates made to the context by this action
    context_updates: Optional[dict] = field(default_factory=dict)
