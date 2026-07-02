# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Module for the calling proper action endpoints based on events received at action server endpoint"""

import importlib.util
import inspect
import logging
import os
from collections.abc import Iterator, Mapping
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol, Tuple, Type, TypeAlias, Union, cast

from nemoguardrails import utils
from nemoguardrails.exceptions import LLMCallException
from nemoguardrails.manifests import ActionRef, RailCatalog, default_rail_catalog, resolve_import_ref

log = logging.getLogger(__name__)


class AsyncInvokableAction(Protocol):
    def ainvoke(self, *args: Any, **kwargs: Any) -> Awaitable[Any]: ...


class RunnableAction(Protocol):
    def run(self, *args: Any, **kwargs: Any) -> Any: ...


RegisteredAction: TypeAlias = Union[
    Callable[..., Any],
    Type[Any],
    AsyncInvokableAction,
    RunnableAction,
]


class _RegisteredActions(Mapping[str, RegisteredAction]):
    def __init__(self, dispatcher: "ActionDispatcher") -> None:
        self._dispatcher = dispatcher

    def __getitem__(self, name: str) -> RegisteredAction:
        action = self._dispatcher.get_action(name)
        if action is None:
            raise KeyError(name)
        return action

    def __iter__(self) -> Iterator[str]:
        return iter(self._dispatcher._action_names())

    def __len__(self) -> int:
        return len(self._dispatcher._action_names())


class ActionDispatcher:
    def __init__(
        self,
        load_all_actions: bool = True,
        config_path: Optional[str] = None,
        import_paths: Optional[List[str]] = None,
        rail_catalog: Optional[RailCatalog] = None,
    ):
        """
        Initializes an actions dispatcher.
        Args:
            load_all_actions (bool, optional): When set to True, it loads all actions in the
                'actions' folder both in the current working directory and in the package.
            config_path (str, optional): The path from which the configuration was loaded.
                If there are actions at the specified path, it loads them as well.
            import_paths (List[str], optional): Additional imported paths from which actions
                should be loaded.
        """
        log.info("Initializing action dispatcher")

        self._registered_actions: Dict[str, RegisteredAction] = {}
        self._lazy_action_refs: Dict[str, ActionRef] = {}
        self._manifested_library_paths: set[Path] = set()
        self._rail_catalog = rail_catalog
        self._registered_actions_view = _RegisteredActions(self)

        if load_all_actions:
            if self._rail_catalog is None:
                self._rail_catalog = default_rail_catalog()
            # TODO: check for better way to find actions dir path or use constants.py
            current_file_path = Path(__file__).resolve()
            parent_directory_path = current_file_path.parents[1]

            # First, we load all actions from the actions folder
            self.load_actions_from_path(parent_directory_path)
            # self.load_actions_from_path(os.path.join(os.path.dirname(__file__), ".."))

            library_path = parent_directory_path / "library"
            self._register_manifest_action_refs()
            self._load_unmanifested_library_actions(library_path)

            # Next, we load all actions from the current working directory
            # TODO: add support for an explicit ACTIONS_PATH
            self.load_actions_from_path(Path.cwd())

            # Last, but not least, if there was a config path, we try to load actions
            # from there as well.
            if config_path:
                split_config_path: List[str] = config_path.split(",")

                # Don't load actions if we have an empty list
                if split_config_path:
                    for path in split_config_path:
                        self.load_actions_from_path(Path(path.strip()))

            # If there are any imported paths, we load the actions from there as well.
            if import_paths:
                for import_path in import_paths:
                    self.load_actions_from_path(Path(import_path.strip()))

        log.info(f"Registered Actions :: {sorted(self.get_registered_actions())}")
        log.info("Action dispatcher initialized")

    @property
    def registered_actions(self):
        """
        Gets the dictionary of registered actions.
        Returns:
            dict: A dictionary where keys are action names and values are callable action functions.
        """
        return self._registered_actions_view

    def load_actions_from_path(self, path: Path):
        """Loads all actions from the specified path.

        This method loads all actions from the `actions.py` file if it exists and
        all actions inside the `actions` folder if it exists.

        Args:
            path (str): A string representing the path from which to load actions.

        """
        actions_path = path / "actions"
        if os.path.exists(actions_path):
            actions = self._find_actions(actions_path)
            self._registered_actions.update(actions)
            for action_name in actions:
                self._lazy_action_refs.pop(action_name, None)

        actions_py_path = os.path.join(path, "actions.py")
        if os.path.exists(actions_py_path):
            actions = self._load_actions_from_module(actions_py_path)
            self._registered_actions.update(actions)
            for action_name in actions:
                self._lazy_action_refs.pop(action_name, None)

    def register_action(self, action: RegisteredAction, name: Optional[str] = None, override: bool = True):
        """Registers an action with the given name.

        Args:
            action (RegisteredAction): The action function or runnable object.
            name (Optional[str]): The name of the action. Defaults to None.
            override (bool): If an action already exists, whether it should be overridden or not.
        """
        if name is None:
            action_meta = getattr(action, "action_meta", None)
            action_name = action_meta["name"] if action_meta else getattr(action, "__name__", None)
            if not isinstance(action_name, str) or not action_name:
                raise ValueError("An explicit name is required for actions without __name__.")
        else:
            action_name = name

        # If we're not allowed to override, we stop.
        if action_name in self._registered_actions or action_name in self._lazy_action_refs:
            if not override:
                return

        self._lazy_action_refs.pop(action_name, None)
        self._registered_actions[action_name] = action

    def _has_action_name(self, action_name: str) -> bool:
        return action_name in self._registered_actions or action_name in self._lazy_action_refs

    def _action_names(self) -> tuple[str, ...]:
        return tuple(self._registered_actions) + tuple(
            name for name in self._lazy_action_refs if name not in self._registered_actions
        )

    def _register_action_ref(self, action_ref: ActionRef, override: bool = True) -> None:
        if self._has_action_name(action_ref.name) and not override:
            return
        self._registered_actions.pop(action_ref.name, None)
        self._lazy_action_refs[action_ref.name] = action_ref

    def register_actions(self, actions_obj: Any, override: bool = True):
        """Registers all the actions from the given object.

        Args:
            actions_obj (any): The object containing actions.
            override (bool): If an action already exists, whether it should be overridden or not.
        """

        # Register the actions
        for attr in dir(actions_obj):
            val = getattr(actions_obj, attr)

            if hasattr(val, "action_meta"):
                self.register_action(val, override=override)

    def _register_manifest_action_refs(self) -> None:
        if self._rail_catalog is None:
            return
        for record in self._rail_catalog.records.values():
            manifest = record.manifest
            if manifest.actions is None:
                continue
            for action_ref in manifest.actions.refs:
                self._register_action_ref(action_ref)
            if record.built_in and manifest.origin:
                module = __import__(manifest.origin, fromlist=["__file__"])
                module_file = getattr(module, "__file__", None)
                if module_file:
                    self._manifested_library_paths.add(Path(module_file).parent.resolve())

    def _load_unmanifested_library_actions(self, library_path: Path) -> None:
        for root, dirs, files in os.walk(library_path):
            if Path(root).resolve() in self._manifested_library_paths:
                continue
            if "actions" in dirs or "actions.py" in files:
                self.load_actions_from_path(Path(root))

    def _resolve_registered_action(self, action_name: str) -> Optional[RegisteredAction]:
        action = self._registered_actions.get(action_name)
        if action is None:
            action_ref = self._lazy_action_refs.get(action_name)
            if action_ref is None:
                return None
            resolved_action = resolve_import_ref(action_ref)
            action_meta = getattr(resolved_action, "action_meta", {})
            if action_meta and action_meta.get("name") != action_ref.name:
                raise ValueError(
                    f"Lazy action ref {action_ref.name!r} resolved to action "
                    f"{action_meta.get('name')!r} from {action_ref.target!r}."
                )
            self._registered_actions[action_name] = resolved_action
            self._lazy_action_refs.pop(action_name, None)
            return cast(RegisteredAction, resolved_action)

        return action

    def _normalize_action_name(self, name: str) -> str:
        """Normalize the action name to the required format."""
        if not self._has_action_name(name):
            if name.endswith("Action"):
                name = name.replace("Action", "")
            name = utils.camelcase_to_snakecase(name)
        return name

    def has_registered(self, name: str) -> bool:
        """Check if an action is registered."""
        name = self._normalize_action_name(name)
        return self._has_action_name(name)

    def get_action(self, name: str) -> Optional[RegisteredAction]:
        """Get the registered action by name.

        Args:
            name (str): The name of the action.

        Returns:
            callable: The registered action.
        """
        name = self._normalize_action_name(name)
        return self._resolve_registered_action(name)

    async def execute_action(
        self, action_name: str, params: Dict[str, Any]
    ) -> Tuple[Union[Optional[str], Dict[str, Any]], str]:
        """Execute a registered action.

        Args:
            action_name (str): The name of the action to execute.
            params (Dict[str, Any]): Parameters for the action.

        Returns:
            Tuple[Union[str, Dict[str, Any]], str]: A tuple containing the result and status.
        """

        action_name = self._normalize_action_name(action_name)

        if self._has_action_name(action_name):
            log.info("Executing registered action: %s", action_name)
            maybe_fn = self._resolve_registered_action(action_name)
            if not maybe_fn:
                raise Exception(f"Action '{action_name}' is not registered.")

            fn: Any = maybe_fn
            # Actions that are registered as classes are initialized lazy, when
            # they are first used.
            if inspect.isclass(fn):
                fn = fn()
                self._registered_actions[action_name] = fn

            if fn:
                try:
                    # We support both functions and classes as actions
                    if inspect.isfunction(fn) or inspect.ismethod(fn):
                        # We support both sync and async actions.
                        result = fn(**params)
                        if inspect.iscoroutine(result):
                            result = await result
                        else:
                            log.warning(f"Synchronous action `{action_name}` has been called.")

                    elif callable(ainvoke := getattr(fn, "ainvoke", None)):
                        # Duck-type check for LangChain Runnables (or any object
                        # with ainvoke) to avoid importing langchain in core.
                        async_invoke = cast(Callable[..., Awaitable[Any]], ainvoke)
                        result = await async_invoke(input=params)
                    else:
                        # TODO: there should be a common base class here
                        fn_run_func = getattr(fn, "run", None)
                        if not callable(fn_run_func):
                            raise Exception(f"No 'run' method defined for action '{action_name}'.")

                        fn_run_func_with_signature = cast(
                            Callable[[], Union[Optional[str], Dict[str, Any]]],
                            fn_run_func,
                        )
                        result = fn_run_func_with_signature(**params)
                    return result, "success"

                # We forward LLM Call exceptions
                except LLMCallException as e:
                    raise e

                except Exception as e:
                    filtered_params = {k: v for k, v in params.items() if k not in ["state", "events", "llm"]}
                    log.warning(
                        "Error while execution '%s' with parameters '%s': %s",
                        action_name,
                        filtered_params,
                        e,
                    )
                    log.exception(e)

        return None, "failed"

    def get_registered_actions(self) -> List[str]:
        """Get the list of available actions.

        Returns:
            List[str]: List of available actions.
        """
        return list(self._action_names())

    @staticmethod
    def _load_actions_from_module(filepath: str):
        """Loads the actions from the specified python module.

        Args:
            filepath (str): The path of the Python module.

        Returns:
            Dict: Dictionary of loaded actions.
        """
        action_objects = {}
        filename = os.path.basename(filepath)
        module = None

        if not os.path.isfile(filepath):
            log.error(f"{filepath} does not exist or is not a file.")
            log.error(f"Failed to load actions from {filename}.")
            return action_objects

        try:
            log.debug(f"Analyzing file {filename}")
            # Import the module from the file

            spec: Optional[ModuleSpec] = importlib.util.spec_from_file_location(filename, filepath)
            if not spec:
                log.error(f"Failed to create a module spec from {filepath}.")
                return action_objects

            module = importlib.util.module_from_spec(spec)
            if spec.loader:
                spec.loader.exec_module(module)

            # Loop through all members in the module and check for the `@action` decorator
            # If class has action decorator is_action class member is true
            for name, obj in inspect.getmembers(module):
                if (inspect.isfunction(obj) or inspect.isclass(obj)) and hasattr(obj, "action_meta"):
                    try:
                        actionable_name: str = getattr(obj, "action_meta").get("name")
                        action_objects[actionable_name] = obj
                        log.info(f"Added {actionable_name} to actions")
                    except Exception as e:
                        log.error(f"Failed to register {name} in action dispatcher due to exception {e}")
        except Exception as e:
            if module is None:
                raise RuntimeError(f"Failed to load actions from module at {filepath}.")
            if not module.__file__:
                raise RuntimeError(f"No file found for module {module} at {filepath}.")

            log.error(f"Failed to register {filename} in action dispatcher due to exception: {e}")

        return action_objects

    def _find_actions(self, directory) -> Dict:
        """Loop through all the subdirectories and check for the class with @action
        decorator and add in action_classes dict.

        Args:
            directory: The directory to search for actions.

        Returns:
            Dict: Dictionary of found actions.
        """
        action_objects = {}

        if not os.path.exists(directory):
            log.debug(f"_find_actions: {directory} does not exist.")
            return action_objects

        # Loop through all files in the directory and its subdirectories
        for root, _dirs, files in os.walk(directory):
            for filename in files:
                if filename.endswith(".py"):
                    filepath = os.path.join(root, filename)
                    if is_action_file(filepath):
                        action_objects.update(ActionDispatcher._load_actions_from_module(filepath))
        if not action_objects:
            log.debug(f"No actions found in {directory}")
            log.exception(f"No actions found in the directory {directory}.")

        return action_objects


def is_action_file(filepath):
    """Heuristics for determining if a Python file can have actions or not.

    Currently, it only excludes the `__init__.py files.
    """
    if "__init__.py" in filepath:
        return False

    return True
