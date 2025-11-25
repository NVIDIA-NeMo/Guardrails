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

"""Module for the calling proper action endpoints based on events received at action server endpoint"""

import importlib.util
import inspect
import logging
import os
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

from langchain_core.runnables import Runnable

from nemoguardrails import utils
from nemoguardrails.actions.llm.utils import LLMCallException

log = logging.getLogger(__name__)


class ActionDispatcher:
    def __init__(
        self,
        load_all_actions: bool = True,
        config_path: Optional[str] = None,
        import_paths: Optional[List[str]] = None,
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

        self._registered_actions: Dict[str, Union[Type, Callable[..., Any]]] = {}

        if load_all_actions:
            # TODO: check for better way to find actions dir path or use constants.py
            current_file_path = Path(__file__).resolve()
            parent_directory_path = current_file_path.parents[1]

            # First, we load all actions from the actions folder
            self.load_actions_from_path(parent_directory_path)
            # self.load_actions_from_path(os.path.join(os.path.dirname(__file__), ".."))

            # Next, we load all actions from the library folder
            library_path = parent_directory_path / "library"

            for root, dirs, files in os.walk(library_path):
                # We only load the actions if there is an `actions` sub-folder or
                # an `actions.py` file.
                if "actions" in dirs or "actions.py" in files:
                    self.load_actions_from_path(Path(root))

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

        log.info(f"Registered Actions :: {sorted(self._registered_actions.keys())}")
        log.info("Action dispatcher initialized")

    @property
    def registered_actions(self):
        """
        Gets the dictionary of registered actions.
        Returns:
            dict: A dictionary where keys are action names and values are callable action functions.
        """
        return self._registered_actions

    def load_actions_from_path(self, path: Path):
        """Loads all actions from the specified path.

        This method loads all actions from the `actions.py` file if it exists and
        all actions inside the `actions` folder if it exists.

        Args:
            path (str): A string representing the path from which to load actions.

        """
        actions_path = path / "actions"
        if os.path.exists(actions_path):
            self._registered_actions.update(self._find_actions(actions_path))

        actions_py_path = os.path.join(path, "actions.py")
        if os.path.exists(actions_py_path):
            self._registered_actions.update(self._load_actions_from_module(actions_py_path))

    def register_action(self, action: Callable, name: Optional[str] = None, override: bool = True):
        """Registers an action with the given name.

        Args:
            action (Callable): The action function.
            name (Optional[str]): The name of the action. Defaults to None.
            override (bool): If an action already exists, whether it should be overridden or not.
        """
        if name is None:
            action_meta = getattr(action, "action_meta", None)
            action_name = action_meta["name"] if action_meta else action.__name__
        else:
            action_name = name

        # If we're not allowed to override, we stop.
        if action_name in self._registered_actions and not override:
            return

        self._registered_actions[action_name] = action

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

    def _normalize_action_name(self, name: str) -> str:
        """Normalize the action name to the required format."""
        if name not in self.registered_actions:
            if name.endswith("Action"):
                name = name.replace("Action", "")
            name = utils.camelcase_to_snakecase(name)
        return name

    def has_registered(self, name: str) -> bool:
        """Check if an action is registered."""
        name = self._normalize_action_name(name)
        return name in self.registered_actions

    def get_action(self, name: str) -> Optional[Callable]:
        """Get the registered action by name.

        Args:
            name (str): The name of the action.

        Returns:
            callable: The registered action.
        """
        name = self._normalize_action_name(name)
        return self._registered_actions.get(name, None)

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

        if action_name in self._registered_actions:
            log.info("Executing registered action: %s", action_name)
            maybe_fn: Optional[Callable] = self._registered_actions.get(action_name, None)
            if not maybe_fn:
                raise Exception(f"Action '{action_name}' is not registered.")

            fn = cast(Callable, maybe_fn)
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

                    elif isinstance(fn, Runnable):
                        # If it's a Runnable, we invoke it as well
                        runnable = fn

                        result = await runnable.ainvoke(input=params)
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
        return list(self._registered_actions.keys())

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

            try:
                relative_filepath = Path(module.__file__).relative_to(Path.cwd())
            except ValueError:
                relative_filepath = Path(module.__file__).resolve()
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
        for root, dirs, files in os.walk(directory):
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
