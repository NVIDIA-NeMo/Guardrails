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

"""Module for the calling proper action endpoints based on events received at action server endpoint"""

import importlib.util
import inspect
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain.chains.base import Chain
from langchain_core.runnables import Runnable

from nemoguardrails import utils
from nemoguardrails.actions.llm.utils import LLMCallException
from nemoguardrails.logging.callbacks import logging_callbacks

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

        self._registered_actions = {}

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
                config_path = config_path.split(",")
                for path in config_path:
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
            self._registered_actions.update(
                self._load_actions_from_module(actions_py_path)
            )

    def register_action(
        self, action: callable, name: Optional[str] = None, override: bool = True
    ):
        """Registers an action with the given name.

        Args:
            action (callable): The action function.
            name (Optional[str]): The name of the action. Defaults to None.
            override (bool): If an action already exists, whether it should be overridden or not.
        """
        if name is None:
            action_meta = getattr(action, "action_meta", None)
            name = action_meta["name"] if action_meta else action.__name__

        # If we're not allowed to override, we stop.
        if name in self._registered_actions and not override:
            return

        self._registered_actions[name] = action

    def register_actions(self, actions_obj: any, override: bool = True):
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

    def get_action(self, name: str) -> callable:
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
    ) -> Tuple[Union[str, Dict[str, Any]], str]:
        """Execute a registered action.

        Args:
            action_name (str): The name of the action to execute.
            params (Dict[str, Any]): Parameters for the action.

        Returns:
            Tuple[Union[str, Dict[str, Any]], str]: A tuple containing the result and status.
        """

        action_name = self._normalize_action_name(action_name)

        if action_name in self._registered_actions:
            log.info(f"Executing registered action: {action_name}")
            fn = self._registered_actions.get(action_name, None)

            # Actions that are registered as classes are initialized lazy, when
            # they are first used.
            if inspect.isclass(fn):
                fn = fn()
                self._registered_actions[action_name] = fn

            if fn is not None:
                try:
                    # We support both functions and classes as actions
                    if inspect.isfunction(fn) or inspect.ismethod(fn):
                        # We support both sync and async actions.
                        result = fn(**params)
                        if inspect.iscoroutine(result):
                            result = await result
                        else:
                            log.warning(
                                f"Synchronous action `{action_name}` has been called."
                            )

                    elif isinstance(fn, Chain):
                        try:
                            chain = fn

                            # For chains with only one output key, we use the `arun` function
                            # to return directly the result.
                            if len(chain.output_keys) == 1:
                                result = await chain.arun(
                                    **params, callbacks=logging_callbacks
                                )
                            else:
                                # Otherwise, we return the dict with the output keys.
                                result = await chain.acall(
                                    inputs=params,
                                    return_only_outputs=True,
                                    callbacks=logging_callbacks,
                                )
                        except NotImplementedError:
                            # Not ideal, but for now we fall back to sync execution
                            # if the async is not available
                            result = fn.run(**params)
                    elif isinstance(fn, Runnable):
                        # If it's a Runnable, we invoke it as well
                        runnable = fn

                        result = await runnable.ainvoke(input=params)
                    else:
                        # TODO: there should be a common base class here
                        result = fn.run(**params)
                    return result, "success"

                # We forward LLM Call exceptions
                except LLMCallException as e:
                    raise e

                except Exception as e:
                    filtered_params = {
                        k: v
                        for k, v in params.items()
                        if k not in ["state", "events", "llm"]
                    }
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

        if not os.path.isfile(filepath):
            log.error(f"{filepath} does not exist or is not a file.")
            log.error(f"Failed to load actions from {filename}.")
            return action_objects

        try:
            log.debug(f"Analyzing file {filename}")
            # Import the module from the file

            spec = importlib.util.spec_from_file_location(filename, filepath)
            if spec is None:
                log.error(f"Failed to create a module spec from {filepath}.")
                return action_objects

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Loop through all members in the module and check for the `@action` decorator
            # If class has action decorator is_action class member is true
            for name, obj in inspect.getmembers(module):
                if (inspect.isfunction(obj) or inspect.isclass(obj)) and hasattr(
                    obj, "action_meta"
                ):
                    try:
                        action_objects[obj.action_meta["name"]] = obj
                        log.info(f"Added {obj.action_meta['name']} to actions")
                    except Exception as e:
                        log.error(
                            f"Failed to register {obj.action_meta['name']} in action dispatcher due to exception {e}"
                        )
        except Exception as e:
            try:
                relative_filepath = Path(module.__file__).relative_to(Path.cwd())
            except ValueError:
                relative_filepath = Path(module.__file__).resolve()
            log.error(
                f"Failed to register {filename} from {relative_filepath} in action dispatcher due to exception: {e}"
            )

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
                        action_objects.update(
                            ActionDispatcher._load_actions_from_module(filepath)
                        )
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
