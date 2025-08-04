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
import asyncio
import inspect
import logging
import uuid
from textwrap import indent
from time import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import aiohttp
from langchain.chains.base import Chain

from nemoguardrails.actions.actions import ActionResult
from nemoguardrails.actions.core import create_event
from nemoguardrails.actions.output_mapping import is_output_blocked
from nemoguardrails.colang import parse_colang_file
from nemoguardrails.colang.runtime import Runtime
from nemoguardrails.colang.v1_0.runtime.flows import (
    FlowConfig,
    _get_flow_params,
    _normalize_flow_id,
    compute_context,
    compute_next_steps,
)
from nemoguardrails.logging.processing_log import processing_log_var
from nemoguardrails.utils import new_event_dict, new_uuid

log = logging.getLogger(__name__)


class RuntimeV1_0(Runtime):
    """Runtime for executing the guardrails."""

    def _load_flow_config(self, flow: dict):
        """
        Load a flow configuration.

        Args:
            flow (dict): The flow data.

        Returns:
            None
        """

        # If we don't have an id, we generate a random UID.
        flow_id = flow.get("id") or new_uuid()

        # If the flow already exists, we stop.
        # This allows us to override flows. The order in which the flows
        # are in the config is such that the first ones are the ones that
        # should be kept.
        if flow_id in self.flow_configs:
            return

        elements = flow["elements"]

        # If we have an element with meta information, we move the relevant properties
        # to top level.
        if elements and elements[0].get("_type") == "meta":
            meta_data = elements[0]["meta"]

            if "priority" in meta_data:
                flow["priority"] = meta_data["priority"]
            if "is_extension" in meta_data:
                flow["is_extension"] = meta_data["is_extension"]
            if "interruptable" in meta_data:
                flow["is_interruptible"] = meta_data["interruptable"]
            if meta_data.get("subflow"):
                flow["is_subflow"] = True
            if meta_data.get("allow_multiple"):
                flow["allow_multiple"] = True

            # Finally, remove the meta element
            elements = elements[1:]

        self.flow_configs[flow_id] = FlowConfig(
            id=flow_id,
            elements=elements,
            priority=flow.get("priority", 1.0),
            is_extension=flow.get("is_extension", False),
            is_interruptible=flow.get("is_interruptible", True),
            is_subflow=flow.get("is_subflow", False),
            source_code=flow.get("source_code"),
            allow_multiple=flow.get("allow_multiple", False),
        )

        # We also compute what types of events can trigger this flow, in addition
        # to the default ones.
        for element in elements:
            if element.get("UtteranceUserActionFinished"):
                self.flow_configs[flow_id].trigger_event_types.append(
                    "UtteranceUserActionFinished"
                )

            # If a flow creates a type of event, we also allow it to trigger the event.
            if (
                element["_type"] == "run_action"
                and element["action_name"] == "create_event"
            ):
                event_type = element["action_params"]["event"]["_type"]
                self.flow_configs[flow_id].trigger_event_types.append(event_type)

    def _init_flow_configs(self):
        """
        Initialize the flow configurations.

        Returns:
            None
        """
        self.flow_configs = {}

        for flow in self.config.flows:
            self._load_flow_config(flow)

    async def generate_events(
        self, events: List[dict], processing_log: Optional[List[dict]] = None
    ) -> List[dict]:
        """Generates the next events based on the provided history.

        This is a wrapper around the `process_events` method, that will keep
        processing the events until the `listen` event is produced.

        Args:
            events (List[dict]): The list of events.
            processing_log (Optional[List[dict]]): The processing log so far. This will be mutated.

        Returns:
            List[dict]: The list of generated events.
        """
        events = events.copy()
        new_events = []
        if processing_log is None:
            processing_log = []

        # We record the processing log in the async context.
        # This is needed to automatically record the LLM calls.
        processing_log_var.set(processing_log)

        processing_log.append(
            {"type": "event", "timestamp": time(), "data": events[-1]}
        )

        while True:
            last_event = events[-1]

            log.info("Processing event: %s", last_event)

            event_type = last_event["type"]
            log.info(
                "Event :: %s %s",
                event_type,
                str({k: v for k, v in last_event.items() if k != "type"}),
            )

            # If we need to execute an action, we start doing that.
            if last_event["type"] == "StartInternalSystemAction":
                next_events = await self._process_start_action(events)

            # If we need to start a flow, we parse the content and register it.
            elif last_event["type"] == "start_flow" and last_event.get("flow_body"):
                next_events = await self._process_start_flow(
                    events, processing_log=processing_log
                )

            else:
                # We need to slide all the flows based on the current event,
                # to compute the next steps.
                next_events = await self._compute_next_steps(
                    events, processing_log=processing_log
                )

                if len(next_events) == 0:
                    next_events = [new_event_dict("Listen")]

            # Otherwise, we append the event and continue the processing.
            events.extend(next_events)
            new_events.extend(next_events)

            for event in next_events:
                if event["type"] != "EventHistoryUpdate":
                    processing_log.append(
                        {"type": "event", "timestamp": time(), "data": event}
                    )

            # If the next event is a listen, we stop the processing.
            if next_events[-1]["type"] == "Listen":
                break

            # As a safety measure, we stop the processing if we have too many events.
            if len(new_events) > 300:
                raise Exception("Too many events.")

        # Unpack and insert events in event history update event if available
        temp_events = []
        for event in new_events:
            if event["type"] == "EventHistoryUpdate":
                temp_events.extend(
                    [e for e in event["data"]["events"] if e["type"] != "Listen"]
                )
            else:
                temp_events.append(event)
        new_events = temp_events

        return new_events

    async def _compute_next_steps(
        self, events: List[dict], processing_log: List[dict]
    ) -> List[dict]:
        """
        Compute the next steps based on the current flow.

        Args:
            events (List[dict]): The list of events.
            processing_log (List[dict]): The processing log so far. This will be mutated.

        Returns:
            List[dict]: The list of computed next steps.
        """
        next_steps = compute_next_steps(
            events,
            self.flow_configs,
            rails_config=self.config,
            processing_log=processing_log,
        )

        # If there are any StartInternalSystemAction events, we mark if they are system actions or not
        for event in next_steps:
            if event["type"] == "StartInternalSystemAction":
                is_system_action = False
                fn = self.action_dispatcher.get_action(event["action_name"])
                if fn:
                    action_meta = getattr(fn, "action_meta", {})
                    is_system_action = action_meta.get("is_system_action", False)
                event["is_system_action"] = is_system_action

        return next_steps

    @staticmethod
    def _internal_error_action_result(message: str):
        """
        Helper to construct an action result for an internal error.

        Args:
            message (str): The error message.

        Returns:
            ActionResult: The action result.
        """
        return ActionResult(
            events=[
                {
                    "type": "BotIntent",
                    "intent": "inform internal error occurred",
                },
                {
                    "type": "StartUtteranceBotAction",
                    "script": message,
                },
                # We also want to hide this from now from the history moving forward
                {"type": "hide_prev_turn"},
                # Stop execution to prevent further LLM generation after internal error
                {
                    "type": "BotIntent",
                    "intent": "stop",
                },
            ]
        )

    async def _run_flows_in_parallel(
        self,
        flows: List[str],
        events: List[dict],
        pre_events: Optional[List[dict]] = None,
        post_events: Optional[List[dict]] = None,
    ) -> ActionResult:
        """
        Run flows in parallel.

        Running flows in parallel is done by triggering a separate event loop with a `start_flow` event for each flow, in the context of the current event loop.

        Args:
            flows (List[str]): The list of flow names to run in parallel.
            events (List[dict]): The current events.
            pre_events (List[dict], optional): Events to be added before starting each flow.
            post_events (List[dict], optional): Events to be added after finishing each flow.
        """

        if pre_events is not None and len(pre_events) != len(flows):
            raise ValueError("Number of pre-events must match number of flows.")
        if post_events is not None and len(post_events) != len(flows):
            raise ValueError("Number of post-events must match number of flows.")

        unique_flow_ids = {}  # Keep track of unique flow IDs order
        task_results: Dict[str, List] = {}  # Store results keyed by flow_id
        task_processing_logs: dict = {}  # Store resulting processing logs for each flow

        # Wrapper function to help reverse map the task result to the flow ID
        async def task_call_helper(flow_uid, post_event, func, *args, **kwargs):
            result = await func(*args, **kwargs)
            if post_event:
                result.append(post_event)
                args[1].append(
                    {"type": "event", "timestamp": time(), "data": post_event}
                )
            return flow_uid, result

        # Create a task for each flow but don't await them yet
        tasks = []
        for index, flow_name in enumerate(flows):
            # Copy the events to avoid modifying the original list
            _events = events.copy()

            flow_params = _get_flow_params(flow_name)
            flow_id = _normalize_flow_id(flow_name)

            if flow_params:
                _events.append(
                    {"type": "start_flow", "flow_id": flow_id, "params": flow_params}
                )
            else:
                _events.append({"type": "start_flow", "flow_id": flow_id})

            # Generate a unique flow ID
            flow_uid = f"{flow_id}:{str(uuid.uuid4())}"

            # Initialize task results and processing logs for this flow
            task_results[flow_uid] = []
            task_processing_logs[flow_uid] = []

            # Add pre-event if provided
            if pre_events:
                task_results[flow_uid].append(pre_events[index])
                task_processing_logs[flow_uid].append(
                    {"type": "event", "timestamp": time(), "data": pre_events[index]}
                )

            task = asyncio.create_task(
                task_call_helper(
                    flow_uid,
                    post_events[index] if post_events else None,
                    self.generate_events,
                    _events,
                    task_processing_logs[flow_uid],
                )
            )
            tasks.append(task)
            unique_flow_ids[flow_uid] = task

        stopped_task_results: List[dict] = []

        # Process tasks as they complete using as_completed
        try:
            for future in asyncio.as_completed(tasks):
                try:
                    (flow_id, result) = await future

                    # Check if this rail requested to stop
                    has_stop = any(
                        event["type"] == "BotIntent" and event["intent"] == "stop"
                        for event in result
                    )

                    # If this flow had a stop event
                    if has_stop:
                        stopped_task_results = task_results[flow_id] + result

                        # Cancel all remaining tasks
                        for pending_task in tasks:
                            # Don't include results and processing logs for cancelled or stopped tasks
                            if (
                                pending_task != unique_flow_ids[flow_id]
                                and not pending_task.done()
                            ):
                                # Cancel the task if it is not done
                                pending_task.cancel()
                                # Find the flow_uid for this task and remove it from the dict
                                for k, v in list(unique_flow_ids.items()):
                                    if v == pending_task:
                                        del unique_flow_ids[k]
                                        break
                        del unique_flow_ids[flow_id]
                        break
                    else:
                        # Store the result for this specific flow
                        task_results[flow_id].extend(result)

                except asyncio.exceptions.CancelledError:
                    pass

        except Exception as e:
            log.error(f"Error in parallel rail execution: {str(e)}")
            raise
        finally:
            # clean up any remaining cancelled tasks to avoid "Task was destroyed but it is pending" warnings
            for task in tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

        context_updates: dict = {}
        processing_log = processing_log_var.get()

        finished_task_processing_logs: List[dict] = []  # Collect all results in order
        finished_task_results: List[dict] = []  # Collect all results in order

        # Compose results in original flow order of all completed tasks
        for flow_id in unique_flow_ids:
            result = task_results[flow_id]

            # Extract context updates
            for event in result:
                if event["type"] == "ContextUpdate":
                    context_updates = {**context_updates, **event["data"]}

            finished_task_results.extend(result)
            finished_task_processing_logs.extend(task_processing_logs[flow_id])

        if processing_log:
            for plog in finished_task_processing_logs:
                # Filter out "Listen" and "start_flow" events from task processing log
                if plog["type"] == "event" and (
                    plog["data"]["type"] == "Listen"
                    or plog["data"]["type"] == "start_flow"
                ):
                    continue
                processing_log.append(plog)

        # We pack all events into a single event to add it to the event history.
        history_events = new_event_dict(
            "EventHistoryUpdate",
            data={"events": finished_task_results},
        )

        return ActionResult(
            events=[history_events] + stopped_task_results,
            context_updates=context_updates,
        )

    async def _run_input_rails_in_parallel(
        self, flows: List[str], events: List[dict]
    ) -> ActionResult:
        """Run the input rails in parallel."""
        pre_events = [
            (await create_event({"_type": "StartInputRail", "flow_id": flow})).events[0]
            for flow in flows
        ]
        post_events = [
            (
                await create_event({"_type": "InputRailFinished", "flow_id": flow})
            ).events[0]
            for flow in flows
        ]

        return await self._run_flows_in_parallel(
            flows=flows, events=events, pre_events=pre_events, post_events=post_events
        )

    async def _run_output_rails_in_parallel(
        self, flows: List[str], events: List[dict]
    ) -> ActionResult:
        """Run the output rails in parallel."""
        pre_events = [
            (await create_event({"_type": "StartOutputRail", "flow_id": flow})).events[
                0
            ]
            for flow in flows
        ]
        post_events = [
            (
                await create_event({"_type": "OutputRailFinished", "flow_id": flow})
            ).events[0]
            for flow in flows
        ]

        return await self._run_flows_in_parallel(
            flows=flows, events=events, pre_events=pre_events, post_events=post_events
        )

    async def _run_output_rails_in_parallel_streaming(
        self, flows_with_params: Dict[str, dict], events: List[dict]
    ) -> ActionResult:
        """Run the output rails in parallel for streaming chunks.

        This is a streamlined version that avoids the full flow state management
        which can cause issues with hide_prev_turn logic during streaming.

        Args:
            flows_with_params: Dictionary mapping flow_id to {"action_name": str, "params": dict}
            events: The events list for context
        """
        tasks = []

        async def run_single_rail(flow_id: str, action_info: dict) -> tuple:
            """Run a single rail flow and return (flow_id, result)"""

            try:
                action_name = action_info["action_name"]
                params = action_info["params"]

                result_tuple = await self.action_dispatcher.execute_action(
                    action_name, params
                )
                result, status = result_tuple

                if status != "success":
                    log.error(f"Action {action_name} failed with status: {status}")
                    return flow_id, False  # Allow on failure

                action_func = self.action_dispatcher.get_action(action_name)

                # use the mapping to decide if the result indicates blocked content.
                # True means blocked, False means allowed
                result = is_output_blocked(result, action_func)

                return flow_id, result

            except Exception as e:
                log.error(f"Error executing rail {flow_id}: {e}")
                return flow_id, False  # Allow on error

        # create tasks for all flows
        for flow_id, action_info in flows_with_params.items():
            task = asyncio.create_task(run_single_rail(flow_id, action_info))
            tasks.append(task)

        stopped_events = []

        try:
            for future in asyncio.as_completed(tasks):
                try:
                    flow_id, is_blocked = await future

                    # check if this rail blocked the content
                    if is_blocked:
                        # create stop events
                        stopped_events = [
                            {
                                "type": "BotIntent",
                                "intent": "stop",
                                "flow_id": flow_id,
                            }
                        ]

                        # cancel remaining tasks
                        for pending_task in tasks:
                            if not pending_task.done():
                                pending_task.cancel()
                        break

                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    log.error(f"Error in parallel rail task: {e}")
                    continue

        except Exception as e:
            log.error(f"Error in parallel rail execution: {e}")
            return ActionResult(events=[])

        return ActionResult(events=stopped_events)

    async def _process_start_action(self, events: List[dict]) -> List[dict]:
        """
        Start the specified action, wait for it to finish, and post back the result.

        Args:
            events (List[dict]): The list of events.

        Returns:
            List[dict]: The list of next steps.
        """

        event = events[-1]

        action_name = event["action_name"]
        action_params = event["action_params"]
        action_result_key = event["action_result_key"]
        action_uid = event["action_uid"]

        context = {}
        action_meta = {}

        fn = self.action_dispatcher.get_action(action_name)

        # TODO: check action is available in action server
        if fn is None:
            status = "failed"
            result = self._internal_error_action_result(
                f"Action '{action_name}' not found."
            )

        else:
            context = compute_context(events)

            # We pass all the parameters that are passed explicitly to the action.
            kwargs = {**action_params}

            action_meta = getattr(fn, "action_meta", {})

            parameters = []
            action_type = "class"

            if inspect.isfunction(fn) or inspect.ismethod(fn):
                # We also add the "special" parameters.
                parameters = inspect.signature(fn).parameters
                action_type = "function"

            elif isinstance(fn, Chain):
                # If we're dealing with a chain, we list the annotations
                # TODO: make some additional type checking here
                parameters = fn.input_keys
                action_type = "chain"

            # For every parameter that start with "__context__", we pass the value
            for parameter_name in parameters:
                if parameter_name.startswith("__context__"):
                    var_name = parameter_name[11:]
                    kwargs[parameter_name] = context.get(var_name)

            # If there are parameters which are variables, we replace with actual values.
            for k, v in kwargs.items():
                if isinstance(v, str) and v.startswith("$"):
                    var_name = v[1:]
                    if var_name in context:
                        kwargs[k] = context[var_name]

            # If we have an action server, we use it for non-system/non-chain actions
            if (
                self.config.actions_server_url
                and not action_meta.get("is_system_action")
                and action_type != "chain"
            ):
                result, status = await self._get_action_resp(
                    action_meta, action_name, kwargs
                )
            else:
                # We don't send these to the actions server;
                # TODO: determine if we should
                if "events" in parameters:
                    kwargs["events"] = events

                if "context" in parameters:
                    kwargs["context"] = context

                if "config" in parameters:
                    kwargs["config"] = self.config

                if "llm_task_manager" in parameters:
                    kwargs["llm_task_manager"] = self.llm_task_manager

                # Add any additional registered parameters
                for k, v in self.registered_action_params.items():
                    if k in parameters:
                        kwargs[k] = v

                if (
                    "llm" in kwargs
                    and f"{action_name}_llm" in self.registered_action_params
                ):
                    kwargs["llm"] = self.registered_action_params[f"{action_name}_llm"]

                log.info("Executing action :: %s", action_name)
                result, status = await self.action_dispatcher.execute_action(
                    action_name, kwargs
                )

            # If the action execution failed, we return a hardcoded message
            if status == "failed":
                # TODO: make this message configurable.
                result = self._internal_error_action_result(
                    "I'm sorry, an internal error has occurred."
                )

        return_value = result
        return_events = []
        context_updates = {}

        if isinstance(result, ActionResult):
            return_value = result.return_value
            return_events = result.events
            context_updates.update(result.context_updates)

        # If we have an action result key, we also record the update.
        if action_result_key:
            context_updates[action_result_key] = return_value

        next_steps = []

        if context_updates:
            next_steps.append(new_event_dict("ContextUpdate", data=context_updates))

        next_steps.append(
            new_event_dict(
                "InternalSystemActionFinished",
                action_uid=action_uid,
                action_name=action_name,
                action_params=action_params,
                action_result_key=action_result_key,
                status=status,
                is_success=status != "failed",
                failure_reason=status,
                return_value=return_value,
                events=return_events,
                is_system_action=action_meta.get("is_system_action", False),
            )
        )

        # If the action returned additional events, we also add them to the next steps.
        if return_events:
            next_steps.extend(return_events)

        return next_steps

    async def _get_action_resp(
        self, action_meta: Dict[str, Any], action_name: str, kwargs: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], str]:
        """
        Interact with actions and get response from the action-server and system actions.

        Args:
            action_meta (Dict[str, Any]): Metadata for the action.
            action_name (str): The name of the action.
            kwargs (Dict[str, Any]): The action parameters.

        Returns:
            Tuple[Dict[str, Any], str]: The response and status.
        """
        result, status = {}, "failed"  # default response
        try:
            # Call the Actions Server if it is available.
            # But not for system actions, those should still run locally.
            if (
                action_meta.get("is_system_action", False)
                or self.config.actions_server_url is None
            ):
                result, status = await self.action_dispatcher.execute_action(
                    action_name, kwargs
                )
            else:
                url = urljoin(
                    self.config.actions_server_url, "/v1/actions/run"
                )  # action server execute action path
                data = {"action_name": action_name, "action_parameters": kwargs}
                async with aiohttp.ClientSession() as session:
                    try:
                        async with session.post(url, json=data) as resp:
                            if resp.status != 200:
                                raise ValueError(
                                    f"Got status code {resp.status} while getting response from {action_name}"
                                )

                            resp = await resp.json()
                            result, status = (
                                resp.get("result", result),
                                resp.get("status", status),
                            )
                    except Exception as e:
                        log.info(f"Exception {e} while making request to {action_name}")
                        return result, status

        except Exception as e:
            log.info(f"Failed to get response from {action_name} due to exception {e}")
        return result, status

    async def _process_start_flow(
        self, events: List[dict], processing_log: List[dict]
    ) -> List[dict]:
        """
        Start a flow.

        Args:
            events (List[dict]): The list of events.
            processing_log (List[dict]): The processing log so far. This will be mutated.

        Returns:
            List[dict]: The list of next steps.
        """

        event = events[-1]

        flow_id = event["flow_id"]

        # Up to this point, the body will be the sequence of instructions.
        # We need to alter it to be an actual flow definition, i.e., add `define flow xxx`
        # and intent the body.
        body = event["flow_body"]
        body = "define flow " + flow_id + ":\n" + indent(body, "  ")

        # We parse the flow
        parsed_data = parse_colang_file("dynamic.co", content=body)

        assert len(parsed_data["flows"]) == 1
        flow = parsed_data["flows"][0]

        # To make sure that the flow will start now, we add a start_flow element at
        # the beginning as well.
        flow["elements"].insert(0, {"_type": "start_flow", "flow_id": flow_id})

        # We add the flow to the list of flows.
        self._load_flow_config(flow)

        # And we compute the next steps. The new flow should match the current event,
        # and start.

        next_steps = await self._compute_next_steps(
            events, processing_log=processing_log
        )

        return next_steps
