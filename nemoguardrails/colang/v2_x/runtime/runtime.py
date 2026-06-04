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

"""Runtime for executing the guardrails based on Colang V2."""

import asyncio
import inspect
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urljoin

import aiohttp
from nemoguardrails.actions.actions import ActionResult
from nemoguardrails.atomic_hydrator import AtomicStateHydrator
from nemoguardrails.colang import parse_colang_file
from nemoguardrails.colang.runtime import Runtime
from nemoguardrails.colang.v2_x.lang.colang_ast import Decorator, Flow
from nemoguardrails.colang.v2_x.lang.utils import format_colang_parsing_error_message
from nemoguardrails.colang.v2_x.runtime.errors import (
    ColangRuntimeError,
    ColangSyntaxError,
)
from nemoguardrails.colang.v2_x.runtime.flows import Event, FlowStatus
from nemoguardrails.colang.v2_x.runtime.statemachine import (
    FlowConfig,
    InternalEvent,
    State,
    expand_elements,
    initialize_flow,
    initialize_state,
    run_to_completion,
)
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.utils import new_event_dict, new_readable_uuid

log = logging.getLogger(__name__)


class RuntimeV2_x(Runtime):
    """Runtime for executing the guardrails."""

    def __init__(self, config: RailsConfig, verbose: bool = False):
        super().__init__(config, verbose)

        # Register local system actions
        self.register_action(self._add_flows_action, "AddFlowsAction", False)
        self.register_action(self._remove_flows_action, "RemoveFlowsAction", False)

        # Inicializa o hydrator passando a instância do runtime para lazy routing
        self.hydrator = AtomicStateHydrator(backend_client=self)

        # Maps main_flow.uid to a dictionary of actions that are run locally, asynchronously.
        # Dict[main_flow_uid, Dict[action_uid, action_data]]
        self.async_actions: Dict[str, List] = {}

        # A way to disable async function execution. Useful for testing.
        self.disable_async_execution = False

    async def _add_flows_action(self, state: "State", **args: dict) -> List[str]:
        log.info("Start AddFlowsAction! %s", args)
        flow_content = args["config"]
        if not isinstance(flow_content, str):
            raise ColangRuntimeError("Parameter 'config' in AddFlowsAction is not of type 'str'!")
        try:
            parsed_flow = parse_colang_file(
                filename="",
                content=flow_content,
                version="2.x",
                include_source_mapping=True,
            )
        except Exception as e:
            log.warning(
                "Failed parsing a generated flow\n%s\n%s",
                flow_content,
                format_colang_parsing_error_message(e, flow_content),
            )

            flow_name = flow_content.split("\n")[0].split(" ", maxsplit=1)[1]
            fixed_body = f"flow {flow_name}\n" + f'    bot say "Internal error on flow `{flow_name}`."'
            log.warning("Using the following flow instead:\n%s", fixed_body)

            parsed_flow = parse_colang_file(
                filename="",
                content=fixed_body,
                version="2.x",
                include_source_mapping=True,
            )

        added_flows: List[str] = []
        for flow in parsed_flow["flows"]:
            if flow.name in state.flow_configs:
                log.warning("Flow '%s' already exists! Not loaded!", flow.name)
                break

            flow_config = FlowConfig(
                id=flow.name,
                elements=expand_elements(flow.elements, state.flow_configs),
                decorators=convert_decorator_list_to_dictionary(flow.decorators),
                parameters=flow.parameters,
                return_members=flow.return_members,
                source_code=flow.source_code,
            )

            initialize_flow(state, flow_config)
            state.flow_configs.update({flow.name: flow_config})
            added_flows.append(flow.name)

        return added_flows

    async def _remove_flows_action(self, state: "State", **args: dict) -> None:
        log.info("Start RemoveFlowsAction! %s", args)
        flow_ids = args["flow_ids"]
        for flow_id in flow_ids:
            if flow_id in state.flow_id_states:
                for flow_state in state.flow_id_states[flow_id]:
                    del state.flow_states[flow_state.uid]
                del state.flow_id_states[flow_id]
            if flow_id in state.flow_configs:
                del state.flow_configs[flow_id]

    def _init_flow_configs(self) -> None:
        """Initializes the flow configs based on the config."""
        self.flow_configs = create_flow_configs_from_flow_list(self.config.flows)

    async def generate_events(self, events: List[dict]) -> List[dict]:
        raise NotImplementedError("Stateless API not supported for Colang 2.x, yet.")

    @staticmethod
    def _internal_error_action_result(message: str) -> ActionResult:
        """Helper to construct an action result for an internal error."""
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
                {"type": "hide_prev_turn"},
            ]
        )

    async def _process_start_action(
        self,
        action_name: str,
        action_params: dict,
        context: dict,
        events: List[dict],
        state: "State",
    ) -> Tuple[Any, List[dict], dict]:
        """Starts the specified action, waits for it to finish and posts back the result."""
        fn = self.action_dispatcher.get_action(action_name)
        if fn is None:
            result = self._internal_error_action_result(f"Action '{action_name}' not found.")
        else:
            kwargs = {**action_params}
            action_meta = getattr(fn, "action_meta", {})
            parameters = []

            if inspect.isfunction(fn) or inspect.ismethod(fn):
                parameters = inspect.signature(fn).parameters

            for parameter_name in parameters:
                if parameter_name.startswith("__context__"):
                    var_name = parameter_name[11:]
                    kwargs[parameter_name] = context.get(var_name)

            for k, v in kwargs.items():
                if isinstance(v, str) and v.startswith("$"):
                    var_name = v[1:]
                    if var_name in context:
                        kwargs[k] = context[var_name]

            if self.config.actions_server_url and not action_meta.get("is_system_action"):
                result, status = await self._get_action_resp(action_meta, action_name, kwargs)
            else:
                if "events" in parameters:
                    kwargs["events"] = events
                if "context" in parameters:
                    kwargs["context"] = context
                if "config" in parameters:
                    kwargs["config"] = self.config
                if "llm_task_manager" in parameters:
                    kwargs["llm_task_manager"] = self.llm_task_manager
                if "state" in parameters:
                    kwargs["state"] = state

                for k, v in self.registered_action_params.items():
                    if k in parameters:
                        kwargs[k] = v

                if "llm" in kwargs and f"{action_name}_llm" in self.registered_action_params:
                    kwargs["llm"] = self.registered_action_params[f"{action_name}_llm"]

                log.info("Running action :: %s", action_name)
                result, status = await self.action_dispatcher.execute_action(action_name, kwargs)

            if status == "failed":
                result = self._internal_error_action_result("I'm sorry, an internal error has occurred.")

        return_value = result
        return_events: List[dict] = []
        context_updates: dict = {}

        if isinstance(result, ActionResult):
            return_value = result.return_value
            if result.events is not None:
                return_events = result.events
            if result.context_updates is not None:
                context_updates.update(result.context_updates)

        return return_value, return_events, context_updates

    async def _get_action_resp(
        self, action_meta: Dict[str, Any], action_name: str, kwargs: Dict[str, Any]
    ) -> Tuple[Union[str, Dict[str, Any]], str]:
        result: Union[str, Dict[str, Any]] = {}
        status: str = "failed"
        try:
            if action_meta.get("is_system_action", False) or self.config.actions_server_url is None:
                result, status = await self.action_dispatcher.execute_action(action_name, kwargs)
            else:
                url = urljoin(self.config.actions_server_url, "/v1/actions/run")
                data = {"action_name": action_name, "action_parameters": kwargs}
                async with aiohttp.ClientSession() as session:
                    try:
                        async with session.post(url, json=data) as resp:
                            if resp.status != 200:
                                raise ValueError(
                                    f"Got status code {resp.status} while getting response from {action_name}"
                                )
                            resp = await resp.json()
                            result, status = resp.get("result", result), resp.get("status", status)
                    except Exception as e:
                        log.info("Exception %s while making request to %s", e, action_name)
                        return result, status
        except Exception as e:
            error_message = f"Failed to get response from {action_name} due to exception {e}"
            log.info(error_message)
            raise ColangRuntimeError(error_message) from e
        return result, status

    @staticmethod
    def _get_action_finished_event(result: dict, **kwargs: Any) -> Dict[str, Any]:
        return new_event_dict(
            f"{result['action_name']}Finished",
            action_uid=result["start_action_event"]["action_uid"],
            action_name=result["action_name"],
            status="success",
            is_success=True,
            return_value=result["return_value"],
            events=result["new_events"],
            **kwargs,
        )

    async def _get_async_actions_finished_events(self, main_flow_uid: str) -> Tuple[List[dict], int]:
        pending_actions = self.async_actions.get(main_flow_uid, [])
        if len(pending_actions) == 0:
            return [], 0

        done, pending = await asyncio.wait(
            pending_actions,
            return_when=asyncio.FIRST_COMPLETED,
            timeout=0,
        )
        if len(done) > 0:
            log.info("%s actions finished.", len(done))

        action_finished_events = []
        for finished_task in done:
            try:
                result = finished_task.result()
            except Exception:
                log.warning("Local action finished with an exception!", exc_info=True)

            self.async_actions[main_flow_uid].remove(finished_task)
            action_finished_event = self._get_action_finished_event(result)
            action_finished_events.append(action_finished_event)

        return action_finished_events, len(pending)

    async def fetch_state(self, conversation_id: str) -> Any:
        """Stub pass-through structure redirection targeting abstract interface compliance."""
        return None

    async def save_state(self, conversation_id: str, updated_state: Any) -> None:
        """Stub pass-through structure redirection targeting abstract interface compliance."""
        pass

    async def process_events(
        self,
        events: List[dict],
        state: Optional[State] = None,
        blocking: bool = False,
        instant_actions: Optional[List[str]] = None,
    ) -> Tuple[List[dict], State]:
        """Process a sequence of events in a given state.

        Alinhado estritamente no plural atendendo ao contrato abstrato da classe Runtime base.
        """

        async def _run_pipeline(current_state: Any) -> Tuple[List[dict], State]:
            return await self._execute_event_cycle_internals(events, current_state, blocking, instant_actions)

        conversation_id = getattr(state, "uid", "global_fallback_session")
        return await self.hydrator.execute_atomic_pipeline(conversation_id, _run_pipeline)

    async def _execute_event_cycle_internals(
        self,
        events: List[dict],
        state: Optional[State] = None,
        blocking: bool = False,
        instant_actions: Optional[List[str]] = None,
    ) -> Tuple[List[dict], State]:
        """Internal encapsulated processing runner protected by the session sharded lock barrier."""
        output_events = []
        input_events: List[Union[dict, InternalEvent]] = events.copy()
        local_running_actions: List[asyncio.Task[dict]] = []

        if state is None or state == {}:
            state = State(flow_states={}, flow_configs=self.flow_configs, rails_config=self.config)
            initialize_state(state)
        elif isinstance(state, dict):
            raise NotImplementedError()

        assert isinstance(state, State)
        assert state.main_flow_state is not None
        main_flow_uid = state.main_flow_state.uid
        if state.main_flow_state.status == FlowStatus.WAITING:
            log.info("Start of story!")
            input_event = InternalEvent(name="StartFlow", arguments={"flow_id": "main"})
            input_events.insert(0, input_event)
            main_flow_state = state.flow_id_states["main"][-1]

            idx = 0
            for flow_config in reversed(state.flow_configs.values()):
                if "active" in flow_config.decorators:
                    input_event = InternalEvent(
                        name="StartFlow",
                        arguments={
                            "flow_id": flow_config.id,
                            "source_flow_instance_uid": main_flow_state.uid,
                            "flow_instance_uid": new_readable_uuid(flow_config.id),
                            "flow_hierarchy_position": f"0.0.{idx}",
                            "source_head_uid": list(main_flow_state.heads.values())[0].uid,
                            "activated": True,
                        },
                    )
                    input_events.insert(0, input_event)
                    idx += 1

        (
            local_action_finished_events,
            pending_local_async_action_counter,
        ) = await self._get_async_actions_finished_events(main_flow_uid)
        input_events.extend(local_action_finished_events)
        local_action_finished_events = []
        return_local_async_action_count = False

        events_counter = 0
        while input_events or local_running_actions:
            new_outgoing_events = []
            for event in input_events:
                events_counter += 1
                if events_counter > self.max_events:
                    log.critical(f"Maximum number of events reached ({events_counter})!")
                    return output_events, state

                log.info("Processing event :: %s", event)
                for watcher in self.watchers:
                    watcher(event)

                event_name = event["type"] if isinstance(event, dict) else event.name
                if event_name == "CheckLocalAsync":
                    return_local_async_action_count = True
                    continue

                state.last_events.append(event)
                new_event: Optional[Union[dict, Event]] = event
                while new_event is not None:
                    try:
                        run_to_completion(state, new_event)
                        new_event = None
                    except Exception as e:
                        log.warning("Colang runtime error!", exc_info=True)
                        new_event = Event(
                            name="ColangError",
                            arguments={"type": str(type(e).__name__), "error": str(e)},
                        )
                    await asyncio.sleep(0.001)

                for out_event in state.outgoing_events:
                    state.last_events.append(out_event)
                    start_action_match = re.match(r"Start(.*Action)", out_event["type"])
                    if start_action_match:
                        action_name = start_action_match[1]
                        if instant_actions and action_name in instant_actions:
                            finished_event_data: dict = {
                                "action_name": action_name,
                                "start_action_event": out_event,
                                "return_value": None,
                                "new_events": [],
                            }
                            extra = {}
                            if action_name == "UtteranceBotAction":
                                extra["final_script"] = out_event["script"]

                            action_finished_event = self._get_action_finished_event(finished_event_data, **extra)
                            output_events.append(action_finished_event)
                            input_events.append(action_finished_event)

                        elif self.action_dispatcher.has_registered(action_name):
                            action_fn = self.action_dispatcher.get_action(action_name)
                            execute_async = getattr(action_fn, "action_meta", {}).get("execute_async", False)

                            local_action = asyncio.create_task(
                                self._run_action(
                                    action_name,
                                    start_action_event=out_event,
                                    events_history=state.last_events,
                                    state=state,
                                )
                            )

                            if not execute_async or self.disable_async_execution or blocking:
                                local_running_actions.append(local_action)
                            else:
                                main_flow_uid = state.main_flow_state.uid
                                if main_flow_uid not in self.async_actions:
                                    self.async_actions[main_flow_uid] = []
                                self.async_actions[main_flow_uid].append(local_action)
                        else:
                            output_events.append(out_event)
                    else:
                        output_events.append(out_event)

                (
                    new_local_action_finished_events,
                    pending_local_async_action_counter,
                ) = await self._get_async_actions_finished_events(main_flow_uid)
                local_action_finished_events.extend(new_local_action_finished_events)
                new_outgoing_events.extend(state.outgoing_events)

            input_events.clear()
            if new_outgoing_events:
                input_events.extend(new_outgoing_events)
                continue

            input_events.extend(local_action_finished_events)
            local_action_finished_events = []

            if local_running_actions:
                log.info("Waiting for %d local actions to finish.", len(local_running_actions))
                done, _pending = await asyncio.wait(local_running_actions, return_when=asyncio.FIRST_COMPLETED)
                log.info("%s actions finished.", len(done))

                for finished_task in done:
                    local_running_actions.remove(finished_task)
                    result = finished_task.result()
                    action_finished_event = self._get_action_finished_event(result)
                    input_events.append(action_finished_event)

        if return_local_async_action_count:
            log.debug("Checking if there are any local async actions that have finished.")
            output_events.append(new_event_dict("LocalAsyncCounter", counter=pending_local_async_action_counter))

        state.last_events = state.last_events[-500:]
        return output_events, state

    async def _run_action(
        self,
        action_name: str,
        start_action_event: dict,
        events_history: List[Union[dict, Event]],
        state: "State",
    ) -> dict:
        ignore_keys = new_event_dict(start_action_event["type"]).keys()
        action_params = {k: v for k, v in start_action_event.items() if k not in ignore_keys}

        return_value, new_events, context_updates = await self._process_start_action(
            action_name,
            action_params=action_params,
            context=state.context,
            events=events_history,
            state=state,
        )

        state.context.update(context_updates)
        return {
            "action_name": action_name,
            "return_value": return_value,
            "new_events": new_events,
            "context_updates": context_updates,
            "start_action_event": start_action_event,
        }


def convert_decorator_list_to_dictionary(decorators: List[Decorator]) -> Dict[str, Dict[str, Any]]:
    """Convert list of decorators to a dictionary merging the parameters."""
    decorator_dict: Dict[str, Dict[str, Any]] = {}
    for decorator in decorators:
        item = decorator_dict.get(decorator.name, None)
        if item:
            item.update(decorator.parameters)
        else:
            decorator_dict[decorator.name] = decorator.parameters
    return decorator_dict


def create_flow_configs_from_flow_list(flows: List[Flow]) -> Dict[str, FlowConfig]:
    """Create a flow config dictionary and resolves flow overriding."""
    flow_configs: Dict[str, FlowConfig] = {}
    override_flows: Dict[str, FlowConfig] = {}

    for flow in flows:
        assert isinstance(flow, Flow)
        if flow.name.split(" ")[0] in ["send", "match", "start", "stop", "await", "activate"]:
            raise ColangSyntaxError(f"Flow '{flow.name}' starts with a keyword!")

        config = FlowConfig(
            id=flow.name,
            elements=flow.elements,
            decorators=convert_decorator_list_to_dictionary(flow.decorators),
            parameters=flow.parameters,
            return_members=flow.return_members,
            source_code=flow.source_code,
            source_file=flow.file_info["name"],
        )

        if config.is_override:
            if flow.name in override_flows:
                raise ColangSyntaxError(
                    f"Multiple override flows with name '{flow.name}' detected! There can only be one!"
                )
            override_flows[flow.name] = config
        elif flow.name in flow_configs:
            raise ColangSyntaxError(
                f"Multiple non-overriding flows with name '{flow.name}' detected! There can only be one!"
            )
        else:
            flow_configs[flow.name] = config

    for override_flow in override_flows.values():
        if override_flow.id not in flow_configs:
            raise ColangSyntaxError(
                f"Override flow with name '{override_flow.id}' does not override any flow with that name!"
            )
        flow_configs[override_flow.id] = override_flow

    return flow_configs
