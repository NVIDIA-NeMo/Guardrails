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

import unittest
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from nemoguardrails.actions.llm.utils import llm_call
from nemoguardrails.llm.params import LLMParams, llm_params, register_param_manager


class FakeLLM(BaseModel):
    """Fake LLM wrapper for testing purposes."""

    model_kwargs: Dict[str, Any] = {}
    param3: str = ""


class FakeLLM2(BaseModel):
    param3: str = ""


class TestLLMParams(unittest.TestCase):
    def setUp(self):
        self.llm = FakeLLM(
            param3="value3", model_kwargs={"param1": "value1", "param2": "value2"}
        )
        self.llm_params = LLMParams(
            self.llm, param1="new_value1", param2="new_value2", param3="new_value3"
        )

    def test_init(self):
        self.assertEqual(self.llm_params.llm, self.llm)
        self.assertEqual(
            self.llm_params.altered_params,
            {"param1": "new_value1", "param2": "new_value2", "param3": "new_value3"},
        )
        self.assertEqual(self.llm_params.original_params, {})

    def test_enter(self):
        llm = self.llm
        with llm_params(
            llm, param1="new_value1", param2="new_value2", param3="new_value3"
        ):
            self.assertEqual(self.llm.param3, "new_value3")
            self.assertEqual(self.llm.model_kwargs["param1"], "new_value1")

    def test_exit(self):
        with self.llm_params:
            pass
        self.assertEqual(self.llm.model_kwargs["param1"], "value1")
        self.assertEqual(self.llm.param3, "value3")

    def test_enter_with_nonexistent_param(self):
        """Test that entering the context manager with a nonexistent parameter logs a warning."""

        with self.assertLogs(level="WARNING") as cm:
            with llm_params(self.llm, nonexistent_param="value"):
                pass
        self.assertIn(
            "Parameter nonexistent_param does not exist for FakeLLM", cm.output[0]
        )

    def test_exit_with_nonexistent_param(self):
        """Test that exiting the context manager with a nonexistent parameter does not raise an error."""

        llm_params = LLMParams(self.llm, nonexistent_param="value")
        llm_params.original_params = {"nonexistent_param": "original_value"}
        try:
            with llm_params:
                pass
        except Exception as e:
            self.fail(f"Exiting the context manager raised an exception: {e}")


class TestLLMParamsWithEmptyModelKwargs(unittest.TestCase):
    def setUp(self):
        self.llm = FakeLLM(param3="value3", model_kwargs={})
        self.llm_params = LLMParams(
            self.llm, param1="new_value1", param2="new_value2", param3="new_value3"
        )

    def test_init(self):
        self.assertEqual(self.llm_params.llm, self.llm)
        self.assertEqual(
            self.llm_params.altered_params,
            {"param1": "new_value1", "param2": "new_value2", "param3": "new_value3"},
        )
        self.assertEqual(self.llm_params.original_params, {})

    def test_enter(self):
        llm = self.llm
        with llm_params(
            llm, param1="new_value1", param2="new_value2", param3="new_value3"
        ):
            self.assertEqual(self.llm.param3, "new_value3")
            self.assertEqual(self.llm.model_kwargs["param1"], "new_value1")
            self.assertEqual(self.llm.model_kwargs["param2"], "new_value2")

    def test_exit(self):
        with self.llm_params:
            pass
        self.assertEqual(self.llm.model_kwargs["param1"], None)
        self.assertEqual(self.llm.param3, "value3")

    def test_enter_with_empty_model_kwargs(self):
        """Test that entering the context manager with empty model_kwargs logs a warning."""
        warning_message = f"Parameter param1 does not exist for {self.llm.__class__.__name__}. Passing to model_kwargs"

        with self.assertLogs(level="WARNING") as cm:
            with llm_params(self.llm, param1="new_value1"):
                pass
        self.assertIn(
            warning_message,
            cm.output[0],
        )

    def test_exit_with_empty_model_kwargs(self):
        """Test that exiting the context manager with empty model_kwargs does not raise an error."""

        llm_params = LLMParams(self.llm, param1="new_value1")
        llm_params.original_params = {"param1": "original_value"}
        try:
            with llm_params:
                pass
        except Exception as e:
            self.fail(f"Exiting the context manager raised an exception: {e}")


class TestLLMParamsWithoutModelKwargs(unittest.TestCase):
    def setUp(self):
        self.llm = FakeLLM2(param3="value3")
        self.llm_params = LLMParams(
            self.llm, param1="new_value1", param2="new_value2", param3="new_value3"
        )

    def test_init(self):
        self.assertEqual(self.llm_params.llm, self.llm)
        self.assertEqual(
            self.llm_params.altered_params,
            {"param1": "new_value1", "param2": "new_value2", "param3": "new_value3"},
        )
        self.assertEqual(self.llm_params.original_params, {})

    def test_enter(self):
        llm = self.llm
        with llm_params(
            llm, param1="new_value1", param2="new_value2", param3="new_value3"
        ):
            self.assertEqual(self.llm.param3, "new_value3")

    def test_exit(self):
        with self.llm_params:
            pass
        self.assertEqual(self.llm.param3, "value3")

    def test_enter_with_empty_model_kwargs(self):
        """Test that entering the context manager with empty model_kwargs logs a warning."""
        warning_message = (
            f"Parameter param1 does not exist for {self.llm.__class__.__name__}"
        )
        with self.assertLogs(level="WARNING") as cm:
            with llm_params(self.llm, param1="new_value1"):
                pass
        self.assertIn(
            warning_message,
            cm.output[0],
        )

    def test_exit_with_empty_model_kwargs(self):
        """Test that exiting the context manager with empty model_kwargs does not raise an error."""

        llm_params = LLMParams(self.llm, param1="new_value1")
        llm_params.original_params = {"param1": "original_value"}
        try:
            with llm_params:
                pass
        except Exception as e:
            self.fail(f"Exiting the context manager raised an exception: {e}")


class TestRegisterParamManager(unittest.TestCase):
    def test_register_param_manager(self):
        """Test that a custom parameter manager can be registered and retrieved."""

        class CustomLLMParams(LLMParams):
            pass

        register_param_manager(FakeLLM, CustomLLMParams)
        self.assertEqual(llm_params(FakeLLM()).__class__, CustomLLMParams)


class TestLLMParamsFunction(unittest.TestCase):
    def test_llm_params_with_registered_manager(self):
        """Test that llm_params returns the registered manager for a given LLM type."""

        class CustomLLMParams(LLMParams):
            pass

        register_param_manager(FakeLLM, CustomLLMParams)
        self.assertIsInstance(llm_params(FakeLLM()), CustomLLMParams)

    def test_llm_params_with_unregistered_manager(self):
        """Test that llm_params returns the default manager for an unregistered LLM type."""

        class UnregisteredLLM(BaseModel):
            pass

        self.assertIsInstance(llm_params(UnregisteredLLM()), LLMParams)


class TestLLMParamsMigration(unittest.TestCase):
    """Test migration from context manager to direct parameter passing."""

    def test_context_manager_equivalent_to_direct_params(self):
        """Test that context manager behavior matches direct parameter passing."""
        llm = FakeLLM(param3="original", model_kwargs={"temperature": 0.5})

        with llm_params(llm, temperature=0.8, param3="modified"):
            context_temp = llm.model_kwargs.get("temperature")
            context_param3 = llm.param3

        assert context_temp == 0.8
        assert context_param3 == "modified"
        assert llm.model_kwargs.get("temperature") == 0.5
        assert llm.param3 == "original"

    @pytest.mark.asyncio
    async def test_llm_call_params_vs_context_manager(self):
        """Test that llm_call with params produces similar results to context manager approach."""
        mock_llm = AsyncMock()
        mock_bound_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Response content"

        mock_llm.bind.return_value = mock_bound_llm
        mock_bound_llm.ainvoke.return_value = mock_response

        params = {"temperature": 0.7, "max_tokens": 100}

        result = await llm_call(mock_llm, "Test prompt", llm_params=params)

        assert result == "Response content"
        mock_llm.bind.assert_called_once_with(**params)
        mock_bound_llm.ainvoke.assert_called_once()

    def test_parameter_isolation_after_migration(self):
        """Test that parameter changes don't persist after llm_call completes."""
        llm = FakeLLM(param3="original", model_kwargs={"temperature": 0.5})
        original_temp = llm.model_kwargs.get("temperature")
        original_param3 = llm.param3

        assert original_temp == 0.5
        assert original_param3 == "original"
