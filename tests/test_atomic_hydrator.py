# Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock

import pytest

from nemoguardrails.atomic_hydrator import AtomicStateHydrator


@pytest.mark.asyncio
class TestAtomicStateHydrator:
    @pytest.fixture
    def mock_backend(self):
        backend = AsyncMock()
        backend.fetch_state.return_value = {"status": "intact"}
        return backend

    @pytest.fixture
    def hydrator(self, mock_backend):
        return AtomicStateHydrator(backend_client=mock_backend)

    async def test_atomic_pipeline_success_and_cleanup(self, hydrator, mock_backend):
        conv_id = "sessao_segura_01"

        async def mock_evaluation(state, *args, **kwargs):
            return "resultado_aprovado", {"status": "updated"}

        events, updated_state = await hydrator.execute_atomic_pipeline(conv_id, mock_evaluation)

        assert events == "resultado_aprovado"
        assert updated_state == {"status": "updated"}
        assert hydrator._ref_counts.get(conv_id) is None
        assert hydrator._locks.get(conv_id) is None

    async def test_atomic_pipeline_failure_preserves_integrity(self, hydrator, mock_backend):
        conv_id = "sessao_falha_02"

        async def failing_evaluation(state, *args, **kwargs):
            raise RuntimeError("Colapso simulado na IA")

        with pytest.raises(RuntimeError, match="Colapso simulado na IA"):
            await hydrator.execute_atomic_pipeline(conv_id, failing_evaluation)

        assert hydrator._ref_counts.get(conv_id) is None
        assert hydrator._locks.get(conv_id) is None
