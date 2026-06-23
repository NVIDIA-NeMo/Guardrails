# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Mock agent tools the guardrail sits in front of.

Stand-ins for real agent capabilities. They perform no authorization of their
own — that is exactly the point: the guard in `policy.py` decides whether a call
is permitted, and only then does dispatch reach these functions.
"""

from __future__ import annotations

_BALANCES = {
    "acct-1001": 4200.00,
    "acct-1002": 75.50,
    "acct-9999": 1_000_000.00,
}


def read_account(account_id: str) -> str:
    balance = _BALANCES.get(account_id)
    if balance is None:
        return f"account {account_id} not found"
    return f"account {account_id} balance: ${balance:,.2f}"


def transfer_funds(from_account: str, to_account: str, amount: float) -> str:
    if _BALANCES.get(from_account, 0) < amount:
        return f"insufficient funds in {from_account}"
    return f"transferred ${amount:,.2f} from {from_account} to {to_account}"


# Dispatch table mapping tool names (as referenced by policies) to implementations.
TOOLS = {
    "read_account": read_account,
    "transfer_funds": transfer_funds,
}
