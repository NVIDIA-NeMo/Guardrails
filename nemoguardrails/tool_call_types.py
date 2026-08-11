from typing import List, Literal

from pydantic import BaseModel


class FunctionCallDelta(BaseModel):
    name: str
    arguments: str


class ToolCallDelta(BaseModel):
    index: int
    id: str
    function: FunctionCallDelta
    type: Literal["function"] = "function"


class ChunkToolCallDelta(BaseModel):
    tool_calls: List[ToolCallDelta]
    type: Literal["tool_call_delta"] = "tool_call_delta"
