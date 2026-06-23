# Speculative multi-turn tool-call loops

Source: https://example.org/research/2026-speculative-loops

Recent red-team work documents agents being driven into long speculative chains
of tool calls, where each step is individually benign but the sequence as a whole
drains resources or escalates state. Defending this likely requires reasoning
about call sequences over a session rather than authorizing each call in
isolation — a control the current per-call model does not express. The affected
surface here includes the `transfer_funds` tool.
