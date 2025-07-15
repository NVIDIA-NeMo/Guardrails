# Nested AsyncIO Loop

NeMo Guardrails is an async-first toolkit, i.e., the core functionality is implemented using async functions. To provide a blocking API, the toolkit must invoke async functions inside synchronous code using `asyncio.run`. However, the current Python implementation for `asyncio` does not allow "nested event loops". This issue is being discussed by the Python core team and, most likely, support will be added (see [GitHub Issue 66435](https://github.com/python/cpython/issues/66435) and [Pull Request 93338](https://github.com/python/cpython/pull/93338)).

Meanwhile, NeMo Guardrails makes use of [nest_asyncio](https://github.com/erdewit/nest_asyncio). The patching is applied when the `nemoguardrails` package is loaded the first time.

If the blocking API is not needed, or the `nest_asyncio` patching causes unexpected problems, you can disable it by setting the `DISABLE_NEST_ASYNCIO=True` environment variable.
