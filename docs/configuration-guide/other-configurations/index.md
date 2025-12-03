# Other Configurations

This section provides additional configuration topics that are not covered in the previous sections of the configuration guide.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Knowledge Base
:link: knowledge-base
:link-type: doc

The NeMo Guardrails toolkit supports using a set of documents as context for generating bot responses through Retrieval-Augmented Generation (RAG). This guide explains how to configure and use the...
:::

:::{grid-item-card} Exceptions and Error Handling
:link: exceptions
:link-type: doc

NeMo Guardrails supports raising exceptions from within flows. An exception is an event whose name ends with `Exception`, e.g., `InputRailException`. When an exception is raised, the final output...
:::

::::

```{toctree}
:hidden:
:maxdepth: 2

knowledge-base
exceptions
```
