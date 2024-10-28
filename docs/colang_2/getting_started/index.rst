.. _getting_started:

Getting Started
===============

This section is a getting started guide for Colang 2.0. It starts with a basic "Hello World" example and then goes into dialog rails, input rails, multimodal rails and other Colang 2.0 concepts like interaction loops and LLM flows. This guide does not assume any experience with Colang 1.0, and all the concepts are explained from scratch.

Prerequisites
-------------

This getting started guide will focus only on the Colang files. For complete details on how to install NeMo Guardrails and create a sample configuration please refer to the `Installation Guide <#>`_ and the `NeMo Guardrails Getting Started Guide <#>`_.

The ``config.yml`` file for all the examples should have the following content:

.. code-block:: yaml
  :caption: config.yml

  colang_version: "2.x"

  models:
    - type: main
      engine: openai
      model: gpt-4-turbo

The above config sets the Colang version to "2.x" (this is needed since "1.0" is currently the default) and the LLM engine to OpenAI's ``gpt-4-turbo``. Make sure to set the required API access key as an environment variable (e.g. OPENAI_API_KEY for OpenAI API). See section :ref:`Supported Models<make-use-of-llms-supported-models>` for all supported models.

.. note::
  Check the section :ref:`development-and-debugging` for how you can install Colang syntax highlighting to make editing Colang scripts easier.

Terminology
-----------

At a high level, Colang adopts as much as possible from the Python terminology. This guide will talk about Colang scripts and modules (i.e., ``.co`` files), packages (i.e., folders), standard library, importing mechanism, etc.

Guides
------
.. toctree::
  :maxdepth: 2

  hello-world.rst
  dialog-rails.rst
  multimodal-rails.rst
  input-rails.rst
  interaction-loop.rst
  llm-flows.rst
  recommended-next-steps.rst
