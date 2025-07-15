------------------------------------------------------------------------------------------------------------------------------------------------------------------
Guardrail Flows (`guardrails.co <../../../nemoguardrails/colang/v2_x/library/guardrails.co>`_)
------------------------------------------------------------------------------------------------------------------------------------------------------------------

Flows to guardrail user inputs and LLM responses.

.. code-block:: colang

    # Check user utterances before they get further processed
    flow run input rails $input_text

    # Check llm responses before they get further processed
    flow run output rails $output_text
