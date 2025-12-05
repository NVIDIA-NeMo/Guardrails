.. _make-use-of-llms:

========================================
Make use of Large Language Models (LLM)
========================================

----------------------------------------
Introduction
----------------------------------------

While at the core Colang does not require a Large Language Model (LLM) as backend, many of the more advanced mechanisms in the :ref:`Colang Standard Library<the-standard-library>` (CSL) depend on it.

To enable the LLM backend you first have to configure the LLM access in the `config.yml` by adding a `models` section like this:

.. code-block:: yaml

    models:
    - type: main
      engine: openai
      model: gpt-4-turbo

Make sure to also define the required API access key, e.g. for OpenAI you will have to set the ``OPENAI_API_KEY`` environment variable.

Every LLM prompt contains a default context that can be modified if needed to adapt to the use case. See this `example configuration <../../../tests/test_configs/multi_modal_demo_v2_x/demo.yml>`_ to get started. This will heavily influence all the LLM invocations.

.. _make-use-of-llms-supported-models:

----------------------------------------
Supported Models
----------------------------------------

Colang currently supports the following models out of the box:

.. code-block:: yaml

    engine: openai
    model: gpt-3.5-turbo-instruct
    model: gpt-3.5-turbo
    model: gpt-4-turbo
    model: gpt-4o
    model: gpt-4o-mini

`NVIDIA AI Foundary <https://www.nvidia.com/en-us/ai/foundry/>`_ hosted NIMs:

.. code-block:: yaml

    engine: nim
    model: meta/llama3-8b-instruct
    model: meta/llama3-70b-instruct
    model: meta/llama-3.1-8b-instruct
    model: meta/llama-3.1-70b-instruct

To support other models you would need to create a set of new `template prompts <../../../nemoguardrails/llm/prompts>`_ that consider the specific capabilities and the API of the model and add them to your bot configuration.

----------------------------------------
Natural Language Description (NLD)
----------------------------------------

One of the main LLM generation mechanism in Colang are the so-called Natural Language Description (NLD) in combination with the "generation" operator ``...``.

.. code-block:: colang

    # Assign result of NLD to a variable
    $world_population = ..."What is the number of people in the world? Give me a number."

    # Extract a value from the current interaction context
    $user_name = ..."What is the name of the user? Return 'friend' if not available."

    # Extract structured information from the current interaction context
    $order_information = ..."Provide the products ordered by the user in a list structure, e.g. ['product a', 'product b']"

    # Use an existing variable in NLD
    $response_to_user = ..."Provide a brief summary of the current order. Order Information: '{$order_information}'"

Every NLD will be interpreted and replaced during runtime by the configured LLM backend and can be used in Colang to generate context dependent values. With NLDs you are able to extract values and summarize content from the conversation with the user or based on results from other sources (like a database or an external service).

.. note::
    NLDs together with the variable name are interpreted by the LLM directly. Depending on the LLM you use you need to make sure to be very specific in what value you would like to generate. It is good practice to always clearly specify how you want the response to be formatted and what type it should have (e.g., ``$user_name = ..."Return the user name as single string between quotes''. If no user name is available return 'friend'"``.

Alternatively, you can also describe the purpose and function of a flow using a docstring like NLD at the beginning of a flow. Using a standalone generation operator ``...`` in the flow will use the flows NLD to infer the right flow expansion automatically:

.. code-block:: colang

    flow main
        """You are an assistant that should talk to the user about cars.
        Politely decline to talk about anything else.

        Last user question is: "{{ question }}"
        Generate the output in the following format:

        bot say "<<the response>>"
        """
        $question = await user said something
        ...

See the example in :ref:`colang_2_getting_started_llm_flows` for more details on how this works.

.. In the future NLDs can also be used in the following ways:

.. .. code-block:: colang

..     # Use NLDs as flow parameters
..     bot say i"Welcome the user with a short sentence."

..     # Lazy evaluation of NLDs using the leading `i` characters
..     bot say i"Welcome the user with a short sentence."

..     # Use lazy NLDs for event match parameters
..     user said i"A question about politics"

..     # Complete flow patterns
..     flow handle payment process
..         i"handle user payment process"

.. In order to work with the configured LLM we need to activate the standard library flow ``polling llm request response`` from the llm module (`Github link <../../../nemoguardrails/colang/v2_x/library/llm.co>`__). This will activate a system timer to actively poll the LLM request response such that the interaction can progress without another system event:

.. .. code-block:: colang
..     :caption: llm/nld_example/main.co

..     import llm

..     flow main
..         activate polling llm request response
..         $text = ..."Welcome the user with a short sentence."
..         bot say $text
..         user said something

.. .. note::
..     Currently, NLDs cannot yet be used directly as flow or event parameters but need to be assigned to a variable first.

Note that there is no explicit control over the NLD response format and sometimes it will fail to generate the expected result. Usually you can improve it by providing more explicit instructions in the NLD, e.g. "Welcome the user with a short sentence that is wrapped in quotation marks like this: 'Hi there!'". Another way is to check the returned value by using e.g. the ``is_str()`` function to make sure that it is of the expected format.

.. _make-use-of-llms-user-intent-matching:

----------------------------------------
User Intent Matching
----------------------------------------

In section :ref:`Defining Flows<action-like-and-intent-like-flows>` we have already seen how we can define user intent flows. The limitation was that they did not generalize to variations of the given user intent examples. With the help of an LLM we can overcome this issue and use its reasoning power by importing the `llm` standard library module and activate the flows ``automating intent detection`` and ``generating user intent for unhandled user utterance`` (`Github link <../../../nemoguardrails/colang/v2_x/library/llm.co>`__) to match unexpected user utterances to currently active user intent flows.

.. code-block:: colang
    :caption: llm/user_intent_match_example/main.co

    import core
    import llm

    flow main
        activate automating intent detection
        activate generating user intent for unhandled user utterance

        while True
            when user greeted
                bot say "Hi there!"
            or when user said goodbye
                bot say "Goodbye!"
            or when unhandled user intent # For any user utterance that does not match
                bot say "Thanks for sharing!"

    flow user greeted
        user said "Hi" or user said "Hello"

    flow user said goodbye
        user said "Bye" or user said "See you"

When running this example:

.. code-block:: text

    > Hi

    Hi there!

    > hi

    Hi there!

    > hallo

    Hi there!

    > How are you?

    Thanks for sharing!

    > bye bye

    Goodbye!


You can see that if we have an exact match for e.g. "Hi", the LLM will not be invoked since it matches directly with one of the awaited ``user said`` flows. For any other user utterance the activated flow ``generating user intent for unhandled user utterance`` will invoke the LLM, before finding a suitable user intent. If the user utterance was close enough to one of the predefined user intent flows (i.e. ``user greeted`` or ``user said goodbye``), it will cause the related flow to finish successfully. This enables you to even talk in a different language (if supported by the LLM) to successfully map to the correct flow. If no good match was found, the flow ``unhandled user intent`` will match.

You might ask yourself how the LLM can know which flows are considered user intent flows. This can either be done based on the flow names by activating the flow ``automating intent detection`` to automatically detect flows starting with 'user', or using an explicit flow decorator to mark them independently of their names:

.. code-block:: colang

    @meta(user_intent=True)
    flow any fancy flow name
        user said "Hi" or user said "Hello"

.. note::
    From a semantic point of view it makes always sense to start a user intent flow with 'user' even if marked by a user intent meta decorator.

----------------------------------------
Bot Action Generation
----------------------------------------

Similarly to how we want to be able to handle variations in the user input, we have seen bot intent flows that define a variation of predefined bot actions. While this can be good enough for responses to expected user inputs we would also like to handle unexpected user utterances and not always reply with "Thanks for sharing!". For this case, another flow from the Standard Library will help us named ``llm continue interaction``:

.. code-block:: colang
    :caption: llm/bot_intent_generation_example/main.co

    import core
    import llm

    flow main
        user said something
        llm continue interaction

.. code-block:: text

    > Hello

    Hi there! How can I help you today?

    > Tell me a funny story

    Sure! Did you hear about the fire at the circus? It was intense!

    > funny!

    I'm glad you liked it! Do you want to hear another one?

    > Bye

    Bye! Have a great day!

You see that with this the bot can react to any user input and respond with a suitable bot answer. This generalizes well to multimodal interactions and can be used to generate bot postures and bot gestures as well if provided with a suitable prompting context.

.. note::
    The generated actions strongly depend on the current interaction context, the general prompt instructions and sample conversation in the `config.yml`. Try updating them to achieve the expected results.

----------------------------------------
Basic Interaction Loop
----------------------------------------

We can combine now everything to a basic interaction loop:

.. code-block:: colang
    :caption: llm/interaction_loop/main.co

    import core
    import timing
    import llm

    flow main
        activate automating intent detection
        activate generating user intent for unhandled user utterance

        while True
            when unhandled user intent
                llm continue interaction
            or when user was silent 12.0
                $response = ..."A random fun fact"
                bot say $response
            or when user expressed greeting
                bot say "Hi there!"
            or when user expressed goodbye
                bot inform "That was fun. Goodbye"

    flow user expressed greeting
        user said "hi"
            or user said "hello"

    flow user expressed goodbye
        user said "goodbye"
            or user said "I am done"
            or user said "I have to go"


This loop will take care of matching user utterances to predefined user intents if possible (e.g. ``user expressed greeting`` or ``user expressed goodbye``) or generate a suitable response to unexpected user intents using the flow ``llm continue interaction``. Furthermore, if the user does not say anything for more than 12 seconds, the bot will say a random fun fact generated through a NLD.

-------------
Guardrailing
-------------

Checkout the examples in the :ref:`getting_started` section or refer to the `NeMo Guardrails documentation <https://github.com/NVIDIA/NeMo-Guardrails>`_ to learn more about how Colang can be used to guardrail LLM responses and user inputs.
