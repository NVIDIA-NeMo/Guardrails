------------------------------------------------------------------------------------------------------------------------------------------------------------------
User Attention Flows (`attention.co <../../../nemoguardrails/colang/v2_x/library/attention.co>`_)
------------------------------------------------------------------------------------------------------------------------------------------------------------------

Flows to handle user attention.

.. py:function:: tracking user attention

     For the automatic handling of user attention events, you need to activate this flow to track user attention levels during the last user utterance. This information will be used to change the functionality of all ``user said`` flows such that they will no longer finish when the user says something while being inattentive.

     Example:

     .. code-block:: colang

          import core
          import attention

          flow main
               # Activate the flow at the beginning to make sure user attention events are tracked properly
               activate tracking user attention

               ...

.. py:function:: user said (overwritten)

     When you include ``attention.co`` in your bot folder, it overrides all ``user said`` related flows so that these flows only consider user utterances when the user is attentive. You can overwrite the default attention check by overwriting the flow ``attention checks`` explained below. For your first test, the default implementation should work well with the Tokkio setup.

     Example:

     .. code-block:: colang

          import core
          import attention

          flow main
               activate tracking user attention

               # Since the attention module overwrites all user said related flows, this line will wait until the user says
               # something while being attentive.
               user said something
               bot say "I heard you and you are attentive"

.. py:function:: attention checks $event -> $is_attentive

     The ``attention checks`` flow is called whenever the system needs to decide if a user utterance was completed while the user was attentive. You can overwrite the default behavior by overwriting this flow in your bot script.

     Example:

     .. code-block:: colang

          import core
          import attention

          @override
          flow attention checks $event -> $is_attentive
               # Implement your custom attention logic here
               $is_attentive = True
               return $is_attentive

.. py:function:: user said something inattentively

     The user said something while being inattentive. Use this flow to let the user know that the bot assumes that the user is not attentive and the utterance will be ignored.

     Example:

     .. code-block:: colang

          import core
          import attention
          import avatar # Only needed for the optional bot gesture we use below

          flow main
               activate tracking user attention
               when user said something
                    bot say "I hear you"
               or when user said something inattentively
                    bot say "You seem distracted. Can you repeat?" and bot gesture "asking if something refers to them, being unsure if they're being addressed"
