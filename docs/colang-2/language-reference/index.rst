.. _colang_2_language_reference:

Language Reference
===================

.. .. note::
..     Feedbacks & TODOs:

..     - Update Colang highlighting ('for' and 'continue' in flow names)
..     - Add teaser image

.. toctree::
   :hidden:

   introduction.rst
   event-generation-and-matching.rst
   working-with-actions.rst
   defining-flows.rst
   working-with-variables-and-expressions.rst
   flow-control.rst
   the-standard-library.rst
   make-use-of-llms.rst
   more-on-flows.rst
   python-actions.rst
   development-and-debugging.rst
..    intent-slot-models-and-rags.rst

This chapter is a comprehensive introduction to Colang. Explaining all important concepts in a bottom up approach.

* :ref:`reference_introduction`
* :ref:`event-generation-and-matching`
* :ref:`working-with-actions`
* :ref:`defining-flows`
* :ref:`working-with-variables-and-expressions`
* :ref:`flow-control`
* :ref:`the-standard-library`
* :ref:`make-use-of-llms`
* :ref:`more-on-flows`
* :ref:`python-actions`
* :ref:`development-and-debugging`

.. * :ref:`intent-slot-models-and-rags`

.. .. code-block:: python

..     # Hallo sdfdsfdsfsdfdsf
..     @test("123")
..     def test()
..     """This is a test"""
..         print("Hello")
..         test = 123
..         b = True or False
..         while True:
..             for a in b:
..                 break

.. .. code-block:: colang
..     :caption: Colang lexer test code

..     @override
..     @loop_id("test")
..     flow this is another to continue $in_var = "default" -> $out_var_1 = "test" $out_var_2 = 23.34
..         """This is a magic flow"""
..         activate manage variables
..         global $var = 0
..         priority 1.0
..         b = True or False

..         send StartUtteranceBotAction(script="Hi") as $event
..         match UtteranceUserActionFinished(final_transcript=$event.final_transcript) # Comment
..         start UtteranceBotAction(script="Hi")
..         match UtteranceUserAction.Finished(final_transcript=regex("\\w*Hi"))
..         await bot say await for continue in test is not $param_1=["a","b"] {"a":1, "b",2} 345
..         bot bot say await for continue in test is not $param_1=["a","b"] {"a":1, "b",2} 345
..             and bot say "Hallo"
..             or bot say "Test"
..             and (
..                 bot gesture "wink" or bot gesture "wave"
..             )

..         while $var <= 4
..             if $var == 3
..                 bot say "Hi"
..                 break
..             elif $var > 5
..                 bot say "Hi"
..                 continue
..             else
..                 break

..         # This is a comment
..         when user said "Hi"
..             $instruction = ..."This is an LLM instruction"
..             break
..         or when user said something
..             log "Yes 'sdf'!"
..             return
..         else
..             print "All good"
