# Running the Evaluation for Topical Rails

The topical rails can be evaluated using the following command:

```nemoguardrails evaluate topical```

The following arguments can be passed for an evaluation:

- `config`: The Guardrails app to be evaluated.
- `verbose`: If the Guardrails app should be run in verbose mode
- `test-percentage`: Percentage of the samples for an intent to be used as test set
- `max-tests-intent`: Maximum number of test samples per intent to be used when testing
(useful to have balanced test data for unbalanced datasets). If the value is 0,
this parameter is not used.
- `max-samples-intent`: Maximum number of samples per intent to be used in the
vector database. If the value is 0, all samples not in test set are used.
- `results-frequency`: If we want to print intermediate results about the
current evaluation, this is the step.
- `sim-threshold`: If larger than 0, for intents that do not have an exact match
pick the most similar intent above this threshold.
- `random-seed`: Random seed used by the evaluation.
- `output-dir`: Output directory for predictions.


For additional information about topical rails evaluation and results on the two datasets, read the [evaluation tools README file](./../../README.md).


## Chit-chat dataset

We are using a slightly modified version of the chit-chat dataset available [here](https://github.com/rahul051296/small-talk-rasa-stack).
For this dataset, we have configured a [Guardrail app](./chitchat) that already has:
- Config file: `config.yml`
- A set of defined flows: `flows.co`
- A set of predefined bot messages for the topical rails: `bot.co`
- A file mapping the user intents in the original dataset to user canonical forms used by Guardrails: `intent_canonical_forms.json`
- An extra user canonical form Colang file added to the evaluation: `user-other.co`


We still need to create the main Colang file for the user intents defined in the original dataset.
This will take into account the mapping file above. To achieve this follow the next steps:

1. Download the user intents file from the original dataset repository from [here](https://github.com/rahul051296/small-talk-rasa-stack/blob/master/data/nlu.md).
2. Move it to the `nemoguardrails/eval/data/topical/chitchat/original_dataset` folder.
3. Run the conversion script `nemoguardrails/evaluate/data/topical/create_colang_intent_file.py --dataset-name=chitchat --dataset-path=./chitchat/original_dataset/`
4. The last step will create a `user.co` Colang file in the configured Guardrails app.

To run the topical evaluation on this dataset run:

```nemoguardrails evaluate topical --config=./nemoguardrails/eval/data/topical/chitchat --verbose```

## Banking dataset

We are starting from the banking dataset available [here](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data).
For this dataset, we have configured a [Guardrail app](./banking) that already has:
- Config file: `config.yml`
- A set of defined flows: `flows.co`
- A file mapping the user intents in the original dataset to user canonical forms used by Guardrails: `categories_canonical_forms.json`


We still need to create the main Colang file for the user intents defined in the original dataset.
This will take into account the mapping file above. To achieve this follow the next steps:

1. Download the user intents files from the original dataset repository from [here](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data) (bot train and test).
2. Move the two files to the `./nemoguardrails/eval/data/topical/banking/original_dataset` folder.
3. Run the conversion script `./nemoguardrails/evaluate/data/topical/create_colang_intent_file.py --dataset-name=banking --dataset-path=./banking/original_dataset/`
4. The last step will create a `user.co` Colang file in the configured Guardrails app.

To run the topical evaluation on this dataset run:

```nemoguardrails evaluate topical --config=./nemoguardrails/eval/data/topical/banking --verbose```

## Experiment with a new NLU dataset

If you want to assess the performance of topical rails with a new NLU dataset, you can use the `./nemoguardrails/evaluate/data/topical/dataset_tools.py` functionality.
For each dataset, you need to define a new class that extends the `DatasetConnector` class and implements the two following two functions:
- `read_dataset`: Reads the dataset from the specified path, instantiating at least intent names, intent canonical forms, and intent samples.
The path received as parameter should contain the original dataset files, in the specific format they were distributed.
- `_read_canonical_forms`: Reads the intent - canonical form mappings from a file.
This can be a `json` or any other format and should be created by the evaluation user as the mapping is not part of the original dataset.
