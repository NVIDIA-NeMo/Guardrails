# Guardrails Evaluation

NeMo Guardrails includes a set of tools that you can use to evaluate the different types of rails. In the current version, these tools test the performance of each type of rail individually. You can use the evaluation tools through the `nemoguardrails` CLI. Examples will be provided for each type of rail.

At the same time, we provide preliminary results on the performance of the rails on a set of public datasets that are relevant to each task at hand.

## Dialog Rails

### Aim and Usage

Dialog rails evaluation focuses on NeMo Guardrails's core mechanism to guide conversations using canonical forms and dialogue flows.
More details about this core functionality are explained [here](./../../docs/architecture/README.md).

Thus, when using dialog rails evaluation, we are assessing the performance for:

1. User canonical form (intent) generation.
2. Next step generation - in the current approach, we only assess the performance of bot canonical forms as next step in a flow.
3. Bot message generation.

The CLI command for evaluating the dialog rails is:

```bash
nemoguardrails evaluate topical --config=<rails_app_path> --verbose
```

A dialog rails evaluation has the following CLI parameters:

- `config`: The Guardrails app to be evaluated.
- `verbose`: If the Guardrails app should be run in verbose mode.
- `test-percentage`: Percentage of the samples for an intent to be used as test set.
- `max-tests-intent`: Maximum number of test samples per intent to be used when testing (useful to have balanced test data for unbalanced datasets). If the value is 0, this parameter is not used.
- `max-samples-intent`: Maximum number of samples per intent to be used in the vector database. If the value is 0, all samples not in test set are used.
- `results-frequency`: If we want to print intermediate results about the
current evaluation, this is the step.
- `sim-threshold`: If larger than 0, for intents that do not have an exact match, pick the most similar intent above this threshold.
- `random-seed`: Random seed used by the evaluation.
- `output-dir`: Output directory for predictions.

### Evaluation Results

For the initial evaluation experiments for dialog rails, we have used two datasets for conversational NLU:

- [_chit-chat_](https://github.com/rahul051296/small-talk-rasa-stack) dataset
- [_banking_](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data) dataset

The datasets were transformed into a NeMo Guardrails app by defining canonical forms for each intent, specific dialogue flows, and even bot messages (for the _chit-chat_ dataset alone).
The two datasets have a large number of user intents, thus dialog rails. One of them is very generic and has higher-grained intents (_chit-chat_), while the _banking_ dataset is domain-specific and more fine-grained.
More details about running the dialog rails evaluation experiments and the evaluation datasets are available [here](../../nemoguardrails/evaluate/data/topical/README.md).

Preliminary evaluation results follow next. In all experiments, we have chosen to have a balanced test set with at most 3 samples per intent.
For both datasets, we have assessed the performance for various LLMs and also for the number of samples (`k = all, 3, 1`) per intent that are indexed in the vector database.

Take into account that the performance of an LLM is heavily dependent on the prompt, especially due to the more complex [prompt used by Guardrails](./../../docs/architecture/README.md#example-prompt).
Therefore, currently, we only release the results for OpenAI models, but more results will follow in the next releases. All results are preliminary, as better prompting can improve them.

Important lessons to be learned from the evaluation results:

- Each step in the three-step approach (user intent, next step/bot intent, bot message) used by Guardrails offers an improvement in performance.
- It is important to have at least k=3 samples in the vector database for each user intent (canonical form) to achieve good performance.
- Some models (e.g., gpt-3.5-turbo) produce a wider variety of canonical forms, even with the few-shot prompting used by Guardrails. In these cases, it is useful to add a similarity match instead of exact match for user intents. In this case, the similarity threshold becomes an important inference parameter.
- Initial results show that even small models, e.g. [dolly-v2-3b](https://huggingface.co/databricks/dolly-v2-3b), [vicuna-7b-v1.3](https://huggingface.co/lmsys/vicuna-7b-v1.3), [mpt-7b-instruct](https://huggingface.co/mosaicml/mpt-7b-instruct), [falcon-7b-instruct](https://huggingface.co/tiiuae/falcon-7b-instruct) have good performance for topical rails.
- Using a single call for topical rails shows similar results to the default method (which uses up to 3 LLM calls for generating the final bot message) in most cases for `text-davinci-003` model.
- Initial experiments show that using compact prompts has similar or even better performance on these two datasets compared to using the longer prompts.

Evaluation Date - June 21, 2023. Updated July 24, 2023 for Dolly, Vicuna and Mosaic MPT models. Updated Mar 13 2024 for `gemini-1.0-pro` and `text-bison`.

| Dataset   | # intents | # test samples |
|-----------|-----------|----------------|
| chit-chat | 76        | 226            |
| banking   | 77        | 231            |

Results on _chit-chat_ dataset, metric used is accuracy.

| Model                                  | User intent, `w.o sim` | User intent, `sim=0.6` | Bot intent, `w.o sim` | Bot intent, `sim=0.6` | Bot message, `w.o sim` | Bot message, `sim=0.6` |
|----------------------------------------|------------------------|------------------------|-----------------------|-----------------------|------------------------|------------------------|
| `gpt-3.5-turbo-instruct, k=all`        | 0.88                   | N/A                    | 0.88                  | N/A                   | 0.88                   | N/A                    |
| `gpt-3.5-turbo-instruct, single call`  | 0.90                   | N/A                    | 0.91                  | N/A                   | 0.91                   | N/A                    |
| `gpt-3.5-turbo-instruct, compact`      | 0.89                   | N/A                    | 0.89                  | N/A                   | 0.90                   | N/A                    |
| `gpt-3.5-turbo, k=all`                 | 0.44                   | 0.56                   | 0.50                  | 0.61                  | 0.54                   | 0.65                   |
| `text-davinci-003, k=all`              | 0.89                   | 0.89                   | 0.90                  | 0.90                  | 0.91                   | 0.91                   |
| `text-davinci-003, k=all, single call` | 0.89                   | N/A                    | 0.91                  | N/A                   | 0.91                   | N/A                    |
| `text-davinci-003, k=all, compact`     | 0.90                   | N/A                    | 0.91                  | N/A                   | 0.91                   | N/A                    |
| `text-davinci-003, k=3`                | 0.82                   | N/A                    | 0.85                  | N/A                   | N/A                    | N/A                    |
| `text-davinci-003, k=1`                | 0.65                   | N/A                    | 0.73                  | N/A                   | N/A                    | N/A                    |
| `llama2-13b-chat, k=all`               | 0.87                   | N/A                    | 0.88                  | N/A                   | 0.89                   | N/A                    |
| `dolly-v2-3b, k=all`                   | 0.80                   | 0.82                   | 0.81                  | 0.83                  | 0.81                   | 0.83                   |
| `vicuna-7b-v1.3, k=all`                | 0.62                   | 0.75                   | 0.69                  | 0.77                  | 0.71                   | 0.79                   |
| `mpt-7b-instruct, k=all`               | 0.73                   | 0.81                   | 0.78                  | 0.82                  | 0.80                   | 0.82                   |
| `falcon-7b-instruct, k=all`            | 0.81                   | 0.81                   | 0.81                  | 0.82                  | 0.82                   | 0.82                   |
| `gemini-1.0-pro`                       | 0.79                   | 0.79                   | 0.80                  | 0.80                  | 0.80                   | 0.80                   |
| `gemini-1.0-pro, single call`          | 0.76                   | 0.76                   | 0.78                  | 0.77                  | 0.78                   | 0.77                   |
| `text-bison`                           | 0.63                   | 0.75                   | 0.67                  | 0.78                  | 0.70                   | 0.79                   |
| `text-bison, single call`              | 0.65                   | 0.75                   | 0.71                  | 0.77                  | 0.73                   | 0.80                   |

Results on _banking_ dataset, metric used is accuracy.

| Model                                  | User intent, `w.o sim` | User intent, `sim=0.6` | Bot intent, `w.o sim` | Bot intent, `sim=0.6` | Bot message, `w.o sim` | Bot message, `sim=0.6` |
|----------------------------------------|------------------------|------------------------|-----------------------|-----------------------|------------------------|------------------------|
| `gpt-3.5-turbo-instruct, k=all`        | 0.73                   | N/A                    | 0.74                  | N/A                   | N/A                    | N/A                    |
| `gpt-3.5-turbo-instruct, single call`  | 0.81                   | N/A                    | 0.83                  | N/A                   | N/A                    | N/A                    |
| `gpt-3.5-turbo-instruct, compact`      | 0.86                   | N/A                    | 0.87                  | N/A                   | N/A                    | N/A                    |
| `gpt-3.5-turbo, k=all`                 | 0.38                   | 0.73                   | 0.45                  | 0.73                  | N/A                    | N/A                    |
| `text-davinci-003, k=all`              | 0.77                   | 0.82                   | 0.83                  | 0.84                  | N/A                    | N/A                    |
| `text-davinci-003, k=all, single call` | 0.75                   | N/A                    | 0.81                  | N/A                   | N/A                    | N/A                    |
| `text-davinci-003, k=all, compact`     | 0.86                   | N/A                    | 0.86                  | N/A                   | N/A                    | N/A                    |
| `text-davinci-003, k=3`                | 0.65                   | N/A                    | 0.73                  | N/A                   | N/A                    | N/A                    |
| `text-davinci-003, k=1`                | 0.50                   | N/A                    | 0.63                  | N/A                   | N/A                    | N/A                    |
| `llama2-13b-chat, k=all`               | 0.76                   | N/A                    | 0.77                  | N/A                   | N/A                    | N/A                    |
| `dolly-v2-3b, k=all`                   | 0.32                   | 0.62                   | 0.40                  | 0.64                  | N/A                    | N/A                    |
| `vicuna-7b-v1.3, k=all`                | 0.39                   | 0.62                   | 0.54                  | 0.65                  | N/A                    | N/A                    |
| `mpt-7b-instruct, k=all`               | 0.45                   | 0.58                   | 0.50                  | 0.60                  | N/A                    | N/A                    |
| `falcon-7b-instruct, k=all`            | 0.70                   | 0.75                   | 0.76                  | 0.78                  | N/A                    | N/A                    |
| `gemini-1.0-pro`                       | 0.89                   | 0.88                   | 0.87                  | 0.91                  | N/A                    | N/A                    |
| `gemini-1.0-pro, single call`          | 0.89                   | 0.89                   | 0.90                  | 0.89                  | N/A                    | N/A                    |
| `text-bison`                           | 0.85                   | 0.92                   | 0.89                  | 0.94                  | N/A                    | N/A                    |
| `text-bison, single call`              | 0.91                   | 0.89                   | 0.92                  | 0.90                  | N/A                    | N/A                    |

## Input and Output Rails

### Fact-checking Rails

In the [Guardrails library](./../../docs/user-guides/guardrails-library.md), we provide two approaches out of the box for the fact-checking rail: the Self-Check fact-checking and AlignScore. For more details, read the [library guide](./../../docs/user-guides/guardrails-library.md).

#### Self-Check

In this approach, the fact-checking rail is implemented as an entailment prediction problem. Given an evidence passage and the predicted answer, we prompt an LLM to predict yes/no whether the answer is grounded in the evidence or not. This is the default approach.

#### AlignScore

This approach is based on the AlignScore model [Zha et al. 2023](https://aclanthology.org/2023.acl-long.634.pdf). Given an evidence passage and the predicted answer, the model is finetuned to predict that they are aligned when:

1. All information in the predicted answer is present in the evidence passage, and
2. None of the information in the predicted answer contradicts the evidence passage.
The response is a value between 0.0 and 1.0. In our testing, the best average accuracies were observed with a threshold of 0.7.

Please see the [user guide documentation](./../../docs/user-guides/guardrails-library.md#alignscore) for detailed steps on how to configure your deployment to use AlignScore.

#### Evaluation

To run the fact-checking rail, you can use the following CLI command:

```bash
nemoguardrails evaluate fact-checking --config=path/to/guardrails/config
```

Here is a list of arguments that you can use to configure the fact-checking rail:

- `config`: The path to the guardrails configuration (this includes the LLM, the prompts and any other information).
- `dataset-path`: Path to the dataset. It should be a JSON file with the following format:

    ```
    [
        {
            "question": "question text",
            "answer": "answer text",
            "evidence": "evidence text",
        },
    }
    ```

- `num-samples`: Number of samples to run the eval on. The default is 50.
- `create-negatives`: Whether to generate synthetic negative examples or not. The default is `True`.
- `output-dir`: The directory to save the output to. The default is `eval_outputs/factchecking`.
- `write-outputs`: Whether to write the outputs to a file or not. The default is `True`.

More details on how to set up the data in the right format and run the evaluation on your own dataset can be found [here](../../nemoguardrails/evaluate/data/factchecking/README.md).

#### Evaluation Results

Evaluation Date - Nov 23, 2023 (Mar 7 2024 for `gemini-1.0-pro`).

We evaluate the performance of the fact-checking rail on the [MSMARCO](https://huggingface.co/datasets/ms_marco) dataset using the Self-Check and the AlignScore approaches. To build the dataset, we randomly sample 100 (question, correct answer, evidence) triples, and then, for each triple, build a non-factual or incorrect answer to yield 100 (question, incorrect answer, evidence) triples.

We breakdown the performance into positive entailment accuracy and negative entailment accuracy. Positive entailment accuracy is the accuracy of the model in correctly identifying answers that are grounded in the evidence passage. Negative entailment accuracy is the accuracy of the model in correctly identifying answers that are **not** supported in the evidence. Details on how to create synthetic negative examples can be found [here](../../nemoguardrails/evaluate/data/factchecking/README.md)

| Model                  | Positive Entailment Accuracy | Negative Entailment Accuracy | Overall Accuracy | Average Time Per Checked Fact (ms) |
|------------------------|------------------------------|------------------------------|------------------|------------------------------------|
| gpt-3.5-turbo-instruct | **92.0%**                    | 69.0%                        | 80.5%            | 188.8ms                            |
| gpt-3.5-turbo          | 76.0%                        | 89.0%                        | 82.5%            | 435.1ms                            |
| text-davinci-003       | 70.0%                        | **93.0%**                    | 81.5%            | 272.2ms                            |
| gemini-1.0-pro         | **92.0%**                    | **93.0%**                    | **92.5%**        | 704.5ms                            |
| align_score-base*      | 81.0%                        | 88.0%                        | 84.5%            | **23.0ms** ^                       |
| align_score-large*     | 87.0%                        | 90.0%                        | **88.5%**        | 46.0ms ^                           |

*The threshold used for align_score is 0.7, i.e. an align_score >= 0.7 is considered a factual statement, and an align_score < 0.7 signifies an incorrect statement.
^When the AlignScore model is loaded in-memory and inference is carried out without network overheads, i.e., not as a RESTful service.

### Moderation Rails

The moderation involves two components: input and output moderation.

- The input moderation attempts to block user inputs that are designed to elicit harmful responses from the bot.
- The output moderation attempts to filter the language model output to avoid unsafe content from being displayed to the user.

#### Self-Check

This rail will prompt the LLM using a custom prompt for input (jailbreak) and output moderation.
Common reasons for rejecting the input from the user include jailbreak attempts, harmful or abusive content, or other inappropriate instructions.
For more details, consult the [Guardrails library]([Guardrails library](./../../docs/user-guides/guardrails-library.md)) guide.

#### Evaluation

The jailbreak and output moderation can be evaluated using the following CLI command:

```bash
nemoguardrails evaluate moderation --config=path/to/guardrails/config
```

The various arguments that can be passed to evaluate the moderation rails are

- `config`: The path to the guardrails configuration (this includes the LLM, the prompts and any other information).
- `dataset-path`: Path to the dataset to evaluate the rails on. The dataset should contain one prompt per line.
- `split`: The split of the dataset to evaluate on. Choices are 'helpful' or 'harmful'. This selection is used to determine the appropriate label for the predictions.
- `num-samples`: Number of samples to evaluate. Default is 50.
- `check-input`: Whether to evaluate the input moderation rail. Default is True.
- `check-output`: Whether to evaluate the output moderation rail. Default is True.
- `output-path`: Folder to write the results to. Default is 'eval_outputs/moderation'.

It is also possible to evaluate each of the rails individually. To evaluate the input rail only, use the following command:

```nemoguardrails evaluate moderation --check-output False --config=path/to/guardrails/config```

To evaluate the output moderation rail only, use the following command:

```nemoguardrails evaluate moderation --check-input False --config=path/to/guardrails/config```

More details on how to set up the data in the right format and run the evaluation on your own dataset can be found [here](../../nemoguardrails/evaluate/data/moderation/README.md).

#### Evaluation Results

We evaluate the moderation rails on the Anthropic [Red Team Attempts dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf/tree/main/red-team-attempts) and the Anthropic [Helpful Base dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf/tree/main/helpful-base). This dataset contains prompts that are labeled by humans as either helpful or harmful. We randomly sample 100 prompts from each of the splits and run the evaluation using OpenAI `text-davinci-003` and `gpt-3.5-turbo` models.

Evaluation Date - June 02, 2023 (Mar 13 2024 for `gemini-1.0-pro`).

We breakdown the performance of the models on the two rails into the following metrics:

- % of the prompts that are **blocked** on the Red Team Attempts dataset
- % of the prompts that are **allowed** on the Helpful Base dataset

We want the models to block as many harmful prompts as possible and allow as many helpful prompts as possible.

#### Moderation Rails Performance

These results are using the _Simple_ prompt defined in the LLM Self-Checking method. For more details, see the [Guardrails library](./../../docs/user-guides/guardrails-library.md).

| Model                  | % of harmful prompts blocked | % harmful prompts triggering model errors | % of helpful prompts allowed |
|------------------------|------------------------------|-------------------------------------------|------------------------------|
| gpt-3.5-turbo-instruct | 78                           | 0                                         | 97                           |
| gpt-3.5-turbo          | 70                           | 0                                         | 100                          |
| text-davinci-003       | 80                           | 0                                         | 97                           |
| gemini-1.0-pro         | 63                           | 36<sup>*</sup>                            | 97                           |

<sup>*</sup> Note that as of Mar 13, 2024 `gemini-1.0-pro` when queried via the Vertex AI API occasionally produces [this error](https://github.com/GoogleCloudPlatform/generative-ai/issues/344). Note that this occurs with a self check prompt, that is when the model is given an input where it is asked to give a yes / no answer to whether it should respond to a particular input. We report these separately since this behavior is triggered by the self check prompt itself in which case it is debatable whether this behavior should be treated as effective moderation or being triggered by a false positive.

##### LlamaGuard-based Moderation Rails Performance

Evaluation date: January 8, 2024.

Guardrails offers out-of-the-box support for Meta's new Llama Guard model for input/output moderation.
Below, we evaluate Llama Guard and compare it to the self-checking approach with the _Complex_ prompt for two popular datasets.

Results on the OpenAI Moderation test set
Dataset size: 1,680
Number of user inputs labeled harmful: 552 (31.1%)

| Main LLM               | Input Rail               | Accuracy | Precision | Recall | F1 score |
|------------------------|--------------------------|----------|-----------|--------|----------|
| gpt-3.5-turbo-instruct | self check input         | 65.9%    | 0.47      | 0.88   | 0.62     |
| gpt-3.5-turbo-instruct | llama guard check input  | 81.9%    | 0.73      | 0.66   | 0.69     |

Results on the ToxicChat dataset:
Dataset size: 10,165
Number of user inputs labeled harmful: 730 (7.2%)

| Main LLM               | Input Rail               | Accuracy | Precision | Recall | F1 score |
|------------------------|--------------------------|----------|-----------|--------|----------|
| gpt-3.5-turbo-instruct | self check input         | 66.5%    | 0.16      | 0.85   | 0.27     |
| gpt-3.5-turbo-instruct | llama guard check input  | 94.4%    | 0.67      | 0.44   | 0.53     |

The low precision and high recall numbers from the self check input with the complex prompt indicates an overly defensive behavior from the self check input rail. We will run this evaluation with more variations of the self check prompt and report numbers.

### Hallucination Rails

For general questions that the model uses parametric knowledge to answer, we can define a hallucination rail to detect when the model is potentially making up facts. The default implementation of the hallucination rails is based on [SelfCheckGPT](https://arxiv.org/abs/2303.08896).

- Given a question, we sample multiple answers from the model, often at a high temperature (temp=1.0).
- We then check if the answers are consistent with each other. This agreement check is implemented using an LLM call similar to the fact checking rail.
- If the answers are inconsistent, it indicates that the model might be hallucinating.

#### Self-Check

This rail will use the LLM for self-checking with a custom prompt if the answers are inconsistent. The custom prompt can be similar to an NLI task.
For more details, consult the [Guardrails library]([Guardrails library](./../../docs/user-guides/guardrails-library.md)) guide.

#### Evaluation

To run the hallucination rail, use the following CLI command:

```bash
nemoguardrails evaluate hallucination --config=path/to/guardrails/config
```

Here is a list of arguments that you can use to configure the hallucination rail:

- `config`: The path to the guardrails configuration (this includes the LLM, the prompts and any other information).
- `dataset-path`: Path to the dataset. It should be a text file with one question per line.
- `num-samples`: Number of samples to run the eval on. Default is 50.
- `output-dir`: The directory to save the output to. Default is eval_outputs/hallucination.
- `write-outputs`: Whether to write the outputs to a file or not. Default is True.

To evaluate the hallucination rail on your own dataset, you can follow the create a text file with the list of questions and run the evaluation using the following command

```nemoguardrails evaluate hallucination --dataset-path <path-to-your-text-file>```

#### Evaluation Results

To evaluate the hallucination rail, we manually curate a set of [questions](../../nemoguardrails/evaluate/data/hallucination/sample.txt) which mainly consists of questions with a false premise, i.e., questions that cannot have a correct answer.

For example, the question "What is the capital of the moon?" has a false premise since the moon does not have a capital. Since the question is stated in a way that implies that the moon has a capital, the model might be tempted to make up a fact and answer the question.

We then run the hallucination rail on these questions and check if the model is able to detect the hallucination. We run the evaluation using OpenAI `text-davinci-003` and `gpt-3.5-turbo` models.

Evaluation Date - June 12, 2023 (Mar 13 2024 for `gemini-1.0-pro`).

We breakdown the performance into the following metrics:

- % of questions that are intercepted by the model, i.e., % of questions where the model detects are not answerable
- % of questions that are intercepted by model + hallucination rail, i.e., % of questions where the either the model detects are not answerable or the hallucination rail detects that the model is making up facts

| Model            | % intercepted - model | % intercepted - model + hallucination rail | % model errored out |
|------------------|-----------------------|--------------------------------------------|---------------------|
| text-davinci-003 | 0                     | 70                                         | 0                   |                                       |
| gpt-3.5-turbo    | 65                    | 90                                         | 0                   |                                       |
| gemini-1.0-pro   | 60                    | 80                                         | 6.7<sup>*</sup>     |

We find that gpt-3.5-turbo is able to intercept 65% of the questions and identify them as not answerable on its own. Adding the hallucination rail helps intercepts 25% more questions and prevents the model from making up facts.

<sup>*</sup> Vertex AI models sometimes error out on hallucination and moderation tests due to [this issue](https://github.com/GoogleCloudPlatform/generative-ai/issues/344).
