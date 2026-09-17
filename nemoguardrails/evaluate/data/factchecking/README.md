# Running the Evaluation for Fact-Checking Rails

To run the fact-checking rail, you can use the following command:

```nemoguardrails eval rail fact-checking --config=path/to/guardrails/config```

Here is a list of arguments that you can use to configure the fact checking rail:

- `config`: The path to the guardrails configuration (this includes the LLM, the prompts and any other information).
- `dataset-path`: Path to the dataset.
- `num-samples`: Number of samples to run the eval on. Default is 50.
- `create-negatives`: Whether to generate synthetic negative examples or not. Default is True.
- `output-dir`: The directory to save the output to. Default is eval_outputs/factchecking.
- `write-outputs`: Whether to write the outputs to a file or not. Default is True.

## Data Format

We require the input data to be in a json file with the following format:

```
{
    "question": "question text",
    "answer": "answer text",
    "evidence": "evidence text",
},
```
## Generating Synthetic Negative Examples

Usually, most datasets contain only positive entailment pairs, i.e., the answers are always grounded in the evidence passage. To reliably evaluate the fact checking rail, we need negative examples as well, i.e., examples where the answer is _not_ grounded in the evidence passage. Randomly sampling answers for other questions to be used as negatives leads to very easily identifiable negatives.

To mine hard negatives, we used OpenAI text-davinci-003 to convert the positive entailment pairs to negative ones. We give the model the evidence and the answer, and ask it to subtly modify the answer to make it not grounded in the evidence. We then use the modified answer as a negative example.

Example:

```
"question": What were the results of the blood pressure test?
"evidence": "On low-dose ACE inhibitor and beta blocker.  Blood pressure today uncharacteristically elevated, 150/70.  He attributes this to rushing to get to his appointment with me.  Blood pressure earlier this month was 120/55 at his pulmonary visit.  I suspect that he is probably well controlled overall.  For now, no change in management."
"answer": "Today, the patient's blood pressure was uncharacteristically elevated at 150/70."
```

The model modifies the answer to:

```
"incorrect answer": "Today, the patient's blood pressure was uncharacteristically elevated at 160/80."
```

By changing small details like the blood pressure value in the answer, the model is able to generate hard negatives that are not easily identifiable. This can be used to evaluate the fact checking rail.


In case you already have negative samples in your dataset, you can set the `create-negatives` flag to `False` and we will not generate synthetic negatives.

```nemoguardrails eval rail fact-checking --config=path/to/guardrails/config --create-negatives False```

## Using the MSMARCO Dataset

The [MSMARCO](https://huggingface.co/datasets/ms_marco) dataset contains a large number of question-answer pairs along with the evidence passage. We can use this dataset to evaluate the fact checking rail. To download and convert the dataset to the required format, the `datasets` library has to be installed using:

```pip install datasets```

Once the library is installed, you can run the following command to download and convert the dataset:

```python process_ms_marco.py```

This will save the dataset to `msmarco.json`. An example from the dataset is shown below:

```
[
    {
        "question": "what is a corporation?",
        "answer": "A corporation is a company or group of people authorized to act as a single entity and recognized as such in law.",
        "evidence": "McDonald's Corporation is one of the most recognizable corporations in the world. A corporation is a company or group of people authorized to act as a single entity (legally a person) and recognized as such in law. Early incorporated entities were established by charter (i.e. by an ad hoc act granted by a monarch or passed by a parliament or legislature)."

    }
]
```

You can then use this file to evaluate the fact checking rail.

```nemoguardrails eval rail fact-checking --config=path/to/guardrails/config --dataset-path msmarco.json```
