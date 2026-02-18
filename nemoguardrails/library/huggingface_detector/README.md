# Huggingface Detector

The Huggingface detector allows you to use any SequenceClassification model from Huggingface Hub to detect and block specific content categories in user inputs, bot outputs, or tool messages. Specifically, this rail supports `transformers` models with  `($xyz)ForSequenceClassification` architectures,
e.g., `DistilBertForSequenceClassification` or `RobertaForSequenceClassification`.

## Features

- Configure multiple Huggingface text classification models
- Selectively activate specific models per flow
- Configure which classes should trigger blocking for each model
- Support for both input and output checking
- Model caching for efficient inference

## Installation

The Huggingface detector requires the `transformers` library:

```bash
pip install transformers torch
```

## Configuration

Add the following to your `config.yml`:

```yaml
  input:
    flows:
      - huggingface detector check input $hf_model="some/hf-model"
  config:
    huggingface_detector:
      models:
        - model_repo: "some/hf-model"
          descriptor: "hate speech detector"
          blocked_classes:
            - "classA"
```

### Configuration Options

- **models** (required): List of model configurations. Each model has:
  - **model_repo** (required): Huggingface model repository ID (e.g., `"ibm-granite/granite-guardian-hap-38m"`)
  - **descriptor** (optional): Human-readable description of what this model detects (e.g., `"Harmful language detector"`). This is useful for providing more informative detection messages if the model repo name is not particularly descriptive.
  - **blocked_classes** (required): List of class labels (strings) or class indices (integers) that should trigger blocking
    - Can use class labels: `["harmful", "violence", "hate_speech"]`
    - Can use class indices: `[0, 1, 2]`
    - Must be either all strings OR all integers per model, not mixed
    - Use indices if your model doesn't provide label mappings (id2label)
  - **device** (optional): Device to load the model onto (e.g., `"cuda"`, `"cpu"`, `"cuda:0"`). If not specified, PyTorch will use its default device selection.

**Important:** Configure all models you want to use here, then specify which model to use in your flows with the `$hf_model` parameter.

## Usage

Configure multiple models and selectively activate specific ones by specifying the model repository parameter.

The flows accept a `$hf_model` parameter to specify which configured model to use:
- `huggingface detector check input $hf_model="<model-repo>"`
- `huggingface detector check output $hf_model="<model-repo>"`

## Example Configuration

### Single Model - Input Checking

Check user inputs using the IBM Granite Guardian HAP model:

```yaml
models:
  - type: main
    engine: openai
    model: gpt-3.5-turbo

rails:
  input:
    flows:
      - huggingface detector check input $hf_model="some/hf-model"
  config:
    huggingface_detector:
      models:
        - model_repo: "some/hf-model"
          descriptor: "hate speech detector"
          blocked_classes:
            - "classA"
```

### Multiple Models - Input Checking

Check inputs against multiple models sequentially:

```yaml
models:
  - type: main
    engine: openai
    model: gpt-3.5-turbo

rails:
  input:
    flows:
      - huggingface detector check input $hf_model="some/hf-model"
      - huggingface detector check input $hf_model="different/hf-model"
  config:
    huggingface_detector:
      models:
        - model_repo: "some/hf-model"
          descriptor: "hate speech detector"
          blocked_classes:
            - "classA"
        - model_repo: "different/hf-model"
          descriptor: "jailbreak detector"
          blocked_classes:
            - "classB"
            - "classD"
```

### Output Checking

Check bot outputs before sending to the user:

```yaml
models:
  - type: main
    engine: openai
    model: gpt-3.5-turbo

rails:
  output:
    flows:
      - huggingface detector check output $hf_model="some/hf-model"
  config:
    huggingface_detector:
      models:
        - model_repo: "some/hf-model"
          descriptor: "hate speech detector"
          blocked_classes:
            - "classA"
```

### Using Class Indices

For models without label mappings, use class indices:

```yaml
rails:
  input:
    flows:
      - huggingface detector check input $hf_model="some-model/text-classifier"
  config:
    huggingface_detector:
      models:
        - model_repo: "some-model/text-classifier"
          blocked_classes: [0, 1]  # Block classes at indices 0 and 1
```

### Specifying Device

To load models on a specific device (CPU or GPU):

```yaml
rails:
  input:
    flows:
      - huggingface detector check input $hf_model="some/hf-model"
  config:
    huggingface_detector:
      models:
        - model_repo: "some/hf-model"
          descriptor: "hate speech detector"
          blocked_classes: ["classA"]
          device: "cuda"  # Load on GPU
        - model_repo: "different/hf-model"
          descriptor: "jailbreak detector"
          blocked_classes: ["classB"]
          device: "cpu"  # Load on CPU
```

## Supported Models

Any Huggingface text classification model should work. Some popular options:

- **HAP Detection**: `ibm-granite/granite-guardian-hap-38m`
- **Prompt Injection**: `protectai/deberta-v3-base-prompt-injection-v2`
- **Content Moderation**: Various models available on Huggingface Hub

## How It Works

1. You configure multiple models in `rails.config.huggingface_detector.models`
2. In your flows, you activate specific models using the `$hf_model` parameter
3. The specified model and tokenizer are loaded (cached after first load)
4. Input/output text is tokenized and passed through the model
5. The model returns classification scores for each class
6. If the top predicted class is in the model's `blocked_classes` list, the content is blocked
7. The result includes the detected class, confidence score, and all class scores

## Return Structure

The detector returns a dictionary with the following structure:

- `allowed`: Boolean indicating if the text is allowed
- `detected_class`: The predicted class label
- `score`: The confidence score for the prediction
- `all_scores`: Dictionary of scores for all classes

**Example return value:**
```python
{
  "allowed": False,
  "detected_class": "harmful",
  "score": 0.95,
  "all_scores": {
    "safe": 0.05,
    "harmful": 0.95,
    "violence": 0.00
  }
}
```

These values are available as flow variables after calling the detector.

## Exception Handling

When `enable_rails_exceptions` is enabled in your config, the detector will raise:
- `HuggingfaceDetectorInputException` for blocked inputs
- `HuggingfaceDetectorOutputException` for blocked outputs

## Performance Considerations
The huggingface_detector rail is intended for use with small predictive text classifiers. Ideally,
ensure that only a few models are called for any given flow, as models are run sequentially and will
introduce latency.

- Models are cached after first load to avoid reloading on each call
- If you activate multiple models in your flows, they run sequentially and will increase latency
- Consider using smaller models for faster inference
- You can specify different devices for different models (e.g., put smaller models on CPU and larger ones on GPU)
- All models run locally, so no external API calls are needed
- Only the models you explicitly activate in your flows will be loaded and run

## Troubleshooting

**Model loading fails**
- Ensure both `transformers` and `torch` are installed: `pip install transformers torch`
- Ensure the model repository ID is correct
- Check your internet connection OR ensure models are pre-cached in the Huggingface Hub cache directory
- Models are downloaded from Huggingface Hub on first use. For offline/air-gapped environments, you need to:
  - Pre-download models using `huggingface-cli download <model-repo>` on a machine with internet access
  - Copy the Huggingface cache directory (usually `~/.cache/huggingface/`) to your offline machine
  - Set the `HF_HOME` or `TRANSFORMERS_CACHE` environment variable to point to the cache directory if needed
- Verify you have enough disk space for the model

**Out of memory errors**
- Try using a smaller model
- Reduce the max_length parameter in tokenization
- Use CPU inference if GPU memory is limited by setting `device: "cpu"` in the model configuration

**ValueError: Model does not provide label mappings**
- Your model doesn't have `id2label` or `label2id` configuration
- Use class indices instead of labels in your `blocked_classes` configuration
- Example: `blocked_classes: [0, 1, 2]` instead of `blocked_classes: ["harmful", "violence"]`

**ValueError: Class label 'X' not found in model's label mapping**
- The label you specified is not recognized by the model
- Check the model's documentation for available labels
- Use class indices instead if you're unsure about label names
- The error message will show all available labels

**ValidationError: blocked_classes validation failed**
- Don't mix class labels and indices in the same model's blocked_classes
- Use either `["label1", "label2"]` OR `[0, 1]`, not `["label1", 1]`
- Each model can independently use labels or indices, but not mixed within one model
