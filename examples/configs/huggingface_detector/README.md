# Huggingface Detector

This example showcases how to use inline Huggingface text classification models for content detection in NeMo Guardrails.

## Overview

The Huggingface detector allows you to use any text classification model from Huggingface to detect
and block specific content categories in user inputs or bot outputs. All models are run locally.

## Prerequisites

Install the `transformers` library:

```bash
pip install transformers
```

For GPU acceleration (recommended):

```bash
pip install transformers torch
```

## Configuration

The example `config.yml` demonstrates:

1. **Multiple Model Configuration**: Configure multiple Huggingface models with different purposes
2. **Device Specification**: Set which device (CPU/GPU) each model should use
3. **Input and Output Checking**: Apply different models to user inputs and bot outputs
4. **Flexible Class Blocking**: Use either class labels or indices to specify which classes trigger blocking

### Key Configuration Options

- **model_repo**: Huggingface model repository ID (e.g., `"ibm-granite/granite-guardian-hap-38m"`)
- **descriptor**: Human-readable description of what the model detects
- **blocked_classes**: List of class labels or indices that should trigger blocking
- **device**: Torch device to load the model onto (`"cuda"`, `"cpu"`, `"cuda:0"`, etc.)

### Device Configuration

The `device` field allows you to optimize performance:
- Use `"cuda"` for GPU acceleration (faster inference, requires GPU)
- Use `"cpu"` for CPU inference (slower but works on any machine)
- Use `"cuda:0"`, `"cuda:1"`, etc. for specific GPU devices in multi-GPU setups
- Omit the field to use the `transformers` default device

See the `torch.device` documentation for full usage. Different models can use different devices
based on your requirements.

## Running the Example

```bash
nemoguardrails chat --config=examples/configs/huggingface_detector
```

## Provided flows

1. `huggingface detector check input $hf_model=$HF_ORG/$HF_MODEL`
2. `huggingface detector check output $hf_model=$HF_ORG/$HF_MODEL`
3. `huggingface detector check tool input $hf_model=$HF_ORG/$HF_MODEL`
4. `huggingface detector check tool output $hf_model=$HF_ORG/$HF_MODEL`


## Performance Tips

- Use smaller models for faster inference
- Put frequently-used models on GPU, less-used models on CPU
- Models are cached after first load to avoid reloading
- Only models activated in your flows will be loaded
