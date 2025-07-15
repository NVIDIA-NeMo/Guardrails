# Language Providers

This directory contains translation providers used in the evaluation features of NeMo-Guardrails. These providers support dataset translation and multilingual evaluation.

## Overview

Language Providers offer an abstraction layer to handle different translation services (local or remote) in a unified way. All providers inherit from the `TranslationProvider` base class and provide a consistent interface.

## Directory Structure

```
langproviders/
├── base.py              # Base class TranslationProvider
├── local.py             # Local translation providers
├── remote.py            # Remote translation providers
├── configs/             # Configuration files
│   └── translation.yaml # Example translation config
└── README.md            # This file
```

## Available Translation Providers

### Local Providers

#### LocalHFTranslator
A local translation provider using Hugging Face models.

**Supported Models:**
- **M2M100**: Multilingual Many-to-Many translation models (supports 100 languages)
  - https://huggingface.co/facebook/m2m100_1.2B
  - https://huggingface.co/facebook/m2m100_418M
- **MarianMT**: Helsinki-NLP/opus-mt-* models
  - https://huggingface.co/docs/transformers/model_doc/marian

**Example Configuration:**
```yaml
langproviders:
  - language: en,ja
    model_type: local.LocalHFTranslator
    model_name: "Helsinki-NLP/opus-mt-{}"
    hf_args:
      device: "cpu"
```

**Features:**
- No internet connection required
- Privacy-friendly
- Customizable model selection
- Supports GPU/CPU

### Remote Providers

#### DeeplTranslator
High-quality translation service using the DeepL API. Requires DeepL API key for using it.
- https://www.deepl.com/en/translator

**Example Configuration:**
```yaml
langproviders:
  - language: en,ja
    model_type: remote.DeeplTranslator
```

**Environment Variable:**
```bash
export DEEPL_API_KEY="your-api-key-here"
```

**Features:**
- High-quality translations
- Commercial use available

#### RivaTranslator
Translation service using NVIDIA Riva. Requires an API key for using it.
- https://developer.nvidia.com/riva

**Example Configuration:**

**For Remote Riva Server:**
```yaml
langproviders:
  - language: en,ja
    model_type: remote.RivaTranslator
    local_mode: false
```

**For Local Riva Server:**
```yaml
langproviders:
  - language: en,ja
    model_type: remote.RivaTranslator
    local_mode: true
    uri: "localhost:50051"
```

**Environment Variable:**
```bash
export RIVA_API_KEY="your-api-key-here"
```

**Features:**
- Optimized for NVIDIA GPUs
- Supports both local and cloud deployment
- Low latency
- Configurable endpoints via YAML

## Usage

### 1. Create a Configuration File

Create a translation configuration file (e.g., `translation_config.yaml`):

```yaml
langproviders:
  - language: en,ja
    model_type: remote.DeeplTranslator
```

### 2. Use in Your Program

```python
from nemoguardrails.evaluate.utils_translate import _load_langprovider

# Load the translation provider
translator = _load_langprovider("translation_config.yaml")

# Translate text
translated_text = translator._translate("Hello, world!")
print(translated_text)  # "こんにちは、世界！"
```

### 3. Translate a Dataset

```python
from nemoguardrails.evaluate.utils_translate import load_dataset

# Load and translate a dataset
translated_dataset = load_dataset(
    "dataset.json",
    translation_config="translation_config.yaml"
)
```

### Translation with NeMo Guardrails Evaluation

NeMo Guardrails supports multilingual evaluation through translation providers. This allows you to evaluate your guardrails configuration on datasets in different languages.

#### Supported Evaluation Types

**1. Moderation Evaluation**
Evaluates input and output moderation rails on translated datasets.

```bash
nemoguardrails eval rail moderation \
  --config examples/configs/llm/my_config \
  --dataset-path nemoguardrails/evaluate/data/moderation/harmful.txt \
  --translation-config translation_config.yaml \
  --enable-translation \
  --num-samples 50
```

**2. Hallucination Evaluation**
Evaluates hallucination detection rails on translated datasets.

```bash
nemoguardrails eval rail hallucination \
  --config examples/configs/llm/my_config \
  --dataset-path nemoguardrails/evaluate/data/hallucination/sample.txt \
  --translation-config translation_config.yaml \
  --enable-translation \
  --num-samples 50
```

**3. Fact-Checking Evaluation**
Evaluates fact-checking rails on translated datasets.

```bash
nemoguardrails eval rail fact-checking \
  --config examples/configs/llm/my_config \
  --dataset-path nemoguardrails/evaluate/data/factchecking/sample.json \
  --translation-config translation_config.yaml \
  --enable-translation \
  --num-samples 50
```

#### Translation Configuration Examples

**For Japanese Translation (DeepL):**
```yaml
# translation_config.yaml
langproviders:
  - language: en,ja
    model_type: remote.DeeplTranslator
```

**For Japanese Translation (Local HF):**
```yaml
# translation_config.yaml
langproviders:
  - language: en,ja
    model_type: local.LocalHFTranslator
    model_name: facebook/m2m100_1.2B
    hf_args:
      device: "cpu"
```

**For Chinese Translation:**
```yaml
# translation_config.yaml
langproviders:
  - language: en,zh
    model_type: remote.DeeplTranslator
```

#### Translation Cache

The evaluation system automatically caches translations to avoid repeated API calls and improve performance. Cache files are stored in the `translation_cache/` directory.

**Cache Benefits:**
- Faster subsequent evaluations
- Reduced API costs
- Consistent translations across runs

**Cache Management:**
```bash
# Clear translation cache (if needed)
rm -rf translation_cache/
```

#### Dataset Format Support

The translation system supports both text and JSON datasets:

**Text Files (.txt):**
```
Question 1
Question 2
Question 3
```

**JSON Files (.json):**
```json
[
  {
    "question": "What is the capital of France?",
    "evidence": "Paris is the capital of France.",
    "answer": "Paris"
  }
]
```

#### Evaluation Output

Translated evaluations produce the same output format as regular evaluations, but with translated content:

```json
{
  "question": "ディングウェルの畳み込み効果は、どのような環境で最もよく観察されますか？",
  "hallucination_agreement": "no",
  "bot_response": "ディングウェルの畳み込み効果は、高圧環境で最もよく観察されます。",
  "extra_responses": [
    "ディングウェルの畳み込み効果は、低圧環境で観察されます。",
    "この効果は、常温環境で最もよく見られます。"
  ]
}
```

#### Best Practices

1. **Use Local Providers for Privacy**: When working with sensitive data, use `LocalHFTranslator` instead of remote services.

2. **Cache Management**: Keep translation caches for repeated evaluations, but clear them when switching between different translation providers.

3. **Language Pair Validation**: Ensure your translation provider supports the desired language pair before running evaluations.

4. **API Key Management**: For remote providers, set environment variables securely:
   ```bash
   export DEEPL_API_KEY="your-deepl-api-key"
   export RIVA_API_KEY="your-riva-api-key"
   ```

5. **Sample Size**: Start with small sample sizes (`--num-samples 5-10`) to test your setup before running full evaluations.

#### Troubleshooting Translation Issues

**Common Issues:**

1. **Translation Provider Not Available**
   ```
   ⚠ Translation provider not available: PluginConfigurationError: No configuration file provided
   ```
   **Solution:** Check that your translation config file exists and has correct syntax.

2. **API Key Issues**
   ```
   Exception: Put the API key in the DEEPL_API_KEY environment variable
   ```
   **Solution:** Set the required environment variable for your chosen provider.

3. **Unsupported Language Pair**
   ```
   Exception: Language pair en,xx is not supported
   ```
   **Solution:** Check the supported languages section and use a supported language pair.

4. **Network Issues (Remote Providers)**
   ```
   ConnectionError: Failed to connect to translation service
   ```
   **Solution:** Check your internet connection and API service status.

## Configuration Parameters

### Common Parameters

The following parameters pass by the yaml file.

- **`language`**: Language pair for translation (e.g., `"en,ja"`)
- **`model_type`**: Provider type (e.g., `"remote.DeeplTranslator"`)

### LocalHFTranslator-specific Parameters

- `model_name`: Model name (default: `"Helsinki-NLP/opus-mt-{}"`)
- `hf_args`: Hugging Face arguments
  - `device`: Device (`"cpu"` or `"cuda"`)

#### Language Code Overrides (`lang_overrides`)

Some language codes used in translation models differ from standard ISO codes. `LocalHFTranslator` uses an internal dictionary called `lang_overrides` to automatically convert certain language codes to the format expected by the model. For example, the code for Japanese is sometimes expected as `jap` instead of `ja` in some MarianMT models.

- Example: If you specify `ja` (Japanese) as the target language, `LocalHFTranslator` will internally convert it to `jap` when constructing the model name for MarianMT.
- This conversion is handled automatically; you do not need to change your configuration.
- The current overrides are:
  - `ja` → `jap`

This mechanism ensures compatibility with Hugging Face model naming conventions and prevents errors when loading models for certain languages.

### RivaTranslator-specific Parameters

- `local_mode`: Flag to use a local server (default: `false`)

## Supported Languages

### LocalHFTranslator (M2M100)
Supports 100 languages (see the [official documentation](https://huggingface.co/facebook/m2m100_418M#languages-covered) for details)

### DeeplTranslator
Supports languages (see the [official documentation](https://developers.deepl.com/docs/getting-started/supported-languages) for details) :

### RivaTranslator
Supports 77 languages (see the [official documentation](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/translation/translation-overview.html#language-pairs-supported) for details) :

## Error Handling

### Common Errors

1. **Configuration file not found**
   ```
   PluginConfigurationError: No configuration file provided
   ```

2. **API key not set**
   ```
   Exception: Put the API key in the DEEPL_API_KEY environment variable
   ```

3. **Unsupported language pair**
   ```
   Exception: Language pair en,xx is not supported
   ```

### Troubleshooting

1. **Check environment variables**
   ```bash
   echo $DEEPL_API_KEY  # For DeepL
   echo $RIVA_API_KEY   # For Riva
   ```

2. **Check configuration file syntax**
   ```bash
   python -c "import yaml; yaml.safe_load(open('translation_config.yaml'))"
   ```

3. **Check network connection** (for remote providers)

## Environment Variable (ENV_VAR) Usage

Some translation providers (such as RivaTranslator and DeeplTranslator) require an API key for authentication. Each provider expects the API key to be set in a specific environment variable. This environment variable is referenced in the provider implementation as `ENV_VAR`.

- For **DeepL**, set the API key in `DEEPL_API_KEY`:
  ```bash
  export DEEPL_API_KEY="your-api-key-here"
  ```
- For **Riva**, set the API key in `RIVA_API_KEY`:
  ```bash
  export RIVA_API_KEY="your-api-key-here"
  ```

The provider will automatically load the API key from the corresponding environment variable at runtime. If the environment variable is not set or is empty, an error will be raised.

This mechanism allows you to securely manage API keys for different translation services without hardcoding them in configuration files.

## For Developers

### Adding a New Provider

1. Inherit from the `TranslationProvider` base class
2. Implement the required methods:
   - `_load_langprovider()`: Provider initialization
   - `_translate(text: str) -> str`: Translation logic

3. Add your provider to the appropriate file (`local.py` or `remote.py`)

### Testing

```bash
# Run tests for translation providers
python -m pytest tests/eval/translate/ -v
```

## Related Links

- [NeMo-Guardrails Documentation](https://docs.nvidia.com/nemo/guardrails/latest/index.html)
- [DeepL API Documentation](https://developers.deepl.com/)
- [NVIDIA Riva Documentation](https://developer.nvidia.com/riva)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
