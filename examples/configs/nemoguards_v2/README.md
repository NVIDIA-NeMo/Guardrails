# NeMoGuard Safety Rails Example

This example showcases the use of NVIDIA's NeMoGuard NIMs for comprehensive AI safety including content moderation, topic control, and jailbreak detection.

## Configuration Files

- `config.yml` - Defines the models configuration including the main LLM and three NeMoGuard NIMs for safety checks
- `prompts.yml` - Contains prompt templates for content safety and topic control checks
- `rails.co` - Implements input and output rails that integrate content safety, topic safety, and jailbreak detection checks
- `main.co` - The entry point Colang 2 file that imports core functionality and activates the LLM continuation flow

## NeMoGuard NIMs Used

1. **Content Safety** (`nvidia/llama-3.1-nemoguard-8b-content-safety`) - Checks for unsafe content across 23 safety categories
2. **Topic Control** (`nvidia/llama-3.1-nemoguard-8b-topic-control`) - Ensures conversations stay within allowed topics
3. **Jailbreak Detection** - Detects and prevents jailbreak attempts (configured via `nim_server_endpoint`)

## Documentation

For more details about NeMoGuard NIMs and deployment options, see:

- [NeMo Guardrails Documentation](https://docs.nvidia.com/nemo/guardrails/index.html)
- [Llama 3.1 NemoGuard 8B ContentSafety NIM](https://docs.nvidia.com/nim/llama-3-1-nemoguard-8b-contentsafety/latest/)
- [Llama 3.1 NemoGuard 8B TopicControl NIM](https://docs.nvidia.com/nim/llama-3-1-nemoguard-8b-topiccontrol/latest/)
- [NemoGuard JailbreakDetect NIM](https://docs.nvidia.com/nim/nemoguard-jailbreakdetect/latest/)
- [NeMoGuard Models on NVIDIA API Catalog](https://build.nvidia.com/search?q=nemoguard)
