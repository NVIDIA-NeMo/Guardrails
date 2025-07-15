# Research on Guardrails

This document summarizes the most important public research on existing guardrails techniques.
We present only the most relevant papers, including surveys, together with their accompanying code repository if there is one.

While the number of recent works on various guardrails topics is quite high, we aim to only present a curated selection.
We also want that this selection to inform our feature roadmap, deciding on what new methods published as a research paper to add to the NeMo Guardrails repository.

The guardrails categories used below follow the ones present in the [Guardrails library](./user-guides/guardrails-library.md).
For each category we present a list of relevant surveys, existing research papers already supported in NeMo Guardrails, and the curated list of selected papers that might influence our roadmap.

## Hallucination rails

Relevant surveys on hallucination detection and checking factuality for large language models.

- Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., ... & Fung, P. (2023). Survey of hallucination in natural language generation. ACM Computing Surveys, 55(12), 1-38. [paper](https://dl.acm.org/doi/pdf/10.1145/3571730)
- Zhang, Y., Li, Y., Cui, L., Cai, D., Liu, L., Fu, T., ... & Shi, S. (2023). Siren's song in the AI ocean: a survey on hallucination in large language models. arXiv preprint arXiv:2309.01219. [paper](https://arxiv.org/pdf/2309.01219.pdf)
- Huang, L., Yu, W., Ma, W., Zhong, W., Feng, Z., Wang, H., ... & Liu, T. (2023). A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. arXiv preprint arXiv:2311.05232. [paper](https://arxiv.org/pdf/2311.05232.pdf)
- Wang, C., Liu, X., Yue, Y., Tang, X., Zhang, T., Jiayang, C., ... & Zhang, Y. (2023). Survey on factuality in large language models: Knowledge, retrieval and domain-specificity. arXiv preprint arXiv:2310.07521. [paper](https://arxiv.org/pdf/2310.07521.pdf) | [repo](https://github.com/wangcunxiang/LLM-Factuality-Survey)

### Fact-checking (implicit hallucination) rails

Supported in NeMo Guardrails:

- Zha, Y., Yang, Y., Li, R., & Hu, Z. (2023). AlignScore: Evaluating factual consistency with a unified alignment function. arXiv preprint arXiv:2305.16739. [paper](https://arxiv.org/pdf/2305.16739.pdf) | [repo](https://github.com/yuh-zha/AlignScore)

Relevant papers:
- Min, S., Krishna, K., Lyu, X., Lewis, M., Yih, W. T., Koh, P. W., ... & Hajishirzi, H. (2023). Factscore: Fine-grained atomic evaluation of factual precision in long form text generation. arXiv preprint arXiv:2305.14251. [paper](https://arxiv.org/pdf/2305.14251.pdf) | [repo](https://github.com/shmsw25/FActScore)

### Explicit hallucination rails

Supported in NeMo Guardrails:

- _(Similar to)_ Manakul, P., Liusie, A., & Gales, M. J. (2023). Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language models. arXiv preprint arXiv:2303.08896. [paper](https://arxiv.org/pdf/2303.08896.pdf) | [repo](https://github.com/potsawee/selfcheckgpt)

## Moderation rails

Supported in NeMo Guardrails:

- Inan, H., Upasani, K., Chi, J., Rungta, R., Iyer, K., Mao, Y., ... & Khabsa, M. (2023). Llama guard: Llm-based input-output safeguard for human-ai conversations. arXiv preprint arXiv:2312.06674. [paper](https://arxiv.org/pdf/2312.06674.pdf) | [repo](https://github.com/meta-llama/PurpleLlama/tree/main/Llama-Guard)

Relevant papers:

- Markov, T., Zhang, C., Agarwal, S., Nekoul, F. E., Lee, T., Adler, S., ... & Weng, L. (2023, June). A holistic approach to undesired content detection in the real world. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 37, No. 12, pp. 15009-15018). [paper](https://ojs.aaai.org/index.php/AAAI/article/download/26752/26524) | [repo (dataset only)](https://github.com/openai/moderation-api-release)

## Jailbreaking rails

Relevant surveys:
- Yao, Y., Duan, J., Xu, K., Cai, Y., Sun, Z., & Zhang, Y. (2024). A survey on large language model (llm) security and privacy: The good, the bad, and the ugly. High-Confidence Computing, 100211. paper

Supported in NeMo Guardrails:
- _(Similar to)_ Alon, G., & Kamfonas, M. (2023). Detecting language model attacks with perplexity. arXiv preprint arXiv:2308.14132. [paper](https://arxiv.org/pdf/2308.14132.pdf) | repo

Relevant papers:
- Kumar, A., Agarwal, C., Srinivas, S., Feizi, S., & Lakkaraju, H. (2023). Certifying llm safety against adversarial prompting. arXiv preprint arXiv:2309.02705. [paper](https://arxiv.org/pdf/2309.02705.pdf) | [repo](https://github.com/aounon/certified-llm-safety)
- Wei, Z., Wang, Y., & Wang, Y. (2023). Jailbreak and guard aligned language models with only few in-context demonstrations. arXiv preprint arXiv:2310.06387. [paper](https://arxiv.org/pdf/2310.06387.pdf)
- Zhang, Y., Ding, L., Zhang, L., & Tao, D. (2024). Intention analysis prompting makes large language models a good jailbreak defender. arXiv preprint arXiv:2401.06561. [paper](https://arxiv.org/pdf/2401.06561.pdf)
- Xu, Z., Jiang, F., Niu, L., Jia, J., Lin, B. Y., & Poovendran, R. (2024). SafeDecoding: Defending against Jailbreak Attacks via Safety-Aware Decoding. arXiv preprint arXiv:2402.08983. [paper](https://arxiv.org/pdf/2402.08983.pdf) | [repo](https://github.com/uw-nsl/SafeDecoding)
- Ji, J., Hou, B., Robey, A., Pappas, G. J., Hassani, H., Zhang, Y., ... & Chang, S. (2024). Defending Large Language Models against Jailbreak Attacks via Semantic Smoothing. arXiv preprint arXiv:2402.16192. [paper](https://arxiv.org/pdf/2402.16192.pdf) | [code](https://github.com/UCSB-NLP-Chang/SemanticSmooth)

## Dialog rails


Supported in NeMo Guardrails:
- Rebedea, T., Dinu, R., Sreedhar, M., Parisien, C., & Cohen, J. (2023). Nemo guardrails: A toolkit for controllable and safe llm applications with programmable rails. arXiv preprint arXiv:2310.10501. [paper](https://arxiv.org/pdf/2310.10501.pdf) | [code](https://github.com/NVIDIA/NeMo-Guardrails)

Relevant papers:
- Sun, A. Y., Nair, V., Schumacher, E., & Kannan, A. (2023). CONSCENDI: A Contrastive and Scenario-Guided Distillation Approach to Guardrail Models for Virtual Assistants. arXiv preprint arXiv:2304.14364. [paper](https://arxiv.org/pdf/2304.14364.pdf) | [code](https://github.com/curai/curai-research/tree/main/CONSCENDI)
