# Domain Hallucination Experiments README

本文档整理 `nemoguardrails/library/domain_hallucination/files` 下当前已有的数据集、实验脚本、主要结果文件和已观察到的结论。统计时间：2026-06-03。

## 1. 数据集

| 文件 | 实际样本数 | 用途 | 类别分布 | 四分类标签分布 |
|---|---:|---|---|---|
| `eval_dataset.json` | 58 | 原始小规模静态评测集；用于快速验证 S1-S4、NeMo baseline、专家模型策略 | real_links 10; hallucinated_links 15; mixed_links 5; no_links 5; typosquatting 5; blacklisted 3; edge_cases 15 | pass 27; warn 12; refine 3; block 16 |
| `expanded_dataset.json` | 151 | 扩展静态评测集；用于扩大 URL/domain/GitHub 幻觉样本覆盖 | real_links 38; hallucinated_links 43; mixed_links 19; no_links 14; typosquatting 19; blacklisted 3; edge_cases 15 | pass 64; warn 12; refine 17; block 58 |
| `full_dataset.json` | 223 | 当前主要静态评测集；由原始集和扩展组合得到，用于最终四分类和阈值搜索 | real_links 73; hallucinated_links 58; mixed_links 26; no_links 19; typosquatting 26; blacklisted 6; edge_cases 15 | pass 104; warn 12; refine 24; block 83 |
| `question_pool_v2.json` | 265 questions | 端到端 E2E 问题池；先诱导 LLM 生成回答，再由独立 verifier 生成 ground truth | 15 个类别，覆盖 API endpoint、安全工具、金融、下载链接、GitHub repo、小众工具、对照组等 | 非预标注；运行时由 DNS/HTTP/GitHub verifier 判定 clean/hallucinated/mixed/no_links |

注意：`eval_dataset.json` 的 metadata 仍写着 `total_samples=120`，但当前文件实际 `test_cases` 为 58 条；引用结果时应以实际 case 数为准。`question_pool_v2.json` 的 metadata 写 `total_questions=250`，但当前嵌套 questions 实际为 265 个。

## 2. 实验方法

| 方法 | 检测对象 | 是否依赖 LLM/API | 当前用途 |
|---|---|---|---|
| `domain_hallucination` | URL、domain、GitHub repo、DNS、HTTP、TLS、WHOIS/RDAP、GitHub API、KB/blacklist/semantic/advanced signals | 非专家模式不依赖 LLM；专家模式依赖配置的 LLM | 主方法 |
| `library/hallucination` NeMo baseline | 回答前后一致性；通过额外 LLM 生成比较是否 hallucinated | 依赖 LLM，多次额外调用 | 实验 A baseline |
| `self_check/facts` | 回答是否忠于上下文 chunks | 依赖上下文和 LLM/模型配置 | 计划中的实验 B baseline |
| `factchecking/align_score` | 回答与上下文的一致性评分 | 依赖模型/上下文 | 计划中的实验 C baseline |
| E2E verifier | 从真实 LLM 回答中抽取链接，再独立做 DNS/HTTP/GitHub 验证 | 生成回答依赖 LLM；ground truth verifier 不依赖 guard | 真实落地 pipeline |

## 3. S1-S4 策略定义

| 策略 | verification level | DNS 失败后是否跳过二级检查 | TLS | WHOIS/RDAP | 专家模型 |
|---|---|---:|---:|---:|---:|
| S1 | full | 否 | 开 | 开 | 可选 |
| S2 | full | 是 | 开 | 开 | 可选 |
| S3 | full | 是 | 开 | 关 | 可选 |
| S4 | http | 是 | 关 | 关 | 可选 |

当前已修改专家策略：专家模型作为 advisory evidence，只允许升级决策，不允许把已有 `block` 降级为 `refine/warn/pass`。

## 4. eval_dataset.json 58 条结果

### 非专家模式优化结果

| 方法 | 结果文件 | Accuracy | URL 幻觉召回率 | FPR | Precision | F1 | Mean latency | P95 latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| S1 full + cache | `opt_S1_cached_full_eval58.json` | 74.14% | 47.37% | 12.82% | 64.29% | 54.55% | 11.87s | 33.02s |
| S2 full + skip DNS-failed secondary | `opt_S2_cached_full_skip_dnsfail_eval58.json` | 74.14% | 47.37% | 12.82% | 64.29% | 54.55% | 5.25s | 15.25s |
| S3 full + skip DNS-failed secondary + no WHOIS | `opt_S3_cached_full_skip_dnsfail_no_whois_eval58.json` | 74.14% | 47.37% | 12.82% | 64.29% | 54.55% | 4.35s | 13.16s |
| S4 http-level | `opt_S4_http_skip_dnsfail_eval58.json` | 74.14% | 89.47% | 33.33% | 56.67% | 69.39% | 1.91s | 8.35s |

解释：S4 速度最快、召回率最高，但误报率也最高；S3 在保持 S1/S2 指标不变的情况下去掉 WHOIS，延迟更低。

### 与 NeMo 原生 hallucination baseline 对比

| 方法 | 结果文件 | Accuracy | URL 幻觉召回率 | FPR | Precision | F1 | Mean latency |
|---|---|---:|---:|---:|---:|---:|---:|
| Domain S3 | `final_A_S3_full_skip_dnsfail_no_whois_vs_nemo_eval58.json` | 77.59% | 57.89% | 12.82% | 68.75% | 62.86% | 3.34s |
| NeMo `library/hallucination` + DeepSeek | 同上 | 56.90% | 94.74% | 61.54% | 42.86% | 59.02% | 6.74s |

解释：NeMo baseline 召回率高，但误报率很高；Domain S3 的 precision 和整体 accuracy 更好。

### 专家模型模式结果

| 方法 | 结果文件 | Accuracy | URL 幻觉召回率 | FPR | Precision | F1 | Mean latency |
|---|---|---:|---:|---:|---:|---:|---:|
| Expert S1 | `expert_S1_cached_full_eval58.json` | 70.69% | 57.89% | 23.08% | 55.00% | 56.41% | 8.11s |
| Expert S2 | `expert_S2_cached_full_skip_dnsfail_eval58_resume.json` | 82.76% | 84.21% | 17.95% | 69.57% | 76.19% | 7.75s |
| Expert S3 | `expert_S3_cached_full_skip_dnsfail_no_whois_eval58_resume.json` | 74.14% | 63.16% | 20.51% | 60.00% | 61.54% | 5.59s |
| Expert S4 | `expert_S4_http_skip_dnsfail_eval58_resume.json` | 63.79% | 52.63% | 30.77% | 45.45% | 48.78% | 2.88s |

解释：58 条上 Expert S2 最优，但这是旧专家策略结果。现在专家策略已改成“只升不降”，后续重新跑时应使用新的结果文件名，避免覆盖旧结果。

## 5. full_dataset.json 223 条结果

### DeepSeek 专家模式 S1-S4 二分类结果

| 方法 | 结果文件 | Accuracy | URL 幻觉召回率 | FPR | Precision | F1 | Mean latency |
|---|---|---:|---:|---:|---:|---:|---:|
| DeepSeek Expert S1 | `deepseek_expert_S1_cached_full_full223.json` | 69.51% | 45.79% | 8.62% | 83.05% | 59.04% | 4.85s |
| DeepSeek Expert S2 | `deepseek_expert_S2_cached_full_skip_dnsfail_full223.json` | 78.92% | 63.55% | 6.90% | 89.47% | 74.32% | 4.00s |
| DeepSeek Expert S3 | `deepseek_expert_S3_cached_full_skip_dnsfail_no_whois_full223.json` | 70.40% | 46.73% | 7.76% | 84.75% | 60.24% | 3.34s |
| DeepSeek Expert S4 | `deepseek_expert_S4_http_skip_dnsfail_full223.json` | 65.92% | 42.99% | 12.93% | 75.41% | 54.76% | 2.20s |
| NeMo baseline + DeepSeek | `baseline_deepseek_nemo_hallucination_full223.json` | 73.09% | 100.00% | 51.72% | 64.07% | 78.10% | 7.65s |

解释：二分类上 NeMo baseline 的召回率达到 100%，但 FPR 为 51.72%，说明它几乎倾向于把大量正常链接也判为风险。DeepSeek Expert S2 的 precision 更高、FPR 更低。

### 四分类原始结果

| 方法 | 结果文件 | Strict Accuracy | Balanced Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---:|---:|---:|---:|---:|
| Domain + DeepSeek S2 | `deepseek_expert_S2_cached_full_skip_dnsfail_full223.json` | 54.26% | 42.15% | 47.73% | 42.15% | 30.75% |
| NeMo baseline + DeepSeek | `baseline_deepseek_nemo_hallucination_full223.json` | 62.33% | 38.46% | 37.42% | 38.46% | 34.10% |

解释：NeMo baseline 的 Strict Accuracy 略高，但它几乎把风险样本全部压成 `block`，`warn/refine` 识别能力为 0。Domain S2 能识别一部分 `refine`，但旧专家策略导致 `block` 召回很低。

### 阈值搜索结果

| 方案 | 结果文件 | 阈值/策略 | Strict Accuracy | Balanced Accuracy | Macro F1 | block Recall |
|---|---|---|---:|---:|---:|---:|
| 原始 Domain S2 | `deepseek_expert_S2_cached_full_skip_dnsfail_full223.json` | 原始 decision policy | 54.26% | 42.15% | 30.75% | 4.82% |
| 阈值最优 | `threshold_sweep_best_deepseek_s2_full223.json` | score_source=recalibrated; expert_policy=none; warn=25; refine=45; block=75 | 71.30% | 59.55% | 60.18% | 59.04% |
| 保留专家 block | `threshold_sweep_best_deepseek_s2_full223_preserve_block.json` | score_source=recalibrated; expert_policy=preserve_block; warn=25; refine=75; block=80 | 69.96% | 55.08% | 55.36% | 57.83% |
| 当前专家覆盖策略 | `threshold_sweep_best_deepseek_s2_full223_current_expert.json` | current expert policy | 50.22% | 41.83% | 32.78% | 4.82% |
| NeMo baseline | `baseline_deepseek_nemo_hallucination_full223.json` | NeMo hallucination baseline | 62.33% | 38.46% | 34.10% | 100.00% |

结论：若只看四分类综合能力，阈值最优的 Domain S2 明显优于 NeMo baseline：Strict Accuracy 71.30% vs 62.33%，Macro F1 60.18% vs 34.10%。如果保留专家模型参与最终决策，应采用“只升不降/保留 block”的策略，而不是让专家模型覆盖硬证据。

## 6. 多模型完整结果

本节补充所有已完成模型的二分类和四分类结果。统一使用 `calculate_strict_multiclass_metrics.py` 从每个结果文件的 `details` 中离线重算，因此与 runner 内部即时 metrics 可能有轻微口径差异；本节用于模型间横向比较。

新增汇总文件：

| 文件 | 内容 |
|---|---|
| `strict_multiclass_all_models_full223.json` | full_dataset.json 223 条，多模型 S1-S4 + NeMo baseline 的二分类和四分类统一指标 |
| `strict_multiclass_all_models_eval58.json` | eval_dataset.json 58 条，多模型 S1-S4 + NeMo baseline 的二分类和四分类统一指标 |

注意：`glm_air_expert_S3_cached_full_skip_dnsfail_no_whois_full223.json` 和 `glm_air_expert_S4_http_skip_dnsfail_full223.json` 这两个 full223 主文件已修复并可直接使用，下面表格和阈值结果统一引用主文件名。

### full_dataset.json 223 条：多模型总表

| 方法 | 结果文件 | Binary F1 | Precision | Recall | FPR | Strict Acc | Balanced Acc | Macro F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DeepSeek S1 | `deepseek_expert_S1_cached_full_full223.json` | 60.67% | 91.53% | 45.38% | 4.81% | 46.19% | 26.48% | 20.88% |
| DeepSeek S2 | `deepseek_expert_S2_cached_full_skip_dnsfail_full223.json` | 74.87% | 96.05% | 61.34% | 2.88% | 54.26% | 42.15% | 30.75% |
| DeepSeek S3 | `deepseek_expert_S3_cached_full_skip_dnsfail_no_whois_full223.json` | 61.80% | 93.22% | 46.22% | 3.85% | 47.53% | 26.59% | 22.12% |
| DeepSeek S4 | `deepseek_expert_S4_http_skip_dnsfail_full223.json` | 56.22% | 78.79% | 43.70% | 13.46% | 41.26% | 22.98% | 18.44% |
| Qwen S1 | `qwen_expert_S1_cached_full_full223.json` | 63.74% | 92.06% | 48.74% | 4.81% | 65.47% | 39.44% | 38.43% |
| Qwen S2 | `qwen_expert_S2_cached_full_skip_dnsfail_full223.json` | 60.67% | 91.53% | 45.38% | 4.81% | 65.47% | 38.70% | 37.36% |
| Qwen S3 | `qwen_expert_S3_cached_full_skip_dnsfail_no_whois_full223.json` | 60.67% | 91.53% | 45.38% | 4.81% | 65.47% | 38.70% | 37.45% |
| Qwen S4 | `qwen_expert_S4_http_skip_dnsfail_full223.json` | 56.68% | 77.94% | 44.54% | 14.42% | 60.09% | 35.69% | 35.08% |
| GLM-Air S1 | `glm_air_expert_S1_cached_full_full223.json` | 61.02% | 93.10% | 45.38% | 3.85% | 47.09% | 26.28% | 21.53% |
| GLM-Air S2 | `glm_air_expert_S2_cached_full_skip_dnsfail_full223.json` | 56.98% | 92.45% | 41.18% | 3.85% | 47.09% | 26.28% | 21.22% |
| GLM-Air S3 | `glm_air_expert_S3_cached_full_skip_dnsfail_no_whois_full223.json` | 58.62% | 92.73% | 42.86% | 3.85% | 49.78% | 28.09% | 24.44% |
| GLM-Air S4 | `glm_air_expert_S4_http_skip_dnsfail_full223.json` | 56.68% | 77.94% | 44.54% | 14.42% | 44.84% | 26.19% | 23.14% |
| OpenRouter GPT-4.1-mini S1 | `openrouter_gpt41mini_expert_S1_cached_full_full223.json` | 56.98% | 92.45% | 41.18% | 3.85% | 54.26% | 32.59% | 29.55% |
| OpenRouter GPT-4.1-mini S2 | `openrouter_gpt41mini_expert_S2_cached_full_skip_dnsfail_full223.json` | 54.97% | 90.38% | 39.50% | 4.81% | 50.22% | 28.45% | 24.85% |
| OpenRouter GPT-4.1-mini S3 | `openrouter_gpt41mini_expert_S3_cached_full_skip_dnsfail_no_whois_full223.json` | 55.29% | 92.16% | 39.50% | 3.85% | 51.57% | 29.30% | 25.92% |
| OpenRouter GPT-4.1-mini S4 | `openrouter_gpt41mini_expert_S4_http_skip_dnsfail_full223.json` | 68.63% | 82.35% | 58.82% | 14.42% | 53.36% | 44.50% | 35.21% |
| NeMo baseline + DeepSeek | `baseline_deepseek_nemo_hallucination_full223.json` | 83.22% | 71.26% | 100.00% | 46.15% | 62.33% | 38.46% | 34.10% |

full223 原始多模型结果的最佳项：

| 指标 | 最佳方法 | 值 | 说明 |
|---|---|---:|---|
| Binary F1 | NeMo baseline + DeepSeek | 83.22% | 召回率 100%，但 FPR 高达 46.15% |
| Binary Precision | DeepSeek S2 | 96.05% | 最适合强调低误报、高置信风险识别 |
| Strict Accuracy | Qwen S1 / Qwen S2 / Qwen S3 | 65.47% | Qwen 在原始四分类映射上最好 |
| Macro F1 | Qwen S1 | 38.43% | 原始多模型四分类 Macro F1 最好 |
| Balanced Accuracy | OpenRouter GPT-4.1-mini S4 | 44.50% | 但 FPR 为 14.42%，且 Binary F1 不如 DeepSeek S2 |

加入阈值搜索后的当前总最佳：

| 方案 | 文件 | Strict Acc | Balanced Acc | Macro F1 | block Recall |
|---|---|---:|---:|---:|---:|
| DeepSeek S2 阈值校准 | `threshold_sweep_best_deepseek_s2_full223.json` | 71.30% | 59.55% | 60.18% | 59.04% |
| DeepSeek S2 保留专家 block | `threshold_sweep_best_deepseek_s2_full223_preserve_block.json` | 69.96% | 55.08% | 55.36% | 57.83% |
| 原始多模型最佳 Qwen S1 | `qwen_expert_S1_cached_full_full223.json` | 65.47% | 39.44% | 38.43% | 见 `strict_multiclass_all_models_full223.json` |
| NeMo baseline + DeepSeek | `baseline_deepseek_nemo_hallucination_full223.json` | 62.33% | 38.46% | 34.10% | 100.00% |

结论：如果允许做阈值校准，当前最佳是 `DeepSeek S2 + recalibrated score thresholds(warn=25, refine=45, block=75)`；如果只比较未校准原始输出，四分类最佳是 `Qwen S1`，二分类低误报最佳是 `DeepSeek S2`。

### DeepSeek 最优阈值跨模型迁移实验

这个实验回答一个更公平的问题：不让每个模型单独搜索自己的最优阈值，而是固定使用 DeepSeek S2 在 full223 上得到的最优阈值，然后套到其他模型的保存结果上，观察这套规则是否有跨模型泛化效果。

固定阈值配置：

```text
### per-model own-threshold summary

每个模型单独搜索自己的最优阈值后，结果如下：

| Model | Best strategy | Strict Acc | Balanced Acc | Macro F1 |
|---|---|---:|---:|---:|
| DeepSeek | S2 | 71.30% | 59.55% | 60.18% |
| Qwen | S1 | 63.23% | 43.82% | 45.17% |
| GLM | S1 | 65.02% | 44.29% | 45.11% |
| GPT-4.1-mini | S4 | 70.85% | 60.61% | 63.35% |

对应文件：

- `per_model_threshold_sweep_summary.json`
- `per_model_threshold_sweep_results.json`

### glm_air rerun files

这两份是 `glm_air` 的 full223 rerun 版本，后续引用结果时优先用它们：

- `glm_air_expert_S3_cached_full_skip_dnsfail_no_whois_full223_rerun.json`
- `glm_air_expert_S4_http_skip_dnsfail_full223_rerun.json`

score_source = recalibrated
expert_policy = none
warn = 25
refine = 45
block = 75
calibrated_from = deepseek_expert_S2_cached_full_skip_dnsfail_full223.json
```

生成脚本：

```text
apply_deepseek_thresholds_to_models.py
```

汇总文件：

```text
deepseek_thresholds_all_models_full223_summary.json
```

每个模型对应的 remap JSON 文件命名格式：

```text
*_deepseek_thresholds.json
```

| Method | Output file | Binary F1 | Precision | Recall | FPR | Strict Acc | Balanced Acc | Macro F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DeepSeek S2 | `deepseek_expert_S2_cached_full_skip_dnsfail_full223_deepseek_thresholds.json` | 72.46% | 85.23% | 63.03% | 12.50% | 71.30% | 59.55% | 60.18% |
| OpenRouter GPT-4.1-mini S4 | `openrouter_gpt41mini_expert_S4_http_skip_dnsfail_full223_deepseek_thresholds.json` | 68.97% | 83.33% | 58.82% | 13.46% | 69.06% | 57.80% | 58.62% |
| OpenRouter GPT-4.1-mini S1 | `openrouter_gpt41mini_expert_S1_cached_full_full223_deepseek_thresholds.json` | 62.50% | 82.19% | 50.42% | 12.50% | 64.13% | 45.85% | 47.73% |
| Qwen S1 | `qwen_expert_S1_cached_full_full223_deepseek_thresholds.json` | 62.18% | 81.08% | 50.42% | 13.46% | 63.23% | 43.82% | 45.17% |
| GLM-Air S1 | `glm_air_expert_S1_cached_full_full223_deepseek_thresholds.json` | 64.29% | 81.82% | 52.94% | 13.46% | 65.02% | 44.29% | 45.11% |
| DeepSeek S3 | `deepseek_expert_S3_cached_full_skip_dnsfail_no_whois_full223_deepseek_thresholds.json` | 61.05% | 81.69% | 48.74% | 12.50% | 63.68% | 43.32% | 44.20% |
| OpenRouter GPT-4.1-mini S2 | `openrouter_gpt41mini_expert_S2_cached_full_skip_dnsfail_full223_deepseek_thresholds.json` | 59.26% | 80.00% | 47.06% | 13.46% | 62.33% | 42.48% | 43.50% |
| DeepSeek S1 | `deepseek_expert_S1_cached_full_full223_deepseek_thresholds.json` | 60.00% | 80.28% | 47.90% | 13.46% | 62.33% | 42.48% | 43.26% |
| Qwen S3 | `qwen_expert_S3_cached_full_skip_dnsfail_no_whois_full223_deepseek_thresholds.json` | 60.00% | 80.28% | 47.90% | 13.46% | 62.33% | 42.48% | 43.26% |
| OpenRouter GPT-4.1-mini S3 | `openrouter_gpt41mini_expert_S3_cached_full_skip_dnsfail_no_whois_full223_deepseek_thresholds.json` | 60.00% | 80.28% | 47.90% | 13.46% | 62.33% | 42.48% | 43.26% |
| Qwen S2 | `qwen_expert_S2_cached_full_skip_dnsfail_full223_deepseek_thresholds.json` | 60.00% | 80.28% | 47.90% | 13.46% | 62.33% | 42.48% | 43.17% |
| GLM-Air S2 | `glm_air_expert_S2_cached_full_skip_dnsfail_full223_deepseek_thresholds.json` | 60.00% | 80.28% | 47.90% | 13.46% | 62.33% | 42.48% | 43.17% |
| GLM-Air S3 | `glm_air_expert_S3_cached_full_skip_dnsfail_no_whois_full223_deepseek_thresholds.json` | 60.32% | 81.43% | 47.90% | 12.50% | 62.33% | 40.64% | 41.40% |
| DeepSeek S4 | `deepseek_expert_S4_http_skip_dnsfail_full223_deepseek_thresholds.json` | 56.52% | 80.00% | 43.70% | 12.50% | 61.88% | 40.34% | 41.14% |
| Qwen S4 | `qwen_expert_S4_http_skip_dnsfail_full223_deepseek_thresholds.json` | 56.99% | 79.10% | 44.54% | 13.46% | 61.43% | 40.10% | 40.83% |
| GLM-Air S4 | `glm_air_expert_S4_http_skip_dnsfail_full223_deepseek_thresholds.json` | 56.99% | 79.10% | 44.54% | 13.46% | 61.43% | 40.10% | 40.83% |

这个结果有两个价值：

1. 固定 DeepSeek 阈值后，多数模型的四分类 Macro F1 明显高于未校准版本，说明问题主要不只是专家模型能力，而是最终四分类映射策略。
2. 跨模型迁移后最强仍是 DeepSeek S2，Macro F1 为 60.18%；第二名是 OpenRouter GPT-4.1-mini S4，Macro F1 为 58.62%。这说明 DeepSeek 阈值不是只对 DeepSeek 有效，但 DeepSeek S2 仍是当前总体最佳。

### eval_dataset.json 58 条：多模型总表

| 方法 | 结果文件 | Binary F1 | Precision | Recall | FPR | Strict Acc | Balanced Acc | Macro F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DeepSeek S1 | `expert_S1_cached_full_eval58.json` | 62.75% | 80.00% | 51.61% | 14.81% | 44.83% | 39.53% | 25.23% |
| DeepSeek S2 | `expert_S2_cached_full_skip_dnsfail_eval58_resume.json` | 77.78% | 91.30% | 67.74% | 7.41% | 48.28% | 41.38% | 27.10% |
| DeepSeek S3 | `expert_S3_cached_full_skip_dnsfail_no_whois_eval58_resume.json` | 66.67% | 85.00% | 54.84% | 11.11% | 44.83% | 32.12% | 23.62% |
| DeepSeek S4 | `expert_S4_http_skip_dnsfail_eval58_resume.json` | 55.17% | 59.26% | 51.61% | 40.74% | 29.31% | 23.15% | 16.06% |
| Qwen S1 | `qwen_expert_S1_cached_full_eval58_resume.json` | 75.00% | 84.00% | 67.74% | 14.81% | 60.34% | 53.59% | 42.98% |
| Qwen S2 | `qwen_expert_S2_cached_full_skip_dnsfail_eval58_resume.json` | 62.75% | 80.00% | 51.61% | 14.81% | 56.90% | 43.69% | 38.31% |
| Qwen S3 | `qwen_expert_S3_cached_full_skip_dnsfail_no_whois_eval58_resume.json` | 62.75% | 80.00% | 51.61% | 14.81% | 56.90% | 43.69% | 38.53% |
| Qwen S4 | `qwen_expert_S4_http_skip_dnsfail_eval58_resume.json` | 56.67% | 58.62% | 54.84% | 44.44% | 43.10% | 36.28% | 32.48% |
| GLM S1 | `glm_expert_S1_cached_full_eval58.json` | 58.82% | 75.00% | 48.39% | 18.52% | 39.66% | 28.70% | 19.10% |
| GLM S2 | `glm_expert_S2_cached_full_skip_dnsfail_eval58.json` | 58.82% | 75.00% | 48.39% | 18.52% | 41.38% | 37.04% | 21.27% |
| GLM S3 | `glm_expert_S3_cached_full_skip_dnsfail_no_whois_eval58.json` | 58.82% | 75.00% | 48.39% | 18.52% | 41.38% | 37.04% | 21.27% |
| GLM S4 | `glm_expert_S4_http_skip_dnsfail_eval58.json` | 65.62% | 63.64% | 67.74% | 44.44% | 31.03% | 38.89% | 19.11% |
| GLM-Air S1 | `glm_air_expert_S1_cached_full_eval58.json` | 47.83% | 73.33% | 35.48% | 14.81% | 46.55% | 34.32% | 27.66% |
| GLM-Air S2 | `glm_air_expert_S2_cached_full_skip_dnsfail_eval58.json` | 55.32% | 81.25% | 41.94% | 11.11% | 46.55% | 33.68% | 25.89% |
| GLM-Air S3 | `glm_air_expert_S3_cached_full_skip_dnsfail_no_whois_eval58.json` | 51.06% | 75.00% | 38.71% | 14.81% | 44.83% | 32.75% | 25.16% |
| GLM-Air S4 | `glm_air_expert_S4_http_skip_dnsfail_eval58.json` | 56.67% | 58.62% | 54.84% | 44.44% | 32.76% | 33.68% | 22.70% |
| OpenRouter GPT-4.1-mini S1 | `openrouter_gpt41mini_expert_S1_cached_full_eval58.json` | 47.83% | 73.33% | 35.48% | 14.81% | 48.28% | 35.88% | 29.76% |
| OpenRouter GPT-4.1-mini S2 | `openrouter_gpt41mini_expert_S2_cached_full_skip_dnsfail_eval58.json` | 47.83% | 73.33% | 35.48% | 14.81% | 44.83% | 32.75% | 25.14% |
| OpenRouter GPT-4.1-mini S3 | `openrouter_gpt41mini_expert_S3_cached_full_skip_dnsfail_no_whois_eval58.json` | 47.83% | 73.33% | 35.48% | 14.81% | 44.83% | 32.75% | 25.14% |
| OpenRouter GPT-4.1-mini S4 | `openrouter_gpt41mini_expert_S4_http_skip_dnsfail_eval58.json` | 56.67% | 58.62% | 54.84% | 44.44% | 31.03% | 25.35% | 20.22% |
| NeMo baseline + DeepSeek | `final_A_S3_full_skip_dnsfail_no_whois_vs_nemo_eval58.json` | 82.19% | 71.43% | 96.77% | 44.44% | 51.72% | 37.33% | 30.37% |

eval58 原始多模型结果的最佳项：

| 指标 | 最佳方法 | 值 | 说明 |
|---|---|---:|---|
| Binary F1 | NeMo baseline + DeepSeek | 82.19% | 召回率高，但 FPR 为 44.44% |
| Binary Precision | DeepSeek S2 | 91.30% | 小样本上最低误报倾向之一 |
| Strict Accuracy | Qwen S1 | 60.34% | 小样本四分类最佳 |
| Balanced Accuracy | Qwen S1 | 53.59% | 四类平均召回最好 |
| Macro F1 | Qwen S1 | 42.98% | 小样本四分类综合最佳 |

## 7. E2E pipeline 结果

| 文件 | LLM | 问题池 | 完成条数 | Guard modes | Ground truth |
|---|---|---|---:|---|---|
| `e2e_all_strategies_COMPLETE_20260603_174423.json` | DeepSeek `deepseek-chat` | `question_pool_v2.json` | 241 results，其中 225 条有 ground truth label | domain-s1, domain-s2, domain-s3, domain-s4 | DNS/HTTP/GitHub verifier |

Ground truth 分布（225 条可计入指标）：

| Label | Count |
|---|---:|
| clean | 99 |
| mixed | 112 |
| hallucinated | 5 |
| no_links | 9 |

E2E 二分类指标：

| Guard mode | Accuracy | Recall | FPR | Precision | F1 | Mean latency | 说明 |
|---|---:|---:|---:|---:|---:|---:|---|
| domain-s1 | 53.78% | 35.90% | 26.85% | 59.15% | 44.68% | 27.96s | 首个策略承担大量外部查询，延迟最高 |
| domain-s2 | 53.78% | 35.90% | 26.85% | 59.15% | 44.68% | 0.001s | 受缓存影响，延迟不可与 S1 直接比较 |
| domain-s3 | 53.78% | 35.90% | 26.85% | 59.15% | 44.68% | 0.001s | 受缓存影响，延迟不可与 S1 直接比较 |
| domain-s4 | 52.89% | 70.94% | 66.67% | 53.55% | 61.03% | 0.001s | 召回更高，但误报率很高 |

注意：E2E 结果是端到端真实 LLM 回答实验，和静态标注集不同。当前 S2/S3/S4 延迟受同一进程缓存影响，不能作为独立冷启动延迟引用；若写论文，需要单独冷启动或清缓存重跑。

## 8. 主要脚本和配置

| 文件 | 用途 |
|---|---|
| `eval_runner.py` | 早期评测 runner；dry-run 曾存在标签泄露问题，后续不应把 dry-run 100% 当真实性能 |
| `run_ablation_experiments.py` | 静态数据集 A/B/C 对照实验主脚本 |
| `run_A_eval58_optimization_sweep.ps1` | 58 条非专家 S1-S4 sweep |
| `run_A_eval58_expert_sweep.ps1` / `run_A_eval58_expert_sweep_resume.ps1` | 58 条专家 S1-S4 sweep |
| `run_A_full223_model_expert_sweep.ps1` | 223 条多模型专家 sweep |
| `calculate_strict_multiclass_metrics.py` | 四分类指标计算 |
| `threshold_sweep_strict_multiclass.py` | 离线阈值搜索 |
| `strict_multiclass_all_models_full223.json` | 223 条全模型统一指标汇总 |
| `strict_multiclass_all_models_eval58.json` | 58 条全模型统一指标汇总 |
| `e2e_eval_pipeline.py` | 端到端 pipeline：问题池 -> LLM 回答 -> guard -> independent verifier -> metrics |
| `deepseek_config.yml` | DeepSeek / OpenAI-compatible LLM 配置 |
| `glm_config.yml`, `qwen_config.yml`, `openrouter_gpt41mini_config.yml` | 其他专家模型配置 |

## 9. 当前最重要结论

1. 静态 58 条上，非专家 S3 相比 NeMo baseline 误报率低很多，且速度更快。
2. 静态 223 条上，DeepSeek Expert S2 的二分类 precision 和 FPR 优于 NeMo baseline，但原始四分类 block 召回偏低。
3. 未校准的多模型四分类中，Qwen S1 当前最好：full223 Macro F1 为 38.43%，eval58 Macro F1 为 42.98%。
4. 阈值搜索后，DeepSeek S2 四分类 Macro F1 提升到 60.18%，显著高于 NeMo baseline 的 34.10%，也是当前总最佳。
5. 专家模型不应直接覆盖最终决策；更合理的策略是 advisory + upgrade-only，即可以提升风险，但不能推翻 DNS/GitHub/HTTP/TLS/WHOIS 等硬证据。
6. E2E pipeline 已经可运行，但当前结果更适合作为“真实 LLM 诱导幻觉 + 独立 verifier”实验原型；最终论文指标建议冷启动、分模式单独运行、避免缓存污染延迟。

## 10. 结果备份

关键结果已有备份文件，常见格式为：

```text
*.backup_YYYYMMDD_HHMMSS
```

阈值搜索最优结果备份时间戳：

```text
backup_20260603_171603
```

后续重新跑专家策略时，建议使用新文件名，例如：

```text
deepseek_expert_S2_upgrade_only_full223.json
threshold_sweep_best_deepseek_s2_upgrade_only_full223.json
```

避免覆盖旧的 `current expert policy` 结果。
