# Domain Hallucination Guard — 评测工具包

## 文件说明

| 文件 | 用途 |
|------|------|
| `eval_dataset.json` | **核心评测数据集**（58条手工标注，7大类别） |
| `expanded_dataset.json` | **扩展数据集**（151条，含自动生成样本） |
| `eval_runner.py` | **评测执行器** — 跑你的 adapter 并计算指标 |
| `generate_dataset.py` | **数据集生成器** — 从模板/seed_kb 扩展数据 |
| `generate_report.py` | **报告生成器** — 从结果 JSON 生成 Markdown 报告 |

## 快速开始

### 1. Dry Run（验证框架可用）

```bash
python eval_runner.py --dry-run
```

### 2. 接入你的 adapter 跑真实评测

```bash
# 确保你的 domain_hallucination 模块可导入
export PYTHONPATH=/path/to/your/Guardrails:$PYTHONPATH

python eval_runner.py \
  --dataset eval_dataset.json \
  --config /path/to/config.json \
  --output results_v1.json
```

### 3. 生成 Baseline 对比数据

```bash
# 先用原版 NeMo Guardrails hallucination rail 跑一遍作为 baseline
python eval_runner.py --config baseline_config.json --output baseline.json

# 再用你的改进版跑
python eval_runner.py --config your_config.json --output improved.json

# 对比
python eval_runner.py \
  --config your_config.json \
  --baseline-results baseline.json \
  --output comparison.json
```

### 4. 从 seed_kb.json 生成更多测试用例

```bash
python generate_dataset.py \
  --expand eval_dataset.json \
  --seed-kb /path/to/seed_kb.json \
  --count 300 \
  --output full_dataset.json
```

### 5. 生成 Markdown 报告

```bash
python generate_report.py \
  --results improved.json \
  --baseline baseline.json \
  --output report.md
```

## 数据集分类说明

| 类别 | 样本数 | 期望行为 | 测试目标 |
|------|--------|----------|----------|
| `real_links` | 10+ | pass | 不误杀真实链接 |
| `hallucinated_links` | 15+ | block/refine | 检出虚构 URL/域名/仓库 |
| `mixed_links` | 5+ | refine/warn | 处理真假混合输出 |
| `no_links` | 5+ | fast pass | 无链接时快速放行 |
| `typosquatting` | 5+ | block/warn | 检测域名仿冒 |
| `blacklisted` | 3+ | block | 拦截已知恶意域名 |
| `edge_cases` | 15+ | varies | IP地址、编码URL、重定向等 |

## 输出指标说明

- **Accuracy**: 全局准确率
- **Binary F1**: 将 block/refine 归为"已标记"，warn/pass 归为"未标记"的二分类 F1
- **Per-Decision**: 对 block/refine/warn/pass 四个决策的分别指标
- **Latency**: 每次检测的耗时（P50/P95/P99）
- **Confusion Matrix**: 真实标签 × 预测标签的矩阵

## 扩展数据集

### 手动添加测试用例

在 `eval_dataset.json` 的 `test_cases` 数组中追加：

```json
{
  "id": "custom_001",
  "category": "hallucinated_links",
  "subcategory": "your_custom_type",
  "user_query": "用户问题",
  "llm_answer": "LLM 的回答（包含待检测的链接）",
  "entities": {
    "urls": ["https://..."],
    "domains": ["..."],
    "github_repos": [{"owner": "...", "repo": "..."}]
  },
  "expected_decision": "block",
  "expected_risk_level": "L3",
  "notes": "说明"
}
```

### 对接真实对话日志

如果你有真实的 LLM 输出日志，可以这样转换：

```python
import json

logs = [...]  # 你的日志数据
cases = []
for i, log in enumerate(logs):
    cases.append({
        "id": f"prod_{i:04d}",
        "category": "production",  # 需要人工标注 expected_decision
        "user_query": log["query"],
        "llm_answer": log["response"],
        "entities": {},  # 由你的 extractor 自动填充
        "expected_decision": "TODO",  # 人工标注
    })
```
