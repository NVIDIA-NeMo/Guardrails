# 发现与研究记录

## Layer 1 设计参考（来自 zip 文件）

### 代码位置
```
/tmp/layer1_plan/
  ├── actions.py (8033 bytes) - 核心 LLM 判断逻辑
  ├── prompts.yml (2527 bytes) - LLM 提示定义
  ├── flows.co (1226 bytes) - 流程定义
  ├── __init__.py (684 bytes) - 模块初始化
  ├── README.md (2060 bytes) - 使用说明
  └── flows_v1.co (1007 bytes) - 旧流程版本
```

### Layer 1 核心设计

#### 实体提取（零依赖）
```python
# 正则提取 URLs
_URL_RE = regex for https?://... or www.

# 解析 domains
domain = urlparse(url).hostname

# GitHub 仓库
owner/repo from github.com/owner/repo
```

#### LLM 判断（5个维度）
1. **Domain Existence** - 域名是否真实存在
2. **GitHub Verification** - 仓库是否真实存在
3. **URL Path Plausibility** - 路径是否合理
4. **Typosquatting** - 是否仿冒
5. **Suspicious Patterns** - 可疑模式

#### 提示关键字
```
答"yes" if 任何 URL/domain/repo 是虚构的
答"no" if 所有引用看起来真实且合法
```

---

## 现有代码分析（v1 - 完整网络验证）

### 现有模块结构
```
nemoguardrails/library/domain_hallucination/
├── actions.py (22867 bytes) - 主入口 + analyze_answer()
├── verification.py (29359 bytes) - DNS/HTTP/TLS/WHOIS/GitHub 验证
├── checkers.py (14784 bytes) - 聚合问题检查
├── scoring.py (7374 bytes) - 风险打分（L0-L4）
├── decision.py (5206 bytes) - 决策逻辑
├── extractors.py (10660 bytes) - 实体提取
├── kb.py (9727 bytes) - 知识库
├── schemas.py (4006 bytes) - 数据结构
├── config.py (9287 bytes) - 配置管理
├── expert_review.py (13807 bytes) - 专家审查
├── semantic.py (6065 bytes) - 语义检查
├── utils.py (9657 bytes) - 工具函数
└── seed_kb.json (3444 bytes) - 种子知识库
```

### 现有 v1 的问题
- ❌ **性能**：每个 URL 都做网络调用（DNS/HTTP/TLS/WHOIS）→ 500-2000ms
- ❌ **复杂性**：233 个测试，大量依赖（socket/ssl/httpx 等）
- ❌ **接受度**：被标记为 "off-topic"，说明功能价值不被重视

---

## 架构决策

### 为什么两层？

| 对比 | 仅 Layer 1 | 仅 Layer 2（现有） | Layer 1+2（提议） |
|------|-----------|-----------------|-----------------|
| **性能** | 100ms | 2000ms | 100ms（默认）+ 2000ms（可选） |
| **依赖** | 零 | 20+ | 1 个（配置选项） |
| **测试** | 10 个 | 233 个 | 243 个 |
| **易接受度** | ✅ 高 | ❌ 低 | ✅ 高（默认轻量） |
| **深度防护** | ❌ 无 | ✅ 有 | ✅ 可选有 |

### 关键决策
1. **默认 Layer 1 only** - 易接受，符合 PR 期望
2. **Layer 2 可选** - `enable_advanced_verification=false`（默认）
3. **零破坏性** - 所有现有代码保留，仅添加新层

---

## PR 价值主张（新方向）

### 旧 PR 的问题（被标记 off-topic）
- 太完整，太复杂
- 233 个测试可能过度工程化
- 网络验证不是"核心"需求

### 新 PR 的优势
- **Layer 1**：轻量级 LLM 判断 → 容易接受的基础特性
- **Layer 2**：可选深度验证 → 高级用户的增强
- **分层思想**：符合 NIST/企业安全框架的"纵深防御"

---

## 实现细节备忘

### Layer 1 提示中的关键措辞
```
"Unlike general hallucination or fact-checking,
you are ONLY checking whether external references are real."
```
这使 LLM 更专注，减少误判。

### Layer 2 可以重用的现有代码
- `verification.py` - DNS/HTTP/TLS/WHOIS（全部保留）
- `checkers.py` - 问题聚合（全部保留）
- `scoring.py` - 打分逻辑（全部保留）
- `decision.py` - 决策规则（全部保留）

→ 不需要重写，仅需在 actions.py 中添加条件逻辑

---

## 测试策略备忘

### Layer 1 测试（新增，~20 个）
- 实体提取正确性
- LLM 判断（mock）
- 边界情况（空输入、无链接等）

### Layer 2 测试（保留，~213 个）
- 所有现有 233 个测试，通过 `enable_advanced_verification=true` 激活
- 网络验证（DNS/HTTP/TLS/WHOIS）
- 打分与决策

### 集成测试（新增，~10 个）
- Layer 1 -> pass -> 响应放行
- Layer 1 -> suspicious -> Layer 2 验证 -> 最终决策
- Layer 2 disabled -> 直接使用 Layer 1 结果

---

## 可能的问题与解决方案

| 问题 | 可能原因 | 解决方案 |
|------|--------|--------|
| Layer 1 LLM 判断不准 | 提示不够清晰 | 调整提示词（5个维度很清晰） |
| 现有测试失败 | Layer 2 逻辑改变 | 默认启用 Layer 2，保持向后兼容 |
| 性能没改进 | Layer 1 调用过多 | 可以缓存 LLM 调用结果 |
| PR 仍被拒 | 功能定位问题 | 强调两层架构，突出轻量级价值 |

---

## 下一步需要的信息

1. ✅ Layer 1 代码已获得（zip 文件）
2. ⏳ Layer 2 重构策略 → 从现有代码提取
3. ⏳ 配置项实现 → `enable_advanced_verification`
4. ⏳ PR 描述重写 → 突出两层架构
