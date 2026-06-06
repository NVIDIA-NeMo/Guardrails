# 两层域名幻觉检测架构重构计划

## 📋 总体目标
将现有网络验证模块改造为两层架构：
- **Layer 1（快速）**：LLM 快速判断（默认）← PR 更可能被接受
- **Layer 2（深度）**：可选的网络验证和精细打分

**为什么？**
1. 现有 PR 被标记为 "off-topic"，说明功能认可度有问题
2. Layer 1 轻量级，更易被接受为基础特性
3. Layer 2 可选，符合开源项目的渐进式设计

---

## 🔄 实现阶段

### Phase 1: 代码组织与备份 [Status: complete]
- [x] 备份原始代码到 `/tmp/backup_original/domain_hallucination_v1`
- [x] 整理 zip 文件中的 Layer 1 参考代码
- [x] 理解两层架构设计

### Phase 2: 提取与创建 Layer 1 [Status: in_progress]
**目标**: 实现快速 LLM 判断，零网络依赖
- [ ] 创建 `layer1_check.py` - LLM 快速判断模块
  - 实体提取（URL、domain、GitHub repos）
  - LLM 评估提示（5个维度）
  - 返回 yes/no 判断
- [ ] 更新 `prompts.yml` - 添加 Layer 1 提示
- [ ] 创建 Layer 1 专用测试
- [ ] 验证：零网络调用

**关键文件**:
- `nemoguardrails/library/domain_hallucination/layer1_check.py` (新建)
- `prompts.yml` (更新)

### Phase 3: 重构现有代码为 Layer 2 [Status: pending]
**目标**: 将网络验证模块整理为可选的"深度层"
- [ ] 创建 `layer2_advanced.py` - 重构现有验证代码
  - DNS/HTTP/TLS/WHOIS 验证
  - 精细打分（L0-L4）
  - 决策逻辑（block/refine/warn/pass）
- [ ] 删除重复代码（提取到 layer2_advanced.py）
- [ ] 验证：保留所有安全防护

**关键文件**:
- `nemoguardrails/library/domain_hallucination/layer2_advanced.py` (新建)
- 保留原有的 `verification.py`, `checkers.py` 等

### Phase 4: 整合两层逻辑 [Status: pending]
**目标**: 在 actions.py 中整合 Layer 1 + Layer 2
- [ ] 更新 `actions.py` 核心逻辑
  - 默认执行 Layer 1
  - 如果启用 Layer 2，则补充深度验证
  - 整合决策和修改
- [ ] 更新 `config.py` - 新增 `enable_advanced_verification` 配置
- [ ] 更新 `flows.co` - 支持两层流程
- [ ] 更新 `__init__.py` - 导出新模块

**关键文件**:
- `nemoguardrails/library/domain_hallucination/actions.py` (修改)
- `nemoguardrails/library/domain_hallucination/config.py` (添加配置)
- `nemoguardrails/library/domain_hallucination/flows.co` (更新流程)

### Phase 5: 测试验证 [Status: pending]
**目标**: 确保所有现有测试通过 + 新增 Layer 1 测试
- [ ] 运行现有 233 个测试
- [ ] 创建 Layer 1 专用测试（快速 LLM 判断）
- [ ] 创建集成测试（Layer 1 + Layer 2 组合）
- [ ] 验证：95%+ 覆盖率

**测试文件**:
- `tests/library/domain_hallucination/test_layer1_check.py` (新建)
- 现有测试文件保持兼容

### Phase 6: 文档与向后兼容性 [Status: pending]
**目标**: 更新文档，确保向后兼容
- [ ] 更新 `README.md` - 解释两层架构
- [ ] 添加配置示例（Layer 1 only vs Layer 1+2）
- [ ] 确保默认行为不变（Layer 1 only）
- [ ] 验证：旧代码仍可用

### Phase 7: PR 准备 [Status: pending]
**目标**: 为新 PR 做准备
- [ ] 提交所有改动
- [ ] 验证 CI 检查通过
- [ ] 撰写新 PR 描述（强调 Layer 1 的轻量级价值）
- [ ] 关联旧 PR（证明改进）

---

## 🎯 关键约束与设计决策

| 约束 | 状态 | 原因 |
|------|------|------|
| 所有现有 233 测试必须通过 | 🔴 待验证 | 向后兼容性 |
| Layer 1 零网络依赖 | ✅ 设计阶段确认 | 性能优先 |
| Layer 2 保留所有安全防护 | ✅ 设计阶段确认 | 深度层不妥协安全 |
| 默认仅启用 Layer 1 | ✅ 配置默认值 | 轻量级优先，易接受 |

---

## 📊 进度跟踪

| 阶段 | 完成度 | 下一步 |
|------|--------|-------|
| Phase 1: 备份 | 100% | 开始 Phase 2 |
| Phase 2: Layer 1 | 0% | 从 zip 提取代码 |
| Phase 3: Layer 2 重构 | 0% | 等待 Phase 2 完成 |
| Phase 4: 整合 | 0% | 等待 Phase 2+3 完成 |
| Phase 5: 测试 | 0% | 等待 Phase 4 完成 |
| Phase 6: 文档 | 0% | 等待所有阶段完成 |
| Phase 7: PR 准备 | 0% | 等待所有测试通过 |

---

## 🚀 立即行动

**现在要做**：启动 Phase 2
1. 从 `/tmp/layer1_plan/` 复制 Layer 1 代码
2. 创建 `layer1_check.py`
3. 运行初步测试

**所需决定**：无，按计划执行即可
