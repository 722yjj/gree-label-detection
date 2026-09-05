# M11 后续 Pair Interaction Adapter 模型建议

> 讨论日期：2026-07-27  
> 当前基线：M11 DINOv2-L/14 + letterbox + patch/OCR head  
> 配套数据：[`d06_single_mutation_dataset.md`](../datasets/d06_single_mutation_dataset.md)

首轮 A0/A1 三 seed 实验已经完成，结果见
[`d06_pair_interaction_a0_a1_experiment.md`](../experiments/d06_pair_interaction_a0_a1_experiment.md)。

## 1. 结论

下一阶段不建议直接把模板和实拍 RGB 图像拼成一张图送入 DINOv2，也不建议完全删除分类头。更合理的方向是：

> 保留共享权重的双路预训练 backbone，在多层 token 特征之间加入可学习的成对交互和定位 adapter，使用 `changed_bbox` 监督变化位置，使用 `value_bbox` 提供业务上下文，最后保留一个极小的二分类决策层。

需要改进的重点是“模板和目标何时、如何交互”，不是简单增加网络层数。

## 2. M11 当前结构

M11 的实际数据流：

```text
模板 crop ─→ 共享冻结 DINOv2 ─→ Ft6/Ft12/Ft18/Ft24 ─┐
                                                       ├─ 对应位置 |Ft-Fx|
目标 crop ─→ 共享冻结 DINOv2 ─→ Fx6/Fx12/Fx18/Fx24 ─┘
                                                       ↓
                          candidate mask 内 mean/max/top-k + global mean
                                                       ↓
                              16384 维视觉特征 + 24 维 OCR
                                                       ↓
                                  128 维 MLP → keep/discard
```

对应实现：

- 两张图分别前向：`/home/jnu/projects/aa_clip_exp/label_pair/train_patch.py`；
- patch 绝对差与统计池化：`/home/jnu/projects/aa_clip_exp/label_pair/model.py`；
- DINOv2 和 letterbox：`/home/jnu/projects/aa_clip_exp/label_pair/visual_backbone.py`。

M11 的主要优点：

- 复用 DINOv2 预训练表示；
- 模板/目标共享参数，特征空间一致；
- letterbox 保留宽文字行两端；
- 推理快，历史开发集 Recall 较高。

主要限制：

- 模板和目标在 backbone 内完全没有交互；
- 只使用对应位置绝对差，残余错位容易被当成内容变化；
- 空间特征很早被 mean/max/top-k 压缩；
- 16384 维统计向量交给 MLP，模型难以显式学习“变化发生在哪个字符”；
- M11 使用的旧 D05v3 曾包含配准污染，只能作为经验基线，不能作为新结构的最终结论。

## 3. “把两张图片叠起来”的四种含义

| 方式 | 张量形式 | 是否发生早期交互 | 主要问题 | 建议 |
| --- | --- | --- | --- | --- |
| batch 合并 | `[2B,3,H,W]` | 否 | 只减少调用次数，数学结构不变 | 可用于加速 |
| 通道拼接 | `[B,6,H,W]` | 是 | DINO patch embedding 只接受 3 通道，破坏预训练接口 | 不作为主线 |
| 横向拼接 | `[B,3,H,2W]` | 是 | 缩回 336 后每张图分辨率减半；保持宽度则注意力成本上升 | 只做消融 |
| 像素差/透明叠加 | `[B,3,H,W]` | 是 | 对光照、印刷、模糊和轻微配准误差敏感 | 不作为主线 |

通道拼接需要修改 DINOv2 第一层 3 通道 patch embedding。虽然可以新增 `6→3` 卷积或重新初始化 patch projection，但在小数据集上容易优先学习模板域与拍摄域的外观差异，而不是字符内容差异。

横向拼接允许 ViT self-attention 跨图通信，但会改变预训练时的空间分布。对上标和单字符变化而言，输入分辨率损失尤其危险。

## 4. 为什么仍需保留决策头

系统最终必须输出 `same/different` 概率或分数，所以无法真正取消决策层：

- `Linear → logit` 是学习式分类头；
- anomaly score + threshold 是固定分类头；
- heatmap max/top-k + threshold 仍然是决策头。

可以删除 M11 的高维 `16384→128→1` 形式，改成更受约束的 `128→1`，但不能没有最后的标量决策。

推荐将分类头缩小为：

```text
LayerNorm(128 + OCR维度) → Linear(1)
```

如果纯视觉模型有效，再追加 OCR 做独立消融；第一轮不要把结构变化和 OCR 变化混在一起。

## 5. 推荐结构

建议暂称为 `Pair Interaction Adapter`：

```text
模板 value crop ─→ 共享冻结 DINO ─→ Ft6/Ft12/Ft18/Ft24 ─┐
                                                          │
目标 value crop ─→ 共享冻结 DINO ─→ Fx6/Fx12/Fx18/Fx24 ─┤
                                                          ↓
                              共享 token projection: 1024 → 128
                                                          ↓
                                Pair Interaction Adapter（逐层）
                    [Ft, Fx, |Ft-Fx|, Ft*Fx, local correlation/confidence]
                                                          ↓
                             interaction projection: 约513 → 128
                             depthwise 3×3 + residual adapter
                                                          ↓
                              多层融合 → change heatmap
                                                          ↓
               value_bbox 限制有效区域；changed_bbox 只做训练定位监督
                                                          ↓
                              heatmap 加权池化 → 128维 pair feature
                                                          ↓
                              轻量 Linear(1) → same/different
```

### 5.1 共享双路 backbone

模板和目标仍分别进入同一个 DINOv2，参数完全共享。这样保留官方预训练权重，不需要修改第一层输入通道。

工程上可以把模板和目标沿 batch 维拼接后只调用一次 backbone，再拆成两组特征；这可以提高吞吐，但不改变模型语义。

### 5.2 Token 级成对交互

每个选定层构造：

```text
pair_token = concat(
    Pt,
    Px,
    abs(Pt - Px),
    Pt * Px,
    local_match_confidence
)
```

其中 `Pt/Px` 是模板和目标的 1024 维 DINO token 经过共享 `1024→128` projection 后的结果，因此拼接维度约为 `128×4+1=513`，再由 interaction projection 压回 128 维。

其中：

- `Pt/Px` 保留有方向的模板和目标信息；
- `|Pt-Px|` 表示变化强度；
- `Pt*Px` 表示局部相似性；
- `local correlation` 允许目标 token 在模板附近寻找对应；
- `match confidence` 用于抑制不可靠配准位置。

相比 M11，这些交互在空间池化之前完成，模型仍能知道差异出现在哪个 token。

### 5.3 Local correlation 或双向 cross-attention

第一版优先使用小窗口局部相关性，不使用全局 cross-attention：

```text
每个目标 token
  → 在模板相同位置附近 ±2 patch 搜索
  → 输出 soft correspondence distribution
  → 同时输出匹配置信度/熵
```

局部窗口更符合标签已经对齐的前提，也不容易把真实变化字符匹配到远处的相似字符。确认有效后，再尝试双向局部 cross-attention。

### 5.4 Pair adapter

推荐轻量结构：

```text
LayerNorm
→ 1×1 Conv / Linear 降到 128维
→ GELU
→ depthwise 3×3 Conv
→ GroupNorm
→ GELU
→ residual connection
```

这里的 adapter 与 M05 不同。M05 适配的是已经统计池化后的向量；新 adapter 作用在带空间位置的成对 token 上，因此 M05 的负结果不能直接否定该方案。

## 6. 双框如何进入模型

### 6.1 `value_bbox`

`value_bbox`在训练和推理阶段都可以使用：

- 从完整标签中裁出业务值；
- 生成 value support mask；
- 限制 local correlation、heatmap 和池化范围；
- 防止模型关注字段名、相邻行和条码。

### 6.2 `changed_bbox`

`changed_bbox`只在训练和评估阶段使用：

- 生成字符级变化 mask；
- 监督 change heatmap；
- 计算 pointing accuracy、IoU、Dice；
- 检查模型是否真的关注变化字符。

推理时不能输入 `changed_bbox`，否则属于标签泄漏。

## 7. 训练目标

建议联合损失：

```text
L = L_cls + λ_loc * L_loc + λ_consistency * L_consistency
```

其中：

- `L_cls`：pair 二分类 BCE 或 focal loss；
- `L_loc`：`changed_bbox` mask 上的 BCE + Dice；
- `L_consistency`：同一 source 的不同拍摄增强保持相近决策和定位。

建议第一轮设置：

```text
L_cls = BCEWithLogitsLoss
L_loc = 0.5 * BCE + 0.5 * Dice
λ_loc = 0.5
λ_consistency = 0（先不启用）
```

先验证分类和定位是否同时改善，再加入一致性损失，避免一次引入过多变量。

## 8. 训练数据前置条件

D06 当前 1000 条全部是 positive，必须补齐以下负样本：

1. 模板内容不变的 print-scan、模糊、曝光、噪声和 JPEG 样本；
2. 轻微透视、尺度和配准残差，但内容不变的样本；
3. 主流程真实误触发候选形成的 hard negative；
4. 变更样本中与真实 `changed_bbox` 无关的其他候选。

负样本定位 mask 应全零。正负样本必须共享同一批成像增强类型，避免模型学习“某种增强等于负样本”的捷径。

原始样本及其所有增强副本必须按 `source_sample_id`放入同一个 split。

## 9. 推荐实验顺序

| 实验 | 结构 | 目的 |
| --- | --- | --- |
| A0 | 干净数据上复现 M11 | 建立新数据基线 |
| A1 | M11 + token Pair Adapter | 验证空间交互和定位监督 |
| A2 | A1 + soft local correlation/confidence | 验证可学习对应关系 |
| A3 | A2 + OCR | 单独确认 OCR 增益 |
| A4 | 6通道或横向早期融合 | 仅作为输入融合消融 |

每轮固定：

- 相同 train/val/test source split；
- 相同 DINOv2 checkpoint 和输入尺寸；
- 三个随机 seed；
- 阈值只从 validation 选择；
- real50/latest15 仅作历史开发集，不再用于选结构、阈值或 seed；
- 最终使用新的真实盲测集报告结果。

## 10. 评价指标

分类层：

- Precision、Recall、F1、PR-AUC；
- 三 seed 均值和标准差；
- validation 阈值跨 seed 波动。

定位层：

- heatmap peak 是否落入 `changed_bbox`；
- changed mask IoU/Dice；
- 上标、插入、删除等类型的分层 Recall。

业务层：

- 完整主流程最终 display box 的 TP/FP/FN；
- 单图平均候选数和总耗时；
- 每模板、每变异类型的错误审计。

## 11. 进入下一阶段的条件

A1/A2 只有同时满足以下条件才继续：

1. synthetic validation/test 三 seed F1 不低于干净 M11 基线；
2. 定位指标稳定提升，heatmap 峰值确实落在变化字符；
3. 全新真实 validation 的三 seed 均值提升，而非单 seed 偶然提升；
4. FP 没有集中来自某个模板或某种拍摄增强；
5. 推理耗时仍满足桌面端候选判别要求。

如果只有合成集改善、真实集 Recall 下降，应停止继续增加 adapter 容量，优先检查合成到真实的成像域差异和阈值校准。

## 12. 已完成实验

- A0/A1 首轮结果：[`d06_pair_interaction_a0_a1_experiment.md`](../experiments/d06_pair_interaction_a0_a1_experiment.md)
- P0-P4 白边、valid mask、矩形分桶与 Pair Adapter 消融：[`d06_padding_ablation_p0_p4_experiment.md`](../experiments/d06_padding_ablation_p0_p4_experiment.md)
