# D06 输入白边与 Pair Adapter P0-P4 实验

> 实验日期：2026-07-28  
> 数据：`D06_pair_training_v1`  
> Backbone：冻结 DINOv2-L/14，patch size 14，提取第 6/12/18/24 层  
> 随机种子：20260710、20260711、20260712

## 1. 实验问题

本实验依次回答三个问题：

1. 336 正方形 letterbox 的白边数值是否干扰 DINO 特征；
2. 无效 padding token 是否稀释 M11 的统计池化；
3. 在近似相同 token 数下，将 token 从无效高度重新分配到标签宽度，是否有利于分类和定位。

比较项固定为：

| 实验 | 输入与结构 | 目的 |
| --- | --- | --- |
| P0 | 336×336 白色 letterbox，原 M11 统计池化 | 基线 |
| P1 | 336×336 ImageNet 均值填充，valid mask | 分离白边像素值影响 |
| P2 | 336×336 白色 letterbox，valid-aware pooling | 检查 padding token 的池化稀释 |
| P3 | 宽高比分桶、均值填充、valid mask，M11 head | 检查有效横向分辨率收益 |
| P4 | P3 输入与 Pair Interaction Adapter | 检查矩形 token 上的空间交互与定位 |

P0 与 P2 复用完全相同的 DINO 前向特征，只改变池化是否使用 valid mask；因此二者能直接隔离池化影响。P1 的填充在归一化前使用 ImageNet mean，归一化后 padding 精确为 0。

## 2. 数据与共同设置

数据路径：

```text
/home/jnu/projects/dataset/outputs/datasets/D06_pair_training_v1
```

| Split | Different | Same | 合计 |
| --- | ---: | ---: | ---: |
| Train | 700 | 700 | 1400 |
| Validation | 150 | 150 | 300 |
| Test | 150 | 150 | 300 |

`different` 是模板 value crop 与单字符变更 value crop；`same` 是模板与独立光度增强后的同内容 crop。所有实验使用相同 split、相同冻结 backbone、30 epoch 和三个 seed。阈值只从 validation 选择：在 `Recall >= 0.95` 的候选中取 Precision 最高者。

P0-P3 使用四层 patch 绝对差统计和 128 维 MLP。P4 使用 128 维 Pair Interaction Adapter，并训练：

```text
L = weighted BCE + 0.5 * (focal localization + Dice)
```

`value_bbox` 生成候选 support，`changed_bbox` 只生成训练定位目标，不作为推理输入。

## 3. 正式矩形分桶

P3/P4 不是简单降低输入高度，而是在近似固定约 576 个 token 的前提下提高宽标签的横向分辨率：

| 输入尺寸（宽×高） | Token 网格 | Token 数 |
| --- | --- | ---: |
| 336×336 | 24×24 | 576 |
| 476×238 | 34×17 | 578 |
| 672×168 | 48×12 | 576 |
| 952×112 | 68×8 | 544 |

样本按原始 crop 宽高比与桶宽高比的 log 距离选择最近桶。图像、candidate mask、mutation mask 和 valid mask 使用同一套缩放与居中偏移，防止监督错位。

曾运行过一轮 `336×336 / 336×168 / 336×84 / 336×42` 的低计算量探索。它只减少竖向 token，没有提高横向 token 数，不能回答“有效分辨率是否有收益”，因此不纳入正式结论。该轮产物保留用于审计。

## 4. 三 seed 正式结果

下表均为独立 test split 的均值 ± 样本标准差：

| 实验 | Precision | Recall | F1 | ROC-AUC | PR-AUC | Pointing | IoU@0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P0 | 1.0000 ± 0.0000 | 0.9578 ± 0.0038 | 0.9784 ± 0.0020 | 0.99929 ± 0.00054 | 0.99931 ± 0.00051 | - | - |
| P1 | 1.0000 ± 0.0000 | 0.9578 ± 0.0077 | 0.9784 ± 0.0040 | 0.99993 ± 0.00013 | 0.99993 ± 0.00013 | - | - |
| P2 | 1.0000 ± 0.0000 | 0.9578 ± 0.0038 | 0.9784 ± 0.0020 | 0.99904 ± 0.00067 | 0.99907 ± 0.00062 | - | - |
| P3 | 1.0000 ± 0.0000 | 0.9444 ± 0.0038 | 0.9714 ± 0.0020 | 0.99976 ± 0.00005 | 0.99977 ± 0.00005 | - | - |
| P4 | 1.0000 ± 0.0000 | 0.9622 ± 0.0154 | **0.9807 ± 0.0080** | **0.99999 ± 0.00003** | **0.99999 ± 0.00003** | 0.9044 ± 0.0192 | 0.5913 ± 0.0199 |

P4 各 seed：

| Seed | Threshold | TP/FP/FN/TN | Recall | F1 | Pointing | IoU@0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20260710 | 0.999165 | 143/0/7/150 | 0.9533 | 0.9761 | 0.8933 | 0.5684 |
| 20260711 | 0.999356 | 147/0/3/150 | 0.9800 | 0.9899 | 0.9267 | 0.6050 |
| 20260712 | 0.999827 | 143/0/7/150 | 0.9533 | 0.9761 | 0.8933 | 0.6005 |

## 5. 结果解释

1. **白边像素值不是当前主要瓶颈。** P1 与 P0 的 F1 均约为 0.9784。均值填充提高了 AUC 并减小部分阈值波动，但没有形成稳定的二分类收益。
2. **padding 对 M11 池化的稀释也不是主要瓶颈。** P2 与 P0 的 Recall/F1 均值相同。原因是 M11 的主要候选统计已经受 `value_bbox` mask 约束，valid mask 只进一步修正全局均值项。
3. **单独提高横向 token 分辨率没有帮助原 M11 head。** P3 的 F1 比 P0 低约 0.0070。可能原因包括 DINO 预训练位置分布发生变化，以及最宽桶只保留 8 行 token；目前证据只说明“P3 + 原统计头”无收益，不能推导矩形输入本身无效。
4. **P4 是本轮均值最优结构，但优势较小且有 seed 波动。** P4 比 P0 的平均 F1 高约 0.0023，相当于 300 条 test 中约少漏 1 条；三 seed 中只有 seed11 明显高于基线。它更明确的收益是获得 90.4% pointing accuracy 和 59.1% IoU@0.5。
5. **P4 分数明显饱和。** 三个验证阈值都接近 0.999，部署前必须在真实 validation 上重新校准，不能直接使用当前阈值。

## 6. 当前结论与下一步

本轮支持将 **P4 作为下一阶段候选结构**，但不支持现在替换生产模型。D06 的 same 样本只有同模板光度增强，没有真实印刷扫描、拍摄透视、弯曲、配准残差和主流程 hard negative；当前 test 已接近饱和，结构差异容易被一两个样本和随机种子放大。

下一步应固定 P0 与 P4，只做一次真实数据对照：

1. 收集内容不变但含真实拍摄、弯曲和配准误差的 same pair；
2. 加入主流程误报形成的 hard negative；
3. 用真实 validation 独立选阈值，再在未参与选择的真实 test 上报告三 seed；
4. 同时报告最终候选框 TP/FP/FN、定位指标和耗时。

历史真实开发集复测已经完成，结果见 [`d06_padding_ablation_real_development_evaluation.md`](d06_padding_ablation_real_development_evaluation.md)。P4 的平均 F1 高于 P0，但三 seed 波动较大，且 real50 的 PR-AUC 下降，因此尚不满足部署条件。

## 7. 代码与产物

实验实现：

```text
/home/jnu/projects/aa_clip_exp/label_pair/run_padding_ablation.py
/home/jnu/projects/aa_clip_exp/label_pair/model.py
/home/jnu/projects/aa_clip_exp/label_pair/test_pair_interaction.py
```

正式产物：

```text
/home/jnu/projects/aa_clip_exp/results/label_pair/padding_ablation_p0_p4_constant_tokens/comparison.json
/home/jnu/projects/aa_clip_exp/results/label_pair/padding_ablation_p0_p4_constant_tokens/  # 约 100 MB
/home/jnu/projects/aa_clip_exp/results/label_pair/feature_cache/d06_padding_ablation_p0_p4_constant_tokens.pt  # 约 4.9 GB
```

不纳入正式结论的低计算量探索产物：

```text
/home/jnu/projects/aa_clip_exp/results/label_pair/padding_ablation_p0_p4/  # 约 100 MB
/home/jnu/projects/aa_clip_exp/results/label_pair/feature_cache/d06_padding_ablation_p0_p4.pt  # 约 2.1 GB
```
