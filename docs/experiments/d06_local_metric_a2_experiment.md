# D06 局部度量损失 A2 实验

日期：2026-08-01

## 目的

验证“相同区域特征拉近、changed 区域特征推远”的局部监督式度量目标，是否能在现有 P4 Pair Interaction Adapter 上提升字符级差异检测。

## 数据

使用：

```text
/home/jnu/projects/dataset/outputs/datasets/D06_single_mutation_1k_v2_boxes
```

该目录本身包含 1000 条变更源样本，全部是 `different`。实际训练使用其派生的平衡 pair 数据：

```text
/home/jnu/projects/dataset/outputs/datasets/D06_pair_training_v1
```

| Split | different | same | 总数 |
|---|---:|---:|---:|
| train | 700 | 700 | 1400 |
| val | 150 | 150 | 300 |
| test | 150 | 150 | 300 |

`same` 是同模板内容不变的光度增强，`different` 是单处字符或上标变更。`value_bbox` 作为推理时的候选支持，`changed_bbox` 只作为训练时的局部监督。

## 模型改动

没有改变 backbone、输入尺寸、局部对应半径、heatmap 或分类头。现有 P4 仍执行：

```text
共享 DINO token
→ 双向局部 soft correspondence
→ residual / product / confidence interaction
→ change heatmap
→ support 加权池化
→ Linear(1) 分类
```

新增一张无参数的 token 距离图：

```text
D = max(1 - cos(target, matched_template),
        1 - cos(template, matched_target))
```

训练增加：

```text
L_metric = unchanged_distance²
         + max(margin - changed_distance, 0)²

L = L_BCE + 0.5 L_localization + 0.25 L_metric
margin = 0.4
```

因此这是“P4 + 局部度量辅助损失”，不是纯 SimCLR，也没有删除最终二分类决策头。

代码位置：

- [`pair_interaction.py`](../../../aa_clip_exp/label_pair/pair_interaction.py)
- [`train_pair_interaction.py`](../../../aa_clip_exp/label_pair/train_pair_interaction.py)
- [`evaluate_pair_interaction_real.py`](../../../aa_clip_exp/label_pair/evaluate_pair_interaction_real.py)

## 合成测试结果

旧 A1 使用相同 split 和训练配置作为基线。

| 方法 | Precision | Recall | F1 | Pointing | IoU@0.5 |
|---|---:|---:|---:|---:|---:|
| A1 P4 + BCE | 1.0000 | 0.9444 ± 0.0371 | 0.9711 ± 0.0198 | 0.7756 ± 0.0913 | 0.5125 ± 0.0700 |
| A2 + 局部 metric | 1.0000 | 0.9667 ± 0.0144 | **0.9830 ± 0.0074** | **0.8644 ± 0.0175** | **0.5644 ± 0.0050** |

A2 在 D06 合成测试上提升了分类召回、F1 和定位稳定性。三个 seed 的 changed token 平均距离约 `0.277`，same token 约 `0.066`；但只有约 `17.4%` 的 changed token 超过设定的 `0.4` margin，说明 margin 对当前特征尺度偏高。

## real50 冻结迁移

使用合成验证集固定阈值，不在 real50 上调参：

| 方法 | 候选 F1 | 框级 F1 |
|---|---:|---:|
| A1 | 0.5412 ± 0.0754 | 0.5237 ± 0.0732 |
| A2 + 局部 metric | 0.4616 ± 0.1096 | 0.4477 ± 0.1072 |

结果文件：

```text
/home/jnu/projects/aa_clip_exp/results/label_pair/a1_d06_pair_interaction_real50_20260801/comparison.json
/home/jnu/projects/aa_clip_exp/results/label_pair/a2_d06_local_metric_real50_20260801/comparison.json
```

例如 A2 seed20260712 的 real50 same/different 支持区域距离均值分别为 `0.1041` 和 `0.1008`，几乎没有可分性。说明合成数据中的度量约束没有学到真实拍摄下的缺陷距离。

## 结论

1. **模型结构方向合理。** P4 上增加局部距离图和辅助度量损失，不需要把系统改成纯对比学习，也不需要删除分类头。
2. **D06 合成数据足以验证机制，但不足以验证真实泛化。** A2 在合成数据上明显优于 A1，在 real50 反而下降。
3. **不能只降低 margin 或继续堆叠 adapter 来解决。** 当前主要瓶颈是训练分布缺少真实 same pair、印刷扫描、配准残差和 hard negative。
4. 下一轮应保持 A2 结构，加入真实正常实拍对、真实缺陷对和主流程误报候选；先使用 `metric_weight=0.1/0.25`、`margin=0.2` 做小规模消融，再固定配置进行 real50 盲测。

## 产物

```text
/home/jnu/projects/aa_clip_exp/results/label_pair/a2_d06_local_metric_w025_m04_seed20260710
/home/jnu/projects/aa_clip_exp/results/label_pair/a2_d06_local_metric_w025_m04_seed20260711
/home/jnu/projects/aa_clip_exp/results/label_pair/a2_d06_local_metric_w025_m04_seed20260712
/home/jnu/projects/aa_clip_exp/results/label_pair/a2_d06_local_metric_real50_20260801
```
