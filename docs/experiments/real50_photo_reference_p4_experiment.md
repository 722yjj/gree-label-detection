# Real50 正常实拍参考实验

日期：2026-07-31

## 目的

保持 real50 的候选生成、目标 crop、候选 mask、标签和 P4-D08 模型不变，只将最终判别器的 PDF 参考 crop 替换为正常实拍参考 crop，验证缩小 PDF/实拍域差异是否有帮助。

## 参考图构建

- 源候选：`real_single10_20260712_v9_gtcorrected_dualmask_modelrow`，共 264 个候选，52 正、212 负。
- 每个候选均映射到同模板的固定 PDF 文本行，最小区域重合系数为 0.7692。
- donor 必须来自同模板、同固定文本行、不同 real50 case。
- donor 的原图是已经配准的 `annotation_base.jpg`。
- 按当前候选原有的 `review_box_final` 从 donor 原图裁切，而不是把小文本行 crop 拉伸到候选框。
- donor 的 review box 与其人工 GT 扩展 2 px 后必须零相交。
- 264 个候选全部找到合格 donor，无 PDF 回退。
- 仅替换 `template_crop`；`target_crop`、`candidate_mask`、标签及候选坐标保持不变。

构建脚本：[`samples/build_real50_photo_reference_pairs.py`](../../samples/build_real50_photo_reference_pairs.py)

数据清单：

- PDF 基线：[`baseline_manifest.jsonl`](../../samples/real50_photo_reference_pairs_v1/baseline_manifest.jsonl)
- 实拍参考：[`photo_reference_manifest.jsonl`](../../samples/real50_photo_reference_pairs_v1/photo_reference_manifest.jsonl)
- donor 映射：[`mapping.jsonl`](../../samples/real50_photo_reference_pairs_v1/mapping.jsonl)
- 构建摘要：[`summary.json`](../../samples/real50_photo_reference_pairs_v1/summary.json)

## 评测设置

- 模型：冻结的 P4-D08 Pair Interaction Adapter。
- checkpoint：`padding_ablation_p0_p4_d08_symmetric_capture/p4_seed*`。
- seeds：20260710、20260711、20260712。
- 阈值：各 checkpoint 的合成验证集固定阈值；real50 未参与训练、阈值选择或调参。
- 评测输出：`aa_clip_exp/results/label_pair/p4_d08_real50_photo_reference_single_20260731`。

## 候选级结果

| 参考 | seed | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| PDF | 20260710 | 0.6410 | 0.9615 | 0.7692 | 50 | 28 | 2 |
| PDF | 20260711 | 0.6000 | 0.9808 | 0.7445 | 51 | 34 | 1 |
| PDF | 20260712 | 0.5952 | 0.9615 | 0.7353 | 50 | 34 | 2 |
| 实拍 | 20260710 | 0.6885 | 0.8077 | 0.7434 | 42 | 19 | 10 |
| 实拍 | 20260711 | 0.6757 | 0.9615 | 0.7937 | 50 | 24 | 2 |
| 实拍 | 20260712 | 0.7206 | 0.9423 | 0.8167 | 49 | 19 | 3 |

三种子均值：

| 参考 | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| PDF | 0.6121 | 0.9679 | 0.7497 | 0.9381 | 0.7269 |
| 实拍 | 0.6949 | 0.9038 | 0.7846 | 0.9602 | 0.8227 |
| 变化 | +0.0828 | -0.0641 | +0.0349 | +0.0221 | +0.0958 |

框级平均 F1 从 0.7308 提高到 0.7638；平均 FP 从 34.0 降至 22.67，但平均 FN 从 1.67 增至 5.0。

## 样本观察

seed 20260712 中，实拍参考相对 PDF 参考修正 19 个 `FP -> TN`，新增 4 个 `TN -> FP`；恢复 1 个 `FN -> TP`，但新增 2 个 `TP -> FN`。

PDF 域差异导致、随后被实拍参考修正的误报示例：

![PDF false positive 1](../../samples/real50_photo_reference_pairs_v1/error_samples/pdf_false_positives_fixed_by_photo/pdf_fp_01.png)

![PDF false positive 2](../../samples/real50_photo_reference_pairs_v1/error_samples/pdf_false_positives_fixed_by_photo/pdf_fp_02.png)

![PDF false positive 3](../../samples/real50_photo_reference_pairs_v1/error_samples/pdf_false_positives_fixed_by_photo/pdf_fp_03.png)

实拍参考新增漏检示例：

![Photo false negative 850](../../samples/real50_photo_reference_pairs_v1/error_samples/photo_false_negatives/photo_fn_01.png)

其中 `850m³/h` 候选在 seed 20260712 下由 PDF 参考概率 0.999923 降到实拍参考概率 0.259953，真实的上标差异被实拍共同噪声掩盖。

![Photo false negative model](../../samples/real50_photo_reference_pairs_v1/error_samples/photo_false_negatives/photo_fn_02.png)

## 结论

正常实拍参考确实降低了 PDF 与实拍之间的外观域差异，三种子平均 Precision、F1、ROC-AUC 和 PR-AUC 均提高，因此这个思路有效。当前版本仍不适合直接替换生产流程：P4-D08 只在 PDF/合成参考分布上训练，实拍参考使召回下降，而且 seed 20260710 出现 10 个 FN，说明模型和固定阈值对 photo-to-photo 输入尚未校准。

下一步应保留本实验的严格 donor 规则，在训练中加入正常实拍参考对或对称参考增强，再用同一 real50 清单做冻结复测。对 `m³/h`、型号字符等微小差异应单独保留高召回约束，避免整体域差异减小后同时抹掉真实细粒度异常。
