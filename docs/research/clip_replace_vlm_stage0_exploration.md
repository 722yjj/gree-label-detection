# CLIP 替代 VLM keep/discard：端到端实验方案

> 更新时间：2026-07-10

## 目标

桌面端默认流程是 `traditional_full_image_diff`：传统 CV 完成预处理、对齐、diff、
候选提取和 review region 构造，最后由 VLM 对每个候选判 `keep/discard`。

本实验只替换最后的判别器：

```text
相同模板/实拍输入
  -> 相同预处理、对齐和候选框
  -> 相同 review_box、template_crop、target_crop
  -> VLM 或 CLIP 判 keep/discard
  -> 相同 refine/merge 和 final_boxes
  -> 同一份 latest15 人工 GT 评估
```

实验重点是完整流程的最终 TP/FP/FN、Precision、Recall、F1 和耗时，不再把历史 VLM
输出作为弱标签，也不把候选 crop AUC 当作最终效果。

## 实现

独立实验脚本：

```text
scripts/experiment_traditional_diff_clip_filter.py
```

脚本不修改桌面端默认主流程。它仍然调用现有
`run_traditional_full_image_diff()` 和 `apply_vlm_filter()`，只在实验进程内把原来的
OpenAI-compatible client 替换成 CLIP adapter。

因此以下逻辑与桌面主流程完全相同：

- panel 尺寸与对齐结果；
- standard / small-text 候选生成和合并；
- PDF 文字行 review box；
- micro crop 上采样；
- template/target review crop；
- keep 后的 display box refine、合并和最终 verdict。

CLIP adapter 截获原本要发给 VLM 的图片，使用前两张未标注的 template/target crop，
计算：

```text
diff_score = 1 - cosine(CLIP(template_crop), CLIP(target_crop))
diff_score >= threshold -> keep
diff_score < threshold  -> discard
```

VLM 对文字候选可能额外收到 difference-emphasis 图和 PDF 文字 prompt；当前裸 CLIP
实验只使用成对图片。这是模型能力差异，不是 crop 或主流程差异。

主项目 `.venv` 有完整标签检测依赖但没有 torch/open_clip；`aa_clip_exp/.venv` 有
CLIP 依赖但没有 Paddle/PyMuPDF。实验脚本由主项目解释器启动，并使用常驻子进程在
`aa_clip_exp` 环境加载一次 CLIP，避免合并两套环境。

## 运行 latest15

先用一个样本做冒烟验证：

```bash
cd /home/jnu/projects/gree-label-detection
.venv/bin/python scripts/experiment_traditional_diff_clip_filter.py \
  --manifest results/batch_desktop_captures/latest15_check_20260701/manifest.json \
  --output-dir results/batch_desktop_captures \
  --run-name latest15_clip_smoke \
  --limit 1 \
  --model ViT-B-32-quickgelu \
  --pretrained openai \
  --threshold 0.25
```

完整运行：

```bash
.venv/bin/python scripts/experiment_traditional_diff_clip_filter.py \
  --manifest results/batch_desktop_captures/latest15_check_20260701/manifest.json \
  --output-dir results/batch_desktop_captures \
  --run-name latest15_clip_vitb32_t025_20260710 \
  --model ViT-B-32-quickgelu \
  --pretrained openai \
  --threshold 0.25
```

`0.25` 只是裸 CLIP 首轮探索阈值，不代表已在独立验证集完成标定。比较多个阈值时，
应保留相同模型和相同 latest15 manifest，并在最终结论中记录阈值。

## 人工 GT 评估

统一使用一个位置匹配规则：

```text
overlap = intersection_area / min(pred_area, gt_area)
overlap >= 0.3 -> TP
```

运行 VLM 与 CLIP 对比：

```bash
.venv/bin/python scripts/evaluate_batch_against_manual_gt.py \
  --method vlm=results/batch_desktop_captures/latest15_traditional_20260701_rerun \
  --method clip=results/batch_desktop_captures/latest15_clip_vitb32_t025_20260710 \
  --overlap-threshold 0.3 \
  --output-dir results/evaluation/latest15_vlm_vs_clip_t025_20260710
```

输出继续使用 `summary.json`、`per_case.csv`、`matches.csv` 和逐图可视化。

## 首轮端到端结果

配置：`ViT-B-32-quickgelu/openai`，冻结 encoder，`diff_score >= 0.25` 判 keep。

已核对 VLM 历史批次与本次 CLIP 批次：

- 15/15 个 case 的 `candidate_boxes_scaled` 完全一致；
- 69/69 个 review candidate 的 `review_box` 完全一致；
- VLM 请求为 0，CLIP 请求为 69。

在同一份人工 GT、`overlap >= 0.3` 下：

| 方法 | TP | FP | FN | Precision | Recall | F1 | 平均耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| VLM | 13 | 5 | 2 | 0.7222 | 0.8667 | 0.7879 | 23.140s |
| CLIP cosine, t=0.25 | 3 | 4 | 12 | 0.4286 | 0.2000 | 0.2727 | 0.793s |

结果目录：

```text
results/batch_desktop_captures/latest15_clip_vitb32_t025_20260710
results/evaluation/latest15_vlm_vs_clip_t025_20260710
```

结论：端到端替换方式已经正确，速度收益明显；冻结 CLIP 的单一全局 cosine 阈值会
丢弃大量真实文字差异，当前不能直接替代 VLM。下一阶段应保持相同输入和评估流程，
训练 pair head 或 adapter，而不是再改变候选 crop。

## 当前边界

- 当前是冻结 CLIP + cosine 的裸基线，没有训练 pair head 或微调 backbone。
- CLIP 阈值决定 keep/discard 数量，必须明确记录并在独立数据上标定。
- 如果裸 CLIP 不够，下一步应在相同 review crop 数据上训练 pair head，而不是改变
  候选生成和 GT 评估流程。
