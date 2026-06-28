# Real Photo Manual Annotation

本文档记录桌面端真实拍照检测后的人工标注流程。目标是把实拍检测结果沉淀成可复查、可导出、可评估的真实照片数据集。

## 目标

真实任务流程是：

1. 打印标签。
2. 使用桌面端拍照。
3. 桌面端走正式检测流程。
4. 在检测实际使用的标签区域图上人工复核最终框。
5. 保存 `manual_annotation.json`。
6. 后续导出真实照片标注集并统计检测效果。

这里的标注不是在原始拍照图上完成。原始拍照图通常未经裁剪、未对齐、包含背景和透视变形；检测框坐标不属于原始拍照图坐标系。

## 坐标系

桌面端传统检测流程 `traditional_full_image_diff` 会先对输入实拍图做标签区域提取、透视/尺度处理和对齐，然后在处理后的标签图上做差异检测。

新结果目录中会保存：

```text
annotation_base.jpg
visualization_diff.jpg
final_result.json
manual_annotation.json
```

含义：

- `annotation_base.jpg`
  检测实际使用的已裁剪、已对齐标签区域图，不带检测框。人工标注优先使用这张图。
- `visualization_diff.jpg`
  和 `annotation_base.jpg` 同坐标系，带最终检测框的结果图。
- `final_result.json`
  检测结果，包含最终框、检测耗时、标注底图路径等信息。
- `manual_annotation.json`
  人工标注结果，保存在同一个 run 目录中。

`final_result.json` 中与标注相关的字段：

```json
{
  "annotation_image": ".../annotation_base.jpg",
  "annotation_coordinate_space": "aligned_label_image",
  "artifacts": {
    "annotation_base": ".../annotation_base.jpg",
    "visualization_diff": ".../visualization_diff.jpg"
  }
}
```

旧结果目录如果没有 `annotation_base.jpg`，桌面端会回退到 `visualization_diff.jpg`。这种情况下仍然可以标注，但底图会包含已有检测框。

## 桌面端标注方式

检测完成后，右侧结果图区域会显示标注底图，并由界面叠加最终检测框。

框颜色含义：

- 黄色：检测框尚未复核。
- 绿色：检测框被标为正确。
- 红色：检测框被标为错误。
- 蓝色：人工补画的漏检框。

框线设置较细：普通框 1px，选中框 2px。

操作方式：

1. 点击已有检测框。
2. 点击 `正确` 或 `错误`。
3. 如果存在漏检区域，在图上空白处拖拽，新增一个蓝色漏检框。
4. 如果漏检框画错，选中该蓝色框后点击 `删除漏检框`。
5. 点击 `保存标注`。

当前设计不再要求：

- 整张样本结论。
- 拍照质量标签。
- 手动输入坐标。
- 框级“不确定”。

原因是当前评估目标只关心最终问题框是否正确，以及是否存在漏检区域。

## 标注文件格式

`manual_annotation.json` 的核心字段如下：

```json
{
  "version": "real_photo_annotation_v1",
  "run_dir": ".../results/desktop_app/<code>/<run>",
  "code": "600004085656",
  "template": ".../samples/pdfs/600004085656.pdf",
  "target_image": ".../results/desktop_app/captures/...jpg",
  "coordinate_space": "aligned_label_image",
  "image": {
    "path": ".../annotation_base.jpg"
  },
  "detection_duration_seconds": 12.34,
  "predicted_boxes": [
    {
      "box_id": "pred_0",
      "bbox": [530, 417, 654, 474],
      "decision": "true_positive"
    }
  ],
  "manual_gt_boxes": [
    {
      "box_id": "miss_0",
      "bbox": [100, 200, 180, 240],
      "source": "manual_draw"
    }
  ],
  "notes": ""
}
```

字段说明：

- `predicted_boxes`
  检测流程最终输出的框。人工只标 `true_positive` 或 `false_positive`。
- `manual_gt_boxes`
  人工补画的漏检框。每个框代表一个检测流程没有框出的真实问题区域。
- `bbox`
  坐标均属于 `image.path` 指向的标注底图，也就是 `aligned_label_image` 坐标系。
- `detection_duration_seconds`
  检测耗时，只用于统计，不需要人工标注 GT。

## 导出真实照片数据集

从项目根目录运行：

```bash
.venv/bin/python scripts/export_real_photo_annotations.py \
  --source-root results/desktop_app \
  --output-dir results/real_photo_dataset
```

默认只导出已经人工复核过的样本。满足以下任一条件即认为已复核：

- 至少一个 `predicted_boxes` 被标为 `true_positive` 或 `false_positive`。
- 至少存在一个 `manual_gt_boxes`。

如果需要导出未复核草稿：

```bash
.venv/bin/python scripts/export_real_photo_annotations.py \
  --source-root results/desktop_app \
  --output-dir results/real_photo_dataset \
  --include-unreviewed
```

导出结果：

```text
results/real_photo_dataset/manifest.json
```

## 评估真实照片标注

运行：

```bash
.venv/bin/python scripts/evaluate_real_photo_annotations.py \
  --manifest results/real_photo_dataset/manifest.json \
  --output results/real_photo_dataset/summary.json
```

当前评估统计：

- `sample_count`
  导出的样本数。
- `reviewed_count`
  有人工框级复核信息的样本数。
- `box_decision_counts`
  `true_positive`、`false_positive`、`unreviewed` 等框级数量。
- `manual_missing_box_count`
  人工补画漏检框数量。
- `precision`
  `true_positive / (true_positive + false_positive)`。
- `recall_proxy`
  `true_positive / (true_positive + manual_missing_box_count)`。
- `detection_duration_seconds`
  检测耗时均值和中位数。

这里的 `recall_proxy` 是近似召回指标，因为真实照片数据集目前不是完整字符级 GT，而是人工在检测底图上补充漏检区域。

## 当前限制

- 标注发生在检测后的标签区域坐标系，不直接服务于原始拍照图坐标。
- 旧 run 没有 `annotation_base.jpg` 时，只能回退到带框的 `visualization_diff.jpg`。
- 如果后续要做更严格的训练/评估数据集，可以在当前 `manual_annotation.json` 基础上增加完整 GT 框类别、问题类型、标注人等字段。
