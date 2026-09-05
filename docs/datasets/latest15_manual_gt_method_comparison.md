# Latest15 Manual GT Method Comparison

本文档记录 2026-07-01 这批 15 张桌面端实拍样本的人工 GT、主流程和 hybrid 方法对比验证过程。

## 目标

验证两种方法在同一批真实拍照样本上的效果和效率：

- `traditional`：当前桌面端默认主流程，最后一步使用 VLM 过滤候选区域。
- `hybrid`：文字区域使用 PP-OCRv6 比对，非文字区域继续使用 VLM 判断。

验证不使用任一方法自己的输出作为真值，而是单独建立一套人工 GT 标注集。之后用同一份 GT 同时评估两种方法。

## 输入数据

本次样本来自桌面端拍照目录：

```text
results/desktop_app/captures
```

固定的 15 条样本 manifest：

```text
results/batch_desktop_captures/latest15_check_20260701/manifest.json
```

每条样本包含：

- `code`
- `case_id`
- 模板 PDF 路径
- 实拍图路径

## 方法输出目录

主流程批量结果：

```text
results/batch_desktop_captures/latest15_traditional_20260701_rerun
```

hybrid 批量结果：

```text
results/batch_desktop_captures/latest15_hybrid_20260701_rerun
```

两个目录下的关键文件：

```text
summary.csv
runs/<code>/<case_id>/result.json
runs/<code>/<case_id>/annotation_base.jpg
runs/<code>/<case_id>/final_result.jpg
```

其中 `result.json` 中的 `final_boxes` 是该方法最终输出框，`detection_duration_seconds` 是检测耗时。

## 人工 GT 工作区

为避免污染两种方法的预测结果，单独建立人工 GT 工作区：

```text
results/manual_gt/latest15_20260701
```

该工作区包含一套桌面端可加载的历史记录：

```text
results/manual_gt/latest15_20260701/results/desktop_app/history.json
```

每条样本的标注文件保存为：

```text
results/manual_gt/latest15_20260701/results/desktop_app/<code>/<case_id>/manual_annotation.json
```

GT 标注只使用 `manual_gt_boxes`，也就是人工在 `annotation_base.jpg` 上拖拽出来的真实异常框。初始创建 GT 工作区的脚本为：

```bash
python3 scripts/create_manual_gt_workspace.py \
  --manifest results/batch_desktop_captures/latest15_check_20260701/manifest.json \
  --source-run-dir results/batch_desktop_captures/latest15_hybrid_20260701_rerun \
  --workspace-dir results/manual_gt/latest15_20260701
```

这里使用 hybrid 结果目录只是为了复制同坐标系的 `annotation_base.jpg`，不使用 hybrid 的预测框作为 GT。

## 桌面端标注入口

桌面端历史记录区增加了“历史来源”下拉框：

- `默认历史`
- `Manual GT latest15_20260701`

平时保持 `默认历史`，桌面端检测流程和默认输出不变。需要标注这 15 条 GT 时，选择 `Manual GT latest15_20260701`，历史列表会加载：

```text
results/manual_gt/latest15_20260701/results/desktop_app/history.json
```

标注操作：

1. 在历史列表中选择样本。
2. 在结果图上拖拽真实异常区域。
3. 生成蓝色漏检框。
4. 点击 `保存标注`。

保存后每个样本目录会生成 `manual_annotation.json`。本次 15 条样本均已保存 GT。

## 坐标系

三份数据使用同一坐标系：

```text
aligned_label_image
```

也就是说：

- GT 框来自 `manual_annotation.json` 的 `manual_gt_boxes[*].bbox`
- traditional 预测框来自 `result.json` 的 `final_boxes`
- hybrid 预测框来自 `result.json` 的 `final_boxes`

评估时可以直接比较框坐标，不需要再做原图到对齐图的映射。

预测框优先使用：

```text
display_box
```

如果没有 `display_box`，则回退到：

```text
box
```

## 评估规则

评估脚本：

```text
scripts/evaluate_batch_against_manual_gt.py
```

只保留一套位置重叠规则：

- 重叠分数定义为 `交集面积 / min(预测框面积, GT 框面积)`。
- `overlap >= 0.3` 记为 TP。
- 未匹配上的预测框记为 FP。
- 未匹配上的 GT 框记为 FN。
- 同一个 GT 只能被一个预测框匹配，按 overlap 从高到低贪心匹配。

该规则关注“预测是否落在人工标出的真实错误位置”。当小预测框位于较大的人工 GT
框内部时，分数仍可达到 1，不会像 IoU 那样仅因两框大小不同而判错；同时要求较小框
至少有 30% 与另一框相交，避免只擦到边缘一个像素也算命中。

复跑命令：

```bash
uv run python scripts/evaluate_batch_against_manual_gt.py \
  --method traditional=results/batch_desktop_captures/latest15_traditional_20260701_rerun \
  --method hybrid=results/batch_desktop_captures/latest15_hybrid_20260701_rerun \
  --output-dir results/evaluation/latest15_gt_compare_20260701
```

如果当前环境中没有 `uv`，但 Python 环境已包含 `cv2` 和 `numpy`，也可以直接使用对应环境的 `python` 运行。

## 输出目录

评估输出目录：

```text
results/evaluation/latest15_gt_compare_20260701
```

关键文件：

```text
summary.json
per_case.csv
matches.csv
visualizations/traditional/<case_id>.jpg
visualizations/hybrid/<case_id>.jpg
visualizations/compare/<case_id>.jpg
```

含义：

- `summary.json`
  两种方法的总体 TP/FP/FN、Precision、Recall、F1、耗时和请求数量统计。
- `per_case.csv`
  每条样本每种方法一行，便于定位具体失败样本。
- `matches.csv`
  框级匹配明细，包括每个 TP/FP/FN 的 bbox 和 overlap。
- `visualizations/traditional`
  主流程单方法可视化。
- `visualizations/hybrid`
  hybrid 单方法可视化。
- `visualizations/compare`
  traditional 与 hybrid 并排对照图。

可视化颜色：

- 绿色：TP，预测框命中 GT。
- 红色：FP，预测框误报。
- 蓝色：FN，GT 漏检。

每张图标题栏包含该样本的 TP/FP/FN 和耗时。

## 本次结果

主指标使用 `overlap >= 0.3`。

| 方法 | TP | FP | FN | Precision | Recall | F1 | 平均耗时 | 总耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| traditional | 13 | 5 | 2 | 0.7222 | 0.8667 | 0.7879 | 23.140s | 347.105s |
| hybrid | 13 | 12 | 2 | 0.5200 | 0.8667 | 0.6500 | 3.578s | 53.677s |

请求数量统计：

| 方法 | raw candidates | merged candidates | OCRv6 requests | VLM requests |
| --- | ---: | ---: | ---: | ---: |
| traditional | 142 | 69 | 0 | 69 |
| hybrid | 142 | 69 | 126 | 6 |

## 结论

在本次 15 条样本上：

- 两种方法的 Recall 相同，`traditional` 的 Precision 和 F1 更高。
- `hybrid` 速度明显更快，平均耗时约为 `traditional` 的 15.5%。
- `hybrid` 大幅减少 VLM 请求，从 69 次降到 6 次，但当前 OCR 比对策略产生了更多 FP。

后续优化方向：

- 对 OCR 保留框增加更严格的文本匹配条件，降低 FP。
- 对 OCR 未识别、识别置信低或文本为空的区域保留回退 VLM，避免增加 FN。
- 分析 `visualizations/compare` 中 hybrid 的 FP/FN 样本，优先定位 OCR 文本归一化和候选框归属问题。
