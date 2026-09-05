# Qwen3.8-27B-NVFP4 真实图片评测

> 当前配置和最新结论以文末“视觉提示词 + 跳过候选 OCR 评测”章节为准。前面的结果、失败案例和提示词均属于上一轮历史评测（含 OCR，仅供历史对照），不代表当前运行配置。

> 评测日期：2026-08-30  
> 模型：`PassingByPixels/Qwen3.8-27B-NVFP4`  
> 本地权重：`/home/jnu/models/Qwen3.8-27B-NVFP4`  
> vLLM：`/home/jnu/venvs/vllm/bin/vllm`（0.28.0）

## 目的

验证 PassingByPixels 针对 DGX Spark 优化的 Qwen3.8-27B NVFP4 是否可以直接替代当前主流程中的 VLM 判别器，并在既有两套真实开发集上取得可复现结果。

本次只替换 VLM 模型。候选生成、模板解析、预处理、SIFT/ECC 对齐、tolerant diff、review region、最终框 refine/merge 和输出格式均保持 `traditional_full_image_diff` 现有流程不变。

## 数据与人工 GT

| 数据集 | 输入 manifest | 样本数 | 人工 GT 根目录 |
| --- | --- | ---: | --- |
| latest15 | `results/batch_desktop_captures/latest15_check_20260701/manifest.json` | 15 | `results/manual_gt/latest15_20260701/results/desktop_app` |
| real50 | `results/batch_desktop_captures/single10_20260712_manifest/manifest.json` | 50 | `results/manual_gt/single10_20260712/results/desktop_app` |

这两套数据均是项目已有的真实开发集，曾用于多轮流程和模型选择，不是新的盲测集。manifest 与人工 GT 的 case 均完成一一对应，实际运行没有跳过样本。

## 运行方式

统一入口脚本：

```bash
cd /home/jnu/projects/gree-label-detection
scripts/run_qwen38_real_evaluation.sh --dataset both
```

脚本依次完成：

1. 启动一个本地 OpenAI-compatible vLLM 服务，模型名为 `qwen3.8-27b-nvfp4`。
2. 通过已有 `scripts/batch_run_desktop_captures.py` 运行 `traditional` + `--vlm-filter`。
3. 使用 `scripts/evaluate_batch_against_manual_gt.py` 对最终 `display_box` 评分。
4. 评测结束后自动关闭本次启动的 vLLM。

本次服务使用受限的单序列配置（KV cache 上限 3 GiB），避免评测服务侵占整张 GPU 显存；没有与已有服务并行启动。运行日志：

```text
results/vllm/qwen38_real_evaluation.log
```

## 评估规则

- 预测取 `result.json` 中的最终 `final_boxes`，优先使用 `display_box`。
- 重叠分数为 `intersection_area / min(pred_area, gt_area)`。
- `overlap >= 0.3` 记为 TP；未匹配预测为 FP；未匹配 GT 为 FN。
- 指标在完整 case 的最终框上统计，不把 raw/merged candidate 当作最终结果。

## 结果

### 总体指标

| 数据集 | Case | TP | FP | FN | Precision | Recall | F1 | 平均耗时 | VLM 请求 | 候选数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| latest15 | 15 | 9 | 0 | 6 | 1.0000 | 0.6000 | 0.7500 | 32.521s | 73 | 73 |
| real50 | 50 | 40 | 2 | 10 | 0.9524 | 0.8000 | 0.8696 | 34.437s | 264 | 264 |

运行总耗时分别为 `487.808s`（latest15）和 `1721.848s`（real50）。两套数据的批处理结果均为全部 case `success=True`，没有缺失 `result.json`。

### 输出位置

批处理结果：

```text
results/batch_desktop_captures/qwen38_nvfp4_latest15_20260830/
results/batch_desktop_captures/qwen38_nvfp4_real50_20260830/
```

人工 GT 评分结果：

```text
results/evaluation/qwen38_nvfp4_latest15_20260830/summary.json
results/evaluation/qwen38_nvfp4_latest15_20260830/per_case.csv
results/evaluation/qwen38_nvfp4_latest15_20260830/matches.csv

results/evaluation/qwen38_nvfp4_real50_20260830/summary.json
results/evaluation/qwen38_nvfp4_real50_20260830/per_case.csv
results/evaluation/qwen38_nvfp4_real50_20260830/matches.csv
```

## 现象与结论

- latest15 没有 FP，但漏检 6/15，F1 为 0.75；说明模型在这套较早观察集上偏保守。
- real50 命中 40/50，FP 为 2，F1 为 0.8696；误报较少，但仍有 10 个 GT 未被最终框命中。
- real50 的平均单图耗时约 34.4 秒，264 个候选均进入 VLM 请求；这次评测没有启用 CLIP/OCR 替代判别器。
- 结果只说明该模型在当前候选生成和最终框评估口径下的表现。漏检可能来自候选阶段，也可能来自 VLM 判别阶段，需结合 `per_case.csv`、`matches.csv` 和对应 `result.json` 分开定位。

完整汇总：

```text
results/evaluation/qwen38_nvfp4_latest15_20260830/summary.json
results/evaluation/qwen38_nvfp4_real50_20260830/summary.json
```

## 错误实例可视化

以下图片来自评估脚本生成的可视化结果。绿色为 TP，红色为 FP，蓝色为 FN；图片中的标题栏同时标出该 case 的 TP/FP/FN 和耗时。

### latest15

#### `600001076226_20260701-150140-302242_600001076226`（TP=0，FP=0，FN=1）

![latest15 600001076226 150140](../../results/evaluation/qwen38_nvfp4_latest15_20260830/visualizations/qwen38_nvfp4/600001076226_20260701-150140-302242_600001076226.jpg)

#### `600004075219_20260701-150453-773669_600004075219`（TP=0，FP=0，FN=1）

![latest15 600004075219 150453](../../results/evaluation/qwen38_nvfp4_latest15_20260830/visualizations/qwen38_nvfp4/600004075219_20260701-150453-773669_600004075219.jpg)

#### `600004078454_20260701-150343-074045_600004078454`（TP=0，FP=0，FN=1）

![latest15 600004078454 150343](../../results/evaluation/qwen38_nvfp4_latest15_20260830/visualizations/qwen38_nvfp4/600004078454_20260701-150343-074045_600004078454.jpg)

#### `600004083205_20260701-150414-076355_600004083205`（TP=0，FP=0，FN=1）

![latest15 600004083205 150414](../../results/evaluation/qwen38_nvfp4_latest15_20260830/visualizations/qwen38_nvfp4/600004083205_20260701-150414-076355_600004083205.jpg)

#### `600004083205_20260701-150424-135536_600004083205`（TP=0，FP=0，FN=1）

![latest15 600004083205 150424](../../results/evaluation/qwen38_nvfp4_latest15_20260830/visualizations/qwen38_nvfp4/600004083205_20260701-150424-135536_600004083205.jpg)

#### `600004085656_20260701-150539-671549_600004085656`（TP=0，FP=0，FN=1）

![latest15 600004085656 150539](../../results/evaluation/qwen38_nvfp4_latest15_20260830/visualizations/qwen38_nvfp4/600004085656_20260701-150539-671549_600004085656.jpg)

### real50

#### `600001076226_20260712-163402-052997_600001076226`（TP=0，FP=0，FN=1）

![real50 600001076226 163402](../../results/evaluation/qwen38_nvfp4_real50_20260830/visualizations/qwen38_nvfp4/600001076226_20260712-163402-052997_600001076226.jpg)

#### `600001076226_20260712-163639-445160_600001076226`（TP=0，FP=0，FN=1）

![real50 600001076226 163639](../../results/evaluation/qwen38_nvfp4_real50_20260830/visualizations/qwen38_nvfp4/600001076226_20260712-163639-445160_600001076226.jpg)

#### `600001076226_20260712-163822-306976_600001076226`（TP=0，FP=0，FN=1）

![real50 600001076226 163822](../../results/evaluation/qwen38_nvfp4_real50_20260830/visualizations/qwen38_nvfp4/600001076226_20260712-163822-306976_600001076226.jpg)

#### `600004075219_20260712-164019-751975_600004075219`（TP=1，FP=1，FN=0）

![real50 600004075219 164019](../../results/evaluation/qwen38_nvfp4_real50_20260830/visualizations/qwen38_nvfp4/600004075219_20260712-164019-751975_600004075219.jpg)

#### `600004075219_20260712-164128-331034_600004075219`（TP=0，FP=1，FN=1）

![real50 600004075219 164128](../../results/evaluation/qwen38_nvfp4_real50_20260830/visualizations/qwen38_nvfp4/600004075219_20260712-164128-331034_600004075219.jpg)

#### `600004078454_20260712-164604-208494_600004078454`（TP=0，FP=0，FN=1）

![real50 600004078454 164604](../../results/evaluation/qwen38_nvfp4_real50_20260830/visualizations/qwen38_nvfp4/600004078454_20260712-164604-208494_600004078454.jpg)

#### `600004078454_20260712-164613-760845_600004078454`（TP=0，FP=0，FN=1）

![real50 600004078454 164613](../../results/evaluation/qwen38_nvfp4_real50_20260830/visualizations/qwen38_nvfp4/600004078454_20260712-164613-760845_600004078454.jpg)

#### `600004083205_20260712-164900-679860_600004083205`（TP=0，FP=0，FN=1）

![real50 600004083205 164900](../../results/evaluation/qwen38_nvfp4_real50_20260830/visualizations/qwen38_nvfp4/600004083205_20260712-164900-679860_600004083205.jpg)

#### `600004083205_20260712-164948-666109_600004083205`（TP=0，FP=0，FN=1）

![real50 600004083205 164948](../../results/evaluation/qwen38_nvfp4_real50_20260830/visualizations/qwen38_nvfp4/600004083205_20260712-164948-666109_600004083205.jpg)

#### `600004083205_20260712-165005-026918_600004083205`（TP=0，FP=0，FN=1）

![real50 600004083205 165005](../../results/evaluation/qwen38_nvfp4_real50_20260830/visualizations/qwen38_nvfp4/600004083205_20260712-165005-026918_600004083205.jpg)

#### `600004083205_20260712-165134-447057_600004083205`（TP=0，FP=0，FN=1）

![real50 600004083205 165134](../../results/evaluation/qwen38_nvfp4_real50_20260830/visualizations/qwen38_nvfp4/600004083205_20260712-165134-447057_600004083205.jpg)

## 失败案例的 VLM 返回（上一轮历史评测，含 OCR，仅供历史对照）

下面的内容直接读取对应 `result.json` 的 `vlm_filter_results`。这些错误 case 都产生了候选，因此 FN 不是候选阶段完全没有触发：latest15 和 real50 中的纯 FN case，其相关候选最终都被 VLM 返回 `discard`，属于确认阶段漏检；real50 的 `600004075219_20260712-164128-331034_600004075219` 是例外，它有一个 `keep`，但保留框位置与 GT 不匹配，同时形成 FP 和 FN。表中的 OCR 是请求实际使用的 `target_ocr_hint.text`；VLM 的 `reason` 是模型自行生成的解释，可能与实际 OCR 字段不一致，不能反向当作 OCR 原文或人工 GT。

### latest15 错误 case

| case | TP/FP/FN | VLM 请求 | 相关返回（decision/confidence；target OCR；reason） |
| --- | ---: | ---: | --- |
| `600001076226_20260701-150140-302242_600001076226` | 0/0/1 | 2 | `discard/0.90`; 实际传入 OCR：`G.W.:16.5kg CalarWhite`；模型理由却说 OCR 把 `14` 读成 `1A`。这是 VLM 理由与请求字段不一致的错误归因，不能认为本次 OCR 返回过 `1A`。完整原文：“The target OCR misread '14' as '1A', but visual inspection confirms the character is a '4' in both images. All other text and symbols are identical.” |
| `600004075219_20260701-150453-773669_600004075219` | 0/0/1 | 6 | 所有 6 个候选均 `discard`（0.90–0.95）。与漏检区域最相关的返回：`discard/0.95`; `850m/h`; “Both images clearly display the text '850m³/h' with the superscript '3' visible in both crops. The target OCR missed the superscript, but visual inspection confirms the content is identical.” |
| `600004078454_20260701-150343-074045_600004078454` | 0/0/1 | 5 | 所有 5 个候选均 `discard`（0.90–0.95）。相关返回：`discard/0.90`; `GWHCACRKB-K3NNA1B`; “The text content is identical in both images. The target OCR error (reading 'I' as '1') is a misinterpretation of the final character, which is clearly the letter 'I' in both crops, matching the template.” |
| `600004083205_20260701-150414-076355_600004083205` | 0/0/1 | 7 | 所有 7 个候选均 `discard`（0.90–0.95）。相关返回：`discard/0.95`; `5.60kW`; “The text content is identical in both images. The target image is slightly blurrier, but all glyphs (5, ., 6, 0, k, W) are clearly present and match the template without any character substitution or missing marks.” |
| `600004083205_20260701-150424-135536_600004083205` | 0/0/1 | 6 | 所有 6 个候选均 `discard`（0.90–0.95）。相关返回：`discard/0.90`; `GWH18AU0XE-K6DNA1AN`; “The final character in both images is clearly the letter 'I' (a vertical bar with serifs), not the number '1' or '7' as suggested by the OCR. The text content is identical in both crops.” |
| `600004085656_20260701-150539-671549_600004085656` | 0/0/1 | 3 | 所有 3 个候选均 `discard`（0.90–0.90）。相关返回：`discard/0.90`; `M0000`; “The first character in the target image is clearly the digit '1', matching the template. The OCR misread it as 'I', but visual inspection confirms the glyph is identical to the template.” |

### real50 错误 case

| case | TP/FP/FN | VLM 请求 | 相关返回（decision/confidence；target OCR；reason） |
| --- | ---: | ---: | --- |
| `600001076226_20260712-163402-052997_600001076226` | 0/0/1 | 2 | 所有候选 `discard`；`discard/0.90`; `GYH24AGD-K6BNA1CWIF`; “The text content is identical in both images. The target image is simply a lower-contrast or grayscale version of the template, but all characters and symbols match exactly.” |
| `600001076226_20260712-163639-445160_600001076226` | 0/0/1 | 1 | `discard/0.90`; `GMH24AGD-K6DNA1CIE`; “The text content is identical in both images. The target image shows a minor printing artifact or smudge on the opening parenthesis, but all alphanumeric characters and symbols are present and unchanged.” |
| `600001076226_20260712-163822-306976_600001076226` | 0/0/1 | 2 | 所有候选 `discard`；`discard/0.90`; `WH24AGD-K6BNA1CIWIF`; “The text content is identical in both images. The target image is slightly blurrier and the font weight appears lighter, but all characters, including the 'I' in 'WIFI' and the parentheses, are present and match the template. The OCR error is due to image quality, not a content change.” |
| `600004075219_20260712-164019-751975_600004075219` | 1/1/0 | 8 | FP 对应请求返回 `keep/0.95`; OCR `ANCE`; “The target image is missing the 'S' in 'APPLIANCES' and the entire segment ',INC.OF ZHU' compared to the template.”（同 case 的另一个 `keep/0.95` 认为 `2026.-01` 多了连字符并形成 TP。） |
| `600004075219_20260712-164128-331034_600004075219` | 0/1/1 | 11 | 同一个请求 `keep/0.75` 同时造成 FP 和 FN；OCR `Weigh`; “The target image shows a distinct character (resembling a 'B' or '3') immediately to the left of the 'W' in 'Weight', which is absent in the template image.” 该 keep 框与 GT 仅部分偏移，故既未命中 GT 又形成 FP。其余 10 个候选均 `discard`。 |
| `600004078454_20260712-164604-208494_600004078454` | 0/0/1 | 5 | 所有候选 `discard`；`discard/0.95`; `8.5kg`; “Both images clearly show the text '8.5kg'. The target image is slightly blurrier but the characters are identical to the template. No content differences are visible.” |
| `600004078454_20260712-164613-760845_600004078454` | 0/0/1 | 6 | 所有候选 `discard`；`discard/0.90`; `GWKIACBKB-K3NNAIBN`; “The text in both images is identical. The target OCR error (reading 'I' as 'N' or '1') is a misinterpretation of the final character which is clearly a vertical bar or 'I' in both images. No visual difference exists.” |
| `600004083205_20260712-164900-679860_600004083205` | 0/0/1 | 7 | 所有候选 `discard`；`discard/0.95`; `GWH18AUDXE-K60NAIAC`; “The text in both images is identical. The target OCR error (reading '0' instead of 'D' and 'C' instead of 'A') is clearly a misread of the visible glyphs, which match the template perfectly.” |
| `600004083205_20260712-164948-666109_600004083205` | 0/0/1 | 5 | 所有候选 `discard`；`discard/0.90`; `5.60KW`; “The 'k' in the target image is clearly lowercase, matching the template. The OCR's suggestion of 'KW' is an error; the visual evidence confirms the case is identical in both images.” |
| `600004083205_20260712-165005-026918_600004083205` | 0/0/1 | 6 | 所有候选 `discard`；`discard/0.90`; `5.60kW`; “The text content is identical in both images. The target image is slightly blurrier and has a different crop aspect ratio, but all visible glyphs (5, ., 6, 0, k, W) are present and unchanged.” |
| `600004083205_20260712-165134-447057_600004083205` | 0/0/1 | 6 | 所有候选 `discard`；`discard/0.90`; `5.30kV`; “The target image clearly shows a zero (0) with a slash, matching the template. The OCR misread the slashed zero as the letter O, but the visual evidence confirms the content is identical.” |

## 代表样本的实际输入提示词（上一轮历史评测，含 OCR，仅供历史对照）

以下选择 latest15 的 `600001076226_20260701-150140-302242_600001076226` 第一个 VLM 请求。`result.json` 没有持久化完整 prompt 文本；下面的 prompt 是依据 `label_detection/workflows/traditional_full_image_diff.py` 中的 `build_vlm_filter_prompt()` 及该请求的结果字段重建。`raw_response` 则是 `result.json` 中保存的模型原文。两张图片在请求时由 `prepare_vlm_candidate_crop()` 生成后直接编码为内存中的 base64，没有单独保存 crop 文件。

输入源：

```text
template: /home/jnu/projects/gree-label-detection/samples/pdfs/600001076226.pdf
target:   /home/jnu/projects/gree-label-detection/results/desktop_app/captures/600001076226/20260701-150140-302242_600001076226.jpg
candidate_focus_box: [0, 120, 900, 166]
review_box:          [0, 114, 872, 172]
matched template text: N.W. :14kg     G.W. :16.5kg     Color:White
matched segment:      :14kg G.W. :16.5kg
target OCR (actual request field): G.W.:16.5kg CalarWhite
target OCR mean confidence: 0.6200155814488729
template foreground ratio: 0.185048367057153
target foreground ratio:    0.1942170305289315
candidate source: candidate_group (因此没有 micro_text_candidate 专用附加规则)
```

按源码和上述动态字段重建的完整 prompt：

```text
You are checking one tightly focused candidate from a product label inspection.

You will see exactly two unannotated images of the same candidate area:
- Image 1 is the template candidate crop.
- Image 2 is the target candidate crop.

Expected template text extracted from the PDF:
N.W. :14kg     G.W. :16.5kg     Color:White

Target OCR hypothesis (fallible and possibly wrong):
G.W.:16.5kg CalarWhite
Target OCR mean confidence: 0.620

Independently read the printed content in both images before deciding whether it differs.
The PDF text is authoritative only for the template. The target OCR is only a hint:
- do not copy the PDF text onto an ambiguous target character
- do not accept the target OCR when it conflicts with visible image evidence
- OCR matching the PDF is neutral and is not evidence that the images are equal
- OCR often normalizes subtle changes such as 5/S, I/1/7, 3/8, or added thin strokes;
  when OCR matches the PDF, ignore that agreement and still compare every visible glyph
- when OCR differs from the PDF, inspect the exact proposed characters as a strong alert
- if OCR proposes a plausible confusable-character change and the target image is ambiguous,
  keep the candidate; discard it as an OCR error only when the target image clearly disproves it
- explicitly inspect confusable pairs such as I/l/1/7, O/0, S/5, Z/2, B/8, and G/6

Decide whether the real printed label content differs between Image 1 and Image 2.

This is a high-recall defect inspection. Use discard only when you can clearly verify that
every visible glyph or printed mark is the same. If blur prevents you from ruling out a plausible character, superscript, case, or thin-stroke change, keep the candidate. Do not use blur or low resolution by itself as a reason to discard a candidate that contains a plausible content difference.

Keep real differences:
- text character changes, including case changes or typos
- number, unit, model, date, barcode value changes
- icon/symbol/logo/barcode graphic added, removed, or meaningfully changed
- missing or extra printed content
- any printed icon, symbol, mark, or text that is visible in one crop but absent in the other, even if it is small or looks decorative

Discard false positives:
- slight alignment shift or crop padding difference
- blur, exposure, shadow, compression, camera noise
- line thickness, ink darkness, antialiasing, small edge residue
- white/black border or crop boundary artifacts

Important:
- Use the two candidate images as the source of truth.
- If one candidate crop is blank and the other contains visible printed content, output keep.
- Do not discard added or missing printed symbols just because they seem minor, decorative, or hard to interpret.
- A one-character or superscript/subscript change is a real difference even when OCR misses it.

Output exactly one JSON object and nothing else:
{"template_reading":"what is visible in Image 1","target_reading":"what is visible in Image 2","target_ocr_agrees":true,"decision":"keep|discard|unknown","confidence":0.0,"reason":"short visual reason"}
```

该请求的实际 `raw_response`（其中 `reason` 对 OCR 的 `1A` 描述与实际请求字段不一致）：

```json
{"template_reading":"N.W. :14kg     G.W. :16.5kg     Color:White","target_reading":"N.W. :14kg     G.W. :16.5kg     Color:White","target_ocr_agrees":false,"decision":"discard","confidence":0.9,"reason":"The target OCR misread '14' as '1A', but visual inspection confirms the character is a '4' in both images. All other text and symbols are identical."}
```

## 视觉提示词 + 跳过候选 OCR 评测（2026-08-30）

> 本章节是当前配置和最新评测结果的准确信息源；前文历史章节仅用于对照。

本次评测使用视觉对比提示词：VLM 输入仅包含模板候选图和目标候选图；PDF 文本仅作辅助参考，不向提示词注入目标 OCR。通过环境变量 `VLM_SKIP_TARGET_OCR=1` 跳过两个 VLM 分支中的 `recognize_vlm_target_crop()`，OCR 诊断字段为空；默认未设置该变量时行为保持不变。

执行时间（Asia/Shanghai）：2026-08-30 20:29:11 至 21:01:23，墙钟总耗时 1932.0 s（32 分 12 s）。批处理耗时：latest15 369.846 s（平均 24.656 s/case）；real50 1350.088 s（平均 27.002 s/case）。

| 数据集 | case | success | TP | FP | FN | Precision | Recall | F1 | VLM 请求/候选数 | 输出 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| latest15 | 15 | 15/15 | 12 | 0 | 3 | 1.0000 | 0.8000 | 0.8889 | 73/73 | `results/evaluation/qwen38_nvfp4_latest15_noocr_20260830/summary.json` |
| real50 | 50 | 50/50 | 43 | 2 | 7 | 0.9556 | 0.8600 | 0.9053 | 264/264 | `results/evaluation/qwen38_nvfp4_real50_noocr_20260830/summary.json` |

两套数据均无缺失结果（`missing_result_count=0`）。`latest15` 和 `real50` 均为开发集，不是未触碰的最终测试集。完整批处理输出分别位于 `results/batch_desktop_captures/qwen38_nvfp4_latest15_noocr_20260830/` 和 `results/batch_desktop_captures/qwen38_nvfp4_real50_noocr_20260830/`。

### 当前 no-OCR 评测的失败案例：VLM 实际输入 crop 与返回

> 本节在 2026-08-31 重写：展示对象从"整图 case 可视化"改为**送入 VLM 的那一对候选 crop**（模板 crop + 实拍 crop）加上该次请求的完整返回，便于直接观察模型看到了什么、判成了什么。

#### crop 的来源与可信度

批处理运行时没有开 `--save-debug`，候选 crop 未落盘，但重建这些 crop 所需的全部输入都记录在 `result.json` 里：源模板/实拍路径、`parameters`、每个候选的 `candidate_focus_box`，以及 crop 的前景比例 `template_foreground_ratio` / `target_foreground_ratio`。

`resolve_and_preprocess` → `align_target_to_template` → `scale_pair_for_canvas` → `prepare_vlm_candidate_crop` 这条链是确定性的，因此 crop 可以离线重建，**不需要重新请求 VLM，也不产生新的评测结果**。重建脚本：

```bash
scripts/export_vlm_candidate_crops.py \
  --eval-dir results/evaluation/qwen38_nvfp4_latest15_noocr_20260830
scripts/export_vlm_candidate_crops.py \
  --eval-dir results/evaluation/qwen38_nvfp4_real50_noocr_20260830
```

每个重建 crop 都用 `result.json` 中存的前景比例做校验（容差 `1e-9`），并核对 SIFT inliers 与 ECC 相关系数。本轮结果：

| 数据集 | 失败 case | 重建候选 | 前景比例校验通过 | 对齐结果复现 |
|---|---:|---:|---:|---:|
| latest15 | 3 | 11 | 11/11 | 3/3 |
| real50 | 8 | 45 | 45/45 | 8/8 |

也就是说下面每一张 crop 都与当时送进模型的图像逐像素一致，不是近似复现。所有候选的 `target_ocr_hint` 均为空，`vlm_input_mode` 均为 `two_image_candidate_focus_visual_only_no_target_ocr`，与 `VLM_SKIP_TARGET_OCR=1` 一致。

#### 阅读方式

- 拼接图上半为**模板 crop**，下半为**实拍 crop**，中间洋红色细线只是拼接标记，不属于图像内容；两张 crop 尺寸相同，可直接上下对位比较。
- 每个候选另存了未拼接的 `*_template.png` 和 `*_target.png`，即模型收到的两张原图。
- `focus overlap` 是候选 crop 区域与该错误框的 `intersection / min(area)`，与评测所用重叠度量一致；只展示 overlap ≥ 0.2 的候选。
- **阶段归因**：若某个 FN 的 GT 区域没有任何候选覆盖，说明它在候选生成阶段就已丢失，VLM 从未见过；若有候选覆盖却被 `discard`，则是判别器漏判。

本轮 10 个 FN 的归因结果：**候选生成阶段丢失 0 个，候选存在但被 VLM 判别器 `discard` 10 个**。所有被覆盖的 FN 的 focus overlap 都在 0.85 以上，即漏检区域确实完整落在送检 crop 内。

#### latest15

##### `600001076226_20260701-150140-302242_600001076226`（TP=0，FP=0，FN=1）

候选总数 2，VLM 请求 2，OCR 请求 0；panel 尺寸 900×265，标注空间 2009×592。

- **FN** `miss_0`：标注空间框 `[39, 271, 1810, 366]`，panel 空间框 `[17, 121, 811, 164]`；阶段归因 **VLM 判别器**。覆盖候选：`cand00` overlap=1.0 → `discard/0.9`。

**候选 `cand00`** — focus box（panel）`[0, 120, 900, 166]`，crop 尺寸 1878×96，匹配模板文本行 `:14kg G.W. :16.5kg`，候选来源 `candidate_group`。

![600001076226_20260701-150140-302242_600001076226 cand00 模板/实拍 crop](../../results/evaluation/qwen38_nvfp4_latest15_noocr_20260830/vlm_candidate_crops/600001076226_20260701-150140-302242_600001076226/cand00_pair.png)

未拼接原图：[`cand00_template.png`](../../results/evaluation/qwen38_nvfp4_latest15_noocr_20260830/vlm_candidate_crops/600001076226_20260701-150140-302242_600001076226/cand00_template.png) ／ [`cand00_target.png`](../../results/evaluation/qwen38_nvfp4_latest15_noocr_20260830/vlm_candidate_crops/600001076226_20260701-150140-302242_600001076226/cand00_target.png)

VLM 返回：`discard` / 置信度 `0.9`；`template_reading` = `N.W. :14kg G.W. :16.5kg Color:White`；`target_reading` = `N.W. :14kg G.W. :16.5kg Color:White`。原始响应：

```json
{"template_reading":"N.W. :14kg G.W. :16.5kg Color:White","target_reading":"N.W. :14kg G.W. :16.5kg Color:White","decision":"discard","confidence":0.9,"reason":"The visible printed content in both images is identical. The slight visual difference in the '4' character in the target image is due to font rendering or minor blur, but the character shape and content remain the same as the template."}
```

##### `600004075219_20260701-150453-773669_600004075219`（TP=0，FP=0，FN=1）

候选总数 6，VLM 请求 6，OCR 请求 0；panel 尺寸 900×586，标注空间 1017×662。

- **FN** `miss_0`：标注空间框 `[852, 180, 1002, 231]`，panel 空间框 `[754, 159, 887, 204]`；阶段归因 **VLM 判别器**。覆盖候选：`cand00` overlap=0.9375 → `discard/0.9`；`cand04` overlap=0.2334 → `discard/0.95`。

**候选 `cand00`** — focus box（panel）`[764, 158, 884, 206]`，crop 尺寸 240×96，匹配模板文本行 `5.20kW`，候选来源 `pdf_text_line`。

![600004075219_20260701-150453-773669_600004075219 cand00 模板/实拍 crop](../../results/evaluation/qwen38_nvfp4_latest15_noocr_20260830/vlm_candidate_crops/600004075219_20260701-150453-773669_600004075219/cand00_pair.png)

未拼接原图：[`cand00_template.png`](../../results/evaluation/qwen38_nvfp4_latest15_noocr_20260830/vlm_candidate_crops/600004075219_20260701-150453-773669_600004075219/cand00_template.png) ／ [`cand00_target.png`](../../results/evaluation/qwen38_nvfp4_latest15_noocr_20260830/vlm_candidate_crops/600004075219_20260701-150453-773669_600004075219/cand00_target.png)

VLM 返回：`discard` / 置信度 `0.9`；`template_reading` = `5.20kW`；`target_reading` = `5.20kW`。原始响应：

```json
{"template_reading":"5.20kW","target_reading":"5.20kW","decision":"discard","confidence":0.9,"reason":"The printed text in both images is identical, reading '5.20kW'. The differences in image quality, such as the blurriness and background texture in the target image, are due to photography and material artifacts rather than changes in the printed content."}
```

**候选 `cand04`** — focus box（panel）`[758, 193, 885, 252]`，crop 尺寸 207×96，匹配模板文本行 `850m³/h`，候选来源 `micro_text_candidate`。

![600004075219_20260701-150453-773669_600004075219 cand04 模板/实拍 crop](../../results/evaluation/qwen38_nvfp4_latest15_noocr_20260830/vlm_candidate_crops/600004075219_20260701-150453-773669_600004075219/cand04_pair.png)

未拼接原图：[`cand04_template.png`](../../results/evaluation/qwen38_nvfp4_latest15_noocr_20260830/vlm_candidate_crops/600004075219_20260701-150453-773669_600004075219/cand04_template.png) ／ [`cand04_target.png`](../../results/evaluation/qwen38_nvfp4_latest15_noocr_20260830/vlm_candidate_crops/600004075219_20260701-150453-773669_600004075219/cand04_target.png)

VLM 返回：`discard` / 置信度 `0.95`；`template_reading` = `850m³/h`；`target_reading` = `850m³/h`。原始响应：

```json
{"template_reading":"850m³/h", "target_reading":"850m³/h", "decision":"discard", "confidence":0.95, "reason":"The printed text in both images is identical. The target image shows the same characters '850m³/h' with the same spacing and formatting. Differences in background color and image sharpness are due to photography and material, not printed content defects."}
```

##### `600004085656_20260701-150539-671549_600004085656`（TP=0，FP=0，FN=1）

候选总数 3，VLM 请求 3，OCR 请求 0；panel 尺寸 662×1017，标注空间 662×1017。

- **FN** `miss_0`：标注空间框 `[519, 416, 647, 469]`，panel 空间框 `[519, 416, 647, 469]`；阶段归因 **VLM 判别器**。覆盖候选：`cand00` overlap=0.936 → `discard/0.9`。

**候选 `cand00`** — focus box（panel）`[520, 419, 662, 471]`，crop 尺寸 262×96，匹配模板文本行 `12600W`，候选来源 `pdf_text_line`。

![600004085656_20260701-150539-671549_600004085656 cand00 模板/实拍 crop](../../results/evaluation/qwen38_nvfp4_latest15_noocr_20260830/vlm_candidate_crops/600004085656_20260701-150539-671549_600004085656/cand00_pair.png)

未拼接原图：[`cand00_template.png`](../../results/evaluation/qwen38_nvfp4_latest15_noocr_20260830/vlm_candidate_crops/600004085656_20260701-150539-671549_600004085656/cand00_template.png) ／ [`cand00_target.png`](../../results/evaluation/qwen38_nvfp4_latest15_noocr_20260830/vlm_candidate_crops/600004085656_20260701-150539-671549_600004085656/cand00_target.png)

VLM 返回：`discard` / 置信度 `0.9`；`template_reading` = `12600W`；`target_reading` = `12600W`。原始响应：

```json
{"template_reading":"12600W", "target_reading":"12600W", "decision":"discard", "confidence":0.9, "reason":"The printed content in both images is identical. The first character is a '1' in both, and the subsequent characters '2600W' match perfectly. Differences are limited to image quality and lighting."}
```

#### real50

##### `600001076226_20260712-163622-315777_600001076226`（TP=0，FP=0，FN=1）

候选总数 4，VLM 请求 4，OCR 请求 0；panel 尺寸 900×265，标注空间 2009×592。

- **FN** `miss_0`：标注空间框 `[59, 43, 1996, 216]`，panel 空间框 `[26, 19, 894, 97]`；阶段归因 **VLM 判别器**。覆盖候选：`cand00` overlap=1.0 → `discard/0.9`；`cand01` overlap=0.859 → `discard/0.9`。

**候选 `cand00`** — focus box（panel）`[853, 29, 887, 97]`，crop 尺寸 48×96，匹配模板文本行 `)`，候选来源 `pdf_text_line`。

![600001076226_20260712-163622-315777_600001076226 cand00 模板/实拍 crop](../../results/evaluation/qwen38_nvfp4_real50_noocr_20260830/vlm_candidate_crops/600001076226_20260712-163622-315777_600001076226/cand00_pair.png)

未拼接原图：[`cand00_template.png`](../../results/evaluation/qwen38_nvfp4_real50_noocr_20260830/vlm_candidate_crops/600001076226_20260712-163622-315777_600001076226/cand00_template.png) ／ [`cand00_target.png`](../../results/evaluation/qwen38_nvfp4_real50_noocr_20260830/vlm_candidate_crops/600001076226_20260712-163622-315777_600001076226/cand00_target.png)

VLM 返回：`discard` / 置信度 `0.9`；`template_reading` = `)`；`target_reading` = `)`。原始响应：

```json
{"template_reading": ")", "target_reading": ")", "decision":"discard", "confidence":0.9, "reason":"Both images show a closing parenthesis. The target image has a vertical bar to the right which is likely a crop artifact or adjacent character, but the primary candidate content (the parenthesis) is identical in shape and orientation."}
```

**候选 `cand01`** — focus box（panel）`[0, 30, 900, 108]`，crop 尺寸 1108×96，匹配模板文本行 `GWH24AGD-K6DNA1C/I WIFI`，候选来源 `candidate_group`。

![600001076226_20260712-163622-315777_600001076226 cand01 模板/实拍 crop](../../results/evaluation/qwen38_nvfp4_real50_noocr_20260830/vlm_candidate_crops/600001076226_20260712-163622-315777_600001076226/cand01_pair.png)

未拼接原图：[`cand01_template.png`](../../results/evaluation/qwen38_nvfp4_real50_noocr_20260830/vlm_candidate_crops/600001076226_20260712-163622-315777_600001076226/cand01_template.png) ／ [`cand01_target.png`](../../results/evaluation/qwen38_nvfp4_real50_noocr_20260830/vlm_candidate_crops/600001076226_20260712-163622-315777_600001076226/cand01_target.png)

VLM 返回：`discard` / 置信度 `0.9`；`template_reading` = `GWH24AGD-K6DNA1C/I(WIFI)`；`target_reading` = `GWH24AGD-K6DNA1C/I(WIFI)`。原始响应：

```json
{"template_reading":"GWH24AGD-K6DNA1C/I(WIFI)","target_reading":"GWH24AGD-K6DNA1C/I(WIFI)","decision":"discard","confidence":0.9,"reason":"The printed text content is identical in both images. The differences are limited to font weight, color (white vs. grey), and photographic artifacts, which are non-content variations."}
```

##### `600001076226_20260712-163639-445160_600001076226`（TP=0，FP=0，FN=1）

候选总数 1，VLM 请求 1，OCR 请求 0；panel 尺寸 900×265，标注空间 2009×592。

- **FN** `miss_0`：标注空间框 `[36, 65, 1963, 209]`，panel 空间框 `[16, 29, 879, 94]`；阶段归因 **VLM 判别器**。覆盖候选：`cand00` overlap=0.9846 → `discard/0.9`。

**候选 `cand00`** — focus box（panel）`[0, 30, 900, 108]`，crop 尺寸 1108×96，匹配模板文本行 `GWH24AGD-K6DNA1C/I WIFI`，候选来源 `micro_text_candidate`。

![600001076226_20260712-163639-445160_600001076226 cand00 模板/实拍 crop](../../results/evaluation/qwen38_nvfp4_real50_noocr_20260830/vlm_candidate_crops/600001076226_20260712-163639-445160_600001076226/cand00_pair.png)

未拼接原图：[`cand00_template.png`](../../results/evaluation/qwen38_nvfp4_real50_noocr_20260830/vlm_candidate_crops/600001076226_20260712-163639-445160_600001076226/cand00_template.png) ／ [`cand00_target.png`](../../results/evaluation/qwen38_nvfp4_real50_noocr_20260830/vlm_candidate_crops/600001076226_20260712-163639-445160_600001076226/cand00_target.png)

VLM 返回：`discard` / 置信度 `0.9`；`template_reading` = `GWH24AGD-K6DNA1C/I(WIFI)`；`target_reading` = `GWH24AGD-K6DNA1C/I(WIFI)`。原始响应：

```json
{"template_reading": "GWH24AGD-K6DNA1C/I(WIFI)", "target_reading": "GWH24AGD-K6DNA1C/I(WIFI)", "decision": "discard", "confidence": 0.9, "reason": "The printed text content is identical in both images. The target image shows a vertical scratch or artifact on the opening parenthesis, but the underlying character shape and all other text remain consistent with the template."}
```

##### `600001076226_20260712-163822-306976_600001076226`（TP=0，FP=0，FN=1）

候选总数 2，VLM 请求 2，OCR 请求 0；panel 尺寸 900×265，标注空间 2009×592。

- **FN** `miss_0`：标注空间框 `[65, 62, 1963, 209]`，panel 空间框 `[29, 28, 879, 94]`；阶段归因 **VLM 判别器**。覆盖候选：`cand00` overlap=0.9697 → `discard/0.9`。

**候选 `cand00`** — focus box（panel）`[0, 30, 900, 108]`，crop 尺寸 1108×96，匹配模板文本行 `GWH24AGD-K6DNA1C/I WIFI`，候选来源 `micro_text_candidate_batch`。

![600001076226_20260712-163822-306976_600001076226 cand00 模板/实拍 crop](../../results/evaluation/qwen38_nvfp4_real50_noocr_20260830/vlm_candidate_crops/600001076226_20260712-163822-306976_600001076226/cand00_pair.png)

未拼接原图：[`cand00_template.png`](../../results/evaluation/qwen38_nvfp4_real50_noocr_20260830/vlm_candidate_crops/600001076226_20260712-163822-306976_600001076226/cand00_template.png) ／ [`cand00_target.png`](../../results/evaluation/qwen38_nvfp4_real50_noocr_20260830/vlm_candidate_crops/600001076226_20260712-163822-306976_600001076226/cand00_target.png)

VLM 返回：`discard` / 置信度 `0.9`；`template_reading` = `GWH24AGD-K6DNA1C/I(WIFI)`；`target_reading` = `GWH24AGD-K6DNA1C/I(WIFI)`。原始响应：

```json
{"template_reading":"GWH24AGD-K6DNA1C/I(WIFI)", "target_reading":"GWH24AGD-K6DNA1C/I(WIFI)", "decision":"discard", "confidence":0.9, "reason":"The printed text content is identical in both images. The differences in brightness, contrast, and slight blurriness in the target image are due to photographic conditions and do not represent changes in the printed characters or layout."}
```

##### `600004075219_20260712-164019-751975_600004075219`（TP=1，FP=1，FN=0）

候选总数 8，VLM 请求 8，OCR 请求 0；panel 尺寸 900×586，标注空间 1017×662。

- **FP** `pred_1`：标注空间框 `[389, 382, 553, 470]`，panel 空间框 `[344, 338, 489, 416]`；阶段归因 **VLM 判别器**。覆盖候选：`cand02` overlap=1.0 → `keep/0.95`；`cand01` overlap=0.2692 → `keep/0.95`；`cand06` overlap=0.0469 → `discard/0.95`。

**候选 `cand02`** — focus box（panel）`[332, 337, 501, 417]`，crop 尺寸 203×96，匹配模板文本行 `2026.01`，候选来源 `pdf_text_line`。

![600004075219_20260712-164019-751975_600004075219 cand02 模板/实拍 crop](../../results/evaluation/qwen38_nvfp4_real50_noocr_20260830/vlm_candidate_crops/600004075219_20260712-164019-751975_600004075219/cand02_pair.png)

未拼接原图：[`cand02_template.png`](../../results/evaluation/qwen38_nvfp4_real50_noocr_20260830/vlm_candidate_crops/600004075219_20260712-164019-751975_600004075219/cand02_template.png) ／ [`cand02_target.png`](../../results/evaluation/qwen38_nvfp4_real50_noocr_20260830/vlm_candidate_crops/600004075219_20260712-164019-751975_600004075219/cand02_target.png)

VLM 返回：`keep` / 置信度 `0.95`；`template_reading` = `2026.01`；`target_reading` = `2026.-01`。原始响应：

```json
{"template_reading":"2026.01","target_reading":"2026.-01","decision":"keep","confidence":0.95,"reason":"The target image clearly shows an extra hyphen between the period and the digits 01 (reading 2026.-01), whereas the template image shows no hyphen (reading 2026.01). This is a distinct printed content difference."}
```

**候选 `cand01`** — focus box（panel）`[0, 395, 578, 438]`，crop 尺寸 1290×96，匹配模板文本行 `ELECTRIC APPLIANCES,INC.OF ZHUHAI`，候选来源 `pdf_text_line`。

![600004075219_20260712-164019-751975_600004075219 cand01 模板/实拍 crop](../../results/evaluation/qwen38_nvfp4_real50_noocr_20260830/vlm_candidate_crops/600004075219_20260712-164019-751975_600004075219/cand01_pair.png)

未拼接原图：[`cand01_template.png`](../../results/evaluation/qwen38_nvfp4_real50_noocr_20260830/vlm_candidate_crops/600004075219_20260712-164019-751975_600004075219/cand01_template.png) ／ [`cand01_target.png`](../../results/evaluation/qwen38_nvfp4_real50_noocr_20260830/vlm_candidate_crops/600004075219_20260712-164019-751975_600004075219/cand01_target.png)

VLM 返回：`keep` / 置信度 `0.95`；`template_reading` = `GREE ELECTRIC APPLIANCES,INC.OF ZHUHAI`；`target_reading` = `GREE ELECTRIC APPLIANCE HAI`。原始响应：

```json
{"template_reading":"GREE ELECTRIC APPLIANCES,INC.OF ZHUHAI", "target_reading":"GREE ELECTRIC APPLIANCE HAI", "decision":"keep", "confidence":0.95, "reason":"The target image is missing the text segment ',INC.OF ZHU' which is clearly present in the template. The target shows a large gap between 'APPLIANCE' and 'HAI', indicating missing printed content."}
```

##### `600004075219_20260712-164128-331034_600004075219`（TP=0，FP=1，FN=1）

候选总数 11，VLM 请求 11，OCR 请求 0；panel 尺寸 900×586，标注空间 1017×662。

- **FP** `pred_0`：标注空间框 `[532, 282, 649, 344]`，panel 空间框 `[471, 250, 574, 304]`；阶段归因 **VLM 判别器**。覆盖候选：`cand01` overlap=0.9259 → `keep/0.9`；`cand09` overlap=0.0032 → `discard/0.95`。
- **FN** `miss_0`：标注空间框 `[384, 281, 542, 335]`，panel 空间框 `[340, 249, 480, 296]`；阶段归因 **VLM 判别器**。覆盖候选：`cand05` overlap=0.88 → `discard/0.9`；`cand01` overlap=0.1257 → `keep/0.9`。

**候选 `cand01`** — focus box（panel）`[463, 252, 582, 302]`，crop 尺寸 228×96，匹配模板文本行 `Weight`，候选来源 `pdf_text_line`。

![600004075219_20260712-164128-331034_600004075219 cand01 模板/实拍 crop](../../results/evaluation/qwen38_nvfp4_real50_noocr_20260830/vlm_candidate_crops/600004075219_20260712-164128-331034_600004075219/cand01_pair.png)

未拼接原图：[`cand01_template.png`](../../results/evaluation/qwen38_nvfp4_real50_noocr_20260830/vlm_candidate_crops/600004075219_20260712-164128-331034_600004075219/cand01_template.png) ／ [`cand01_target.png`](../../results/evaluation/qwen38_nvfp4_real50_noocr_20260830/vlm_candidate_crops/600004075219_20260712-164128-331034_600004075219/cand01_target.png)

VLM 返回：`keep` / 置信度 `0.9`；`template_reading` = `Weight`；`target_reading` = `3Weight`。原始响应：

```json
{"template_reading":"Weight", "target_reading":"3Weight", "decision":"keep", "confidence":0.9, "reason":"The target image shows an extra character '3' (or a similar glyph) immediately preceding the word 'Weight', which is not present in the template image."}
```

**候选 `cand05`** — focus box（panel）`[342, 252, 461, 302]`，crop 尺寸 228×96，匹配模板文本行 `4.60kW`，候选来源 `pdf_text_line`。

![600004075219_20260712-164128-331034_600004075219 cand05 模板/实拍 crop](../../results/evaluation/qwen38_nvfp4_real50_noocr_20260830/vlm_candidate_crops/600004075219_20260712-164128-331034_600004075219/cand05_pair.png)

未拼接原图：[`cand05_template.png`](../../results/evaluation/qwen38_nvfp4_real50_noocr_20260830/vlm_candidate_crops/600004075219_20260712-164128-331034_600004075219/cand05_template.png) ／ [`cand05_target.png`](../../results/evaluation/qwen38_nvfp4_real50_noocr_20260830/vlm_candidate_crops/600004075219_20260712-164128-331034_600004075219/cand05_target.png)

VLM 返回：`discard` / 置信度 `0.9`；`template_reading` = `4.60kW`；`target_reading` = `4.60kW`。原始响应：

```json
{"template_reading": "4.60kW", "target_reading": "4.60kW", "decision": "discard", "confidence": 0.9, "reason": "The printed text '4.60kW' is identical in both images. The vertical line visible on the right edge of the target image is a crop artifact or border, not a printed character, and the slight blur is due to image quality differences."}
```

##### `600004083205_20260712-164844-134966_600004083205`（TP=0，FP=0，FN=1）

候选总数 7，VLM 请求 7，OCR 请求 0；panel 尺寸 900×586，标注空间 1017×662。

- **FN** `miss_0`：标注空间框 `[803, 332, 987, 391]`，panel 空间框 `[711, 294, 873, 346]`；阶段归因 **VLM 判别器**。覆盖候选：`cand00` overlap=0.9231 → `discard/0.9`。

**候选 `cand00`** — focus box（panel）`[704, 298, 889, 348]`，crop 尺寸 355×96，匹配模板文本行 `12VDC, 2W`，候选来源 `pdf_text_line`。

![600004083205_20260712-164844-134966_600004083205 cand00 模板/实拍 crop](../../results/evaluation/qwen38_nvfp4_real50_noocr_20260830/vlm_candidate_crops/600004083205_20260712-164844-134966_600004083205/cand00_pair.png)

未拼接原图：[`cand00_template.png`](../../results/evaluation/qwen38_nvfp4_real50_noocr_20260830/vlm_candidate_crops/600004083205_20260712-164844-134966_600004083205/cand00_template.png) ／ [`cand00_target.png`](../../results/evaluation/qwen38_nvfp4_real50_noocr_20260830/vlm_candidate_crops/600004083205_20260712-164844-134966_600004083205/cand00_target.png)

VLM 返回：`discard` / 置信度 `0.9`；`template_reading` = `12VDC, 2W`；`target_reading` = `12VDC, 2W`。原始响应：

```json
{
"template_reading": "12VDC, 2W",
"target_reading": "12VDC, 2W",
"decision": "discard",
"confidence": 0.9,
"reason": "The text content in both images is identical. The target image is slightly blurrier and has a different background color, but the characters '12VDC, 2W' are clearly legible and match the template exactly with no visible substitutions or defects."
}
```

##### `600004083205_20260712-165005-026918_600004083205`（TP=0，FP=0，FN=1）

候选总数 6，VLM 请求 6，OCR 请求 0；panel 尺寸 900×586，标注空间 1017×662。

- **FN** `miss_0`：标注空间框 `[857, 180, 1017, 221]`，panel 空间框 `[758, 159, 900, 196]`；阶段归因 **VLM 判别器**。覆盖候选：`cand01` overlap=0.8451 → `discard/0.95`；`cand04` overlap=0.0731 → `discard/0.95`。

**候选 `cand01`** — focus box（panel）`[764, 153, 884, 202]`，crop 尺寸 235×96，匹配模板文本行 `5.60kW`，候选来源 `micro_text_candidate_batch`。

![600004083205_20260712-165005-026918_600004083205 cand01 模板/实拍 crop](../../results/evaluation/qwen38_nvfp4_real50_noocr_20260830/vlm_candidate_crops/600004083205_20260712-165005-026918_600004083205/cand01_pair.png)

未拼接原图：[`cand01_template.png`](../../results/evaluation/qwen38_nvfp4_real50_noocr_20260830/vlm_candidate_crops/600004083205_20260712-165005-026918_600004083205/cand01_template.png) ／ [`cand01_target.png`](../../results/evaluation/qwen38_nvfp4_real50_noocr_20260830/vlm_candidate_crops/600004083205_20260712-165005-026918_600004083205/cand01_target.png)

VLM 返回：`discard` / 置信度 `0.95`；`template_reading` = `5.60kW`；`target_reading` = `5.60kW`。原始响应：

```json
{"template_reading": "5.60kW", "target_reading": "5.60kW", "decision": "discard", "confidence": 0.95, "reason": "The printed text '5.60kW' is identical in both images. The target image shows a partial crop on the right edge, but the visible characters match the template exactly with no substitutions, missing strokes, or extra marks."}
```

##### `600004083205_20260712-165134-447057_600004083205`（TP=0，FP=0，FN=1）

候选总数 6，VLM 请求 6，OCR 请求 0；panel 尺寸 900×586，标注空间 1017×662。

- **FN** `miss_0`：标注空间框 `[394, 274, 534, 329]`，panel 空间框 `[349, 242, 473, 291]`；阶段归因 **VLM 判别器**。覆盖候选：`cand00` overlap=0.8542 → `discard/0.9`。

**候选 `cand00`** — focus box（panel）`[352, 250, 472, 298]`，crop 尺寸 240×96，匹配模板文本行 `5.30kW`，候选来源 `pdf_text_line`。

![600004083205_20260712-165134-447057_600004083205 cand00 模板/实拍 crop](../../results/evaluation/qwen38_nvfp4_real50_noocr_20260830/vlm_candidate_crops/600004083205_20260712-165134-447057_600004083205/cand00_pair.png)

未拼接原图：[`cand00_template.png`](../../results/evaluation/qwen38_nvfp4_real50_noocr_20260830/vlm_candidate_crops/600004083205_20260712-165134-447057_600004083205/cand00_template.png) ／ [`cand00_target.png`](../../results/evaluation/qwen38_nvfp4_real50_noocr_20260830/vlm_candidate_crops/600004083205_20260712-165134-447057_600004083205/cand00_target.png)

VLM 返回：`discard` / 置信度 `0.9`；`template_reading` = `5.30kW`；`target_reading` = `5.30kW`。原始响应：

```json
{"template_reading": "5.30kW", "target_reading": "5.30kW", "decision": "discard", "confidence": 0.9, "reason": "The text content is identical in both images. The apparent difference in the zero (0) is due to font rendering and image resolution; the character shape in the target is consistent with a zero in the context of the number 5.30."}
```
