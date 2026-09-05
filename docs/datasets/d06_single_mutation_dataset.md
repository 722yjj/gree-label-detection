# D06 单变更标签数据集

> 数据生成项目：`/home/jnu/projects/dataset`  
> 主项目：`/home/jnu/projects/gree-label-detection`  
> 推荐版本：`D06_single_mutation_1k_v2_boxes`

## 1. 目标与边界

D06 用于固定五种企业标签模板下的细粒度内容差异研究。每个生成样本严格只包含一次文字改动，便于定位监督、错误归因和按变异类型统计。

需要特别区分：`A`～`E` 是五个模板域的短别名，不是模型最终输出的五个类别。模型的业务输出仍是二分类：

- `different / keep`：模板与目标存在真实内容差异；
- `same / discard`：只有拍摄、印刷、噪声或配准差异，内容相同。

当前 D06 的 1000 个样本全部是 `different` 正样本。它可以用于正样本生成、定位监督和变异覆盖分析，但不能单独训练二分类模型；正式训练前还必须加入 clean negative 和 hard negative。

## 2. 数据版本

### 2.1 原始单变更版本

路径：

```text
/home/jnu/projects/dataset/outputs/datasets/D06_single_mutation_1k
```

该版本完成以下工作：

- 五个模板分别使用 `A`～`E` 命名；
- 每个模板保留 200 个唯一单变更样本；
- 唯一性由“模板、位置、原文、改后文本、策略”共同确定；
- 每模板固定切分为 `140 train / 30 val / 30 test`；
- 保留原始模板编号映射和逐样本变异元数据。

### 2.2 双框推荐版本

路径：

```text
/home/jnu/projects/dataset/outputs/datasets/D06_single_mutation_1k_v2_boxes
```

该版本在原始样本内容不变的基础上重新计算两种框：

```json
{
  "changed_bbox": "实际新增、删除或替换字符的字符级范围",
  "value_bbox": "变化字符所属的完整业务值范围"
}
```

预览中的颜色约定：

- 红色实线：`changed_bbox`；
- 黄色虚线：`value_bbox`。

旧的 PDF span 框以 `legacy_span_bbox` 保留，只用于追溯，不应再作为训练定位框。

## 3. 模板映射

| 别名 | 原模板 ID |
| --- | --- |
| A | `600001076226` |
| B | `600004075219` |
| C | `600004078454` |
| D | `600004083205` |
| E | `600004085656` |

映射文件：

```text
D06_single_mutation_1k_v2_boxes/template_mapping.json
```

任何导出、训练和评估清单都必须同时保留 `template_alias` 和 `template_id`，不能只保留字母别名。

## 4. 数量与切分

| 项目 | 数量 |
| --- | ---: |
| 模板 | 5 |
| 每模板样本 | 200 |
| 总样本 | 1000 |
| Train | 700，每模板 140 |
| Validation | 150，每模板 30 |
| Test | 150，每模板 30 |
| 每样本改动数 | 1 |

变异策略分布：

| 策略 | 数量 |
| --- | ---: |
| `confusable` 易混字符 | 247 |
| `substitute` 普通替换 | 166 |
| `insert` 插入 | 139 |
| `delete` 删除 | 115 |
| `duplicate` 重复 | 111 |
| `swap` 交换 | 109 |
| `case` 大小写 | 105 |
| `superscript` 上标 | 8 |

上标唯一组合只有 8 条，这是源模板中可变上标位置有限导致的。D06 不应冒充上标均衡数据；后续训练仍需使用专门的上标困难集补充。

## 5. 目录结构

```text
D06_single_mutation_1k_v2_boxes/
├── templates/
│   ├── A.pdf ... E.pdf              # 裁剪后的标准标签模板
├── source_templates/
│   ├── A.pdf ... E.pdf              # 原始完整模板 PDF
├── samples/
│   ├── A/A_0001.pdf ... A_0200.pdf
│   ├── B/B_0001.pdf ... B_0200.pdf
│   └── ...
├── previews/
│   ├── A/A_0001.png ...             # 红/黄双框审计图
│   └── ...
├── metadata/
│   ├── A/A_0001.json ...            # 单样本元数据
│   └── ...
├── splits/
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── manifest.jsonl                    # 1000 条统一清单
├── summary.json
└── template_mapping.json
```

双框版本的样本 PDF 通过硬链接复用原始 D06 文件。目录逻辑大小约 1.1 GB，实际新增磁盘占用约 53 MB，主要来自新版预览和元数据。

## 6. 单条记录结构

`manifest.jsonl`、`splits/*.jsonl` 和 `metadata/<alias>/<sample_id>.json` 使用同一记录结构：

```json
{
  "schema_version": "d06_single_mutation_v2_dual_bbox",
  "sample_id": "C_0033",
  "source_sample_id": "sample_0033",
  "template_alias": "C",
  "template_id": "600004078454",
  "split": "train",
  "label": 1,
  "label_name": "different",
  "mutation_count": 1,
  "mutation_strategy": "duplicate",
  "original_text": "590",
  "mutated_text": "5590",
  "changed_bbox": [207.40, 57.26, 211.56, 67.97],
  "value_bbox": [207.40, 56.22, 236.20, 68.22],
  "legacy_span_bbox": [207.40, 58.55, 219.88, 68.22],
  "target_pdf": ".../samples/C/C_0033.pdf",
  "annotated_preview": ".../previews/C/C_0033.png",
  "template_pdf": ".../templates/C.pdf",
  "source_template_pdf": ".../source_templates/C.pdf"
}
```

字段用途：

| 字段 | 用途 |
| --- | --- |
| `sample_id` | 对外稳定样本编号 |
| `template_alias` | 简短模板域标识 |
| `template_id` | 与生产 PDF、实拍图和历史实验关联的原始编号 |
| `split` | 固定 train/val/test 切分 |
| `mutation_strategy` | 变异类型统计和分层评估 |
| `original_text` / `mutated_text` | 内容变化审计 |
| `changed_bbox` | 精确字符定位监督、定位指标和错误审计 |
| `value_bbox` | 完整业务值裁剪、上下文输入和最终候选范围 |
| `legacy_span_bbox` | 旧算法追溯，不用于新训练 |

所有 1000 条记录已经验证满足：

```text
changed_bbox 非空
value_bbox 非空
changed_bbox 包含于 value_bbox
```

## 7. A 类样本

模板 A，对应原模板 `600001076226`：

![A类模板](../report_assets/d06_dataset/A/template.png)

### A_0006：易混字符

`I → 1`。红框只覆盖变化字符，黄色框覆盖完整型号值。

![A_0006](../report_assets/d06_dataset/A/A_0006.png)

### A_0031：插入

`GWH24AGD-K6DNA1C/ → GWH24AGD-K6DNA15C/`。

![A_0031](../report_assets/d06_dataset/A/A_0031.png)

### A_0147：删除

`Connection Pipes :1/4"/1/2" → Connection Pipes :1/"/1/2"`。

![A_0147](../report_assets/d06_dataset/A/A_0147.png)

## 8. B 类样本

模板 B，对应原模板 `600004075219`：

![B类模板](../report_assets/d06_dataset/B/template.png)

### B_0027：上标变化

`3 → 2`。`changed_bbox`只覆盖上标，`value_bbox`同时包含数值、单位和上标。

![B_0027](../report_assets/d06_dataset/B/B_0027.png)

### B_0170：易混字符

`220-240V~ → 220-24DV~`。

![B_0170](../report_assets/d06_dataset/B/B_0170.png)

### B_0192：删除

`5.20kW → 5.20W`。

![B_0192](../report_assets/d06_dataset/B/B_0192.png)

## 9. C 类样本

模板 C，对应原模板 `600004078454`：

![C类模板](../report_assets/d06_dataset/C/template.png)

### C_0030：上标内插入

`3 → 35`。红框覆盖新增字符，黄色框覆盖完整风量值。

![C_0030](../report_assets/d06_dataset/C/C_0030.png)

### C_0033：数字重复

`590 → 5590`。黄色框将数字、`m`、上标和`/h`合并为一个业务值。

![C_0033](../report_assets/d06_dataset/C/C_0033.png)

### C_0142：字符交换

`Weight → Weihgt`。

![C_0142](../report_assets/d06_dataset/C/C_0142.png)

## 10. D 类样本

模板 D，对应原模板 `600004083205`：

![D类模板](../report_assets/d06_dataset/D/template.png)

### D_0074：上标变化

`3 → 2`。

![D_0074](../report_assets/d06_dataset/D/D_0074.png)

### D_0024：易混字符

`12VDC, 2W → I2VDC, 2W`。

![D_0024](../report_assets/d06_dataset/D/D_0024.png)

### D_0198：插入

`5.30kW → 5.307kW`。

![D_0198](../report_assets/d06_dataset/D/D_0198.png)

## 11. E 类样本

模板 E，对应原模板 `600004085656`：

![E类模板](../report_assets/d06_dataset/E/template.png)

### E_0042：上标变化

`3 → 2`。

![E_0042](../report_assets/d06_dataset/E/E_0042.png)

### E_0116：易混字符

`220-240V~ → 220-2A0V~`。

![E_0116](../report_assets/d06_dataset/E/E_0116.png)

### E_0153：重复

`53.5kg → 53.5kgg`。

![E_0153](../report_assets/d06_dataset/E/E_0153.png)

## 12. 用于模型训练前还缺什么

D06 v2 当前是正样本源数据，不是可直接训练 M11 或后续 Pair Interaction Adapter 的最终 pair 数据集。进入训练前必须完成：

1. 根据 `value_bbox` 构造模板/目标 value crop；
2. 根据 `changed_bbox` 生成精确定位 mask；
3. 生成内容不变的 print-scan、模糊、光照、噪声和轻微配准负样本；
4. 从主流程误触发候选中加入 hard negative；
5. 原样本及其所有增强副本保持在同一个 split；
6. 分别报告分类指标、定位指标和完整主流程最终框指标。

建议模型方案见 [`pair_interaction_adapter_model_proposal.md`](../research/pair_interaction_adapter_model_proposal.md)。

## 13. 复现命令

生成原始单变更数据：

```bash
cd /home/jnu/projects/dataset
.venv/bin/python scripts/build_d06_single_mutation_dataset.py \
  --source-dir inputs/pdfs \
  --output-dir outputs/datasets/D06_single_mutation_1k \
  --samples-per-template 200 \
  --candidate-multiplier 3 \
  --train-per-template 140 \
  --val-per-template 30 \
  --test-per-template 30 \
  --seed 20260727
```

生成双框版本：

```bash
cd /home/jnu/projects/dataset
.venv/bin/python scripts/build_d06_dual_bbox_dataset.py \
  --input-dir outputs/datasets/D06_single_mutation_1k \
  --output-dir outputs/datasets/D06_single_mutation_1k_v2_boxes
```
