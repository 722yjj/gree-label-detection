# OCR 准确率、显存与延迟记录

记录日期：2026-04-27
当前分支：`experiment/graphic-diff-mask`
样本图：`samples/images/produce/type1/600004075219_1.jpg`

## 背景

当前目标不是让模型“理解标签大意”，而是把标签上的真实印刷文字尽量准确、可追溯地提取出来，并减少 OCR / VLM 把模糊文字自动纠正成合理值的幻觉。

因此 OCR 方案要同时看三件事：

- 字段级准确率：型号、条码、日期、电压、频率、重量、单位等是否读对。
- 资源占用：显存、内存、是否能和 layout / VLM 同时驻留。
- 延迟：冷启动和常驻进程下的单图推理时间。

## 当前是否已有指标

仓库中没有找到正式记录的 OCR 显存/延迟基准。已有文档只描述了当前 OCR 流程和配置，没有固定样本下的耗时、显存、CPU/GPU 对比。

## 当前环境兼容问题

修复前，直接在当前 `.venv` 中初始化 `PaddleOCR` 会失败：

```text
ModuleNotFoundError: No module named 'langchain.docstore'
```

原因是当前环境中：

- `paddleocr==3.4.0`
- `paddlex==3.4.0`
- `langchain==1.2.0`
- `langchain-classic==1.0.0`
- `langchain-community==0.4.1`

`paddlex` 仍导入旧路径 `langchain.docstore.document` 和 `langchain.text_splitter`，但 LangChain 1.x 已拆分路径。

当前项目已通过 `label_detection.core.langchain_compat` 在导入 PaddleOCR / PaddleX 前安装窄范围兼容别名，覆盖：

- `langchain.docstore.document.Document`
- `langchain.text_splitter.RecursiveCharacterTextSplitter`

该修复不修改 `.venv` 或 site-packages，只在项目 Python 进程内生效。

修复后已验证：

```text
from label_detection.services.ocr_service import get_ocr_with_boxes
text, boxes = get_ocr_with_boxes("samples/images/produce/type1/600004075219_1.jpg")

elapsed_s=3.627
boxes=44
first_lines=['GREE', 'SPLITAIR CONDITIONER', 'INDOOR', 'LINN', 'Model', 'GWH18AAD-K6DNA2E/I', 'Rated', 'Voltage']
```

## 单样本基准

测试方式：

- 调用 `label_detection.services.ocr_service.get_ocr_with_boxes(...)`。
- 同一进程连续跑两次同一张图。
- 第 1 次包含 OCR 引擎初始化和首次推理。
- 第 2 次代表 OCR 引擎已常驻后的单图推理。
- GPU 显存通过 `nvidia-smi --query-compute-apps=pid,used_memory` 记录当前 Python 进程占用。

硬件：

```text
GPU: NVIDIA GeForce RTX 4090
Total VRAM: 24564 MB
Baseline VRAM before OCR process: 451 MB
```

### PaddleOCR GPU

配置：

```text
OCR_DEVICE=gpu:0
PaddleOCR models:
- PP-LCNet_x1_0_doc_ori
- UVDoc
- PP-LCNet_x1_0_textline_ori
- PP-OCRv5_server_det
- PP-OCRv5_server_rec
```

结果：

| 指标 | 第 1 次：初始化 + 推理 | 第 2 次：常驻推理 |
|---|---:|---:|
| 耗时 | 3.467 s | 1.244 s |
| OCR boxes | 44 | 44 |
| 文本行数 | 44 | 44 |
| 进程 RSS | 52.0 MB -> 1632.4 MB | 1632.4 MB -> 1580.3 MB |
| Python 进程 GPU 显存 | 0 MB -> 7234 MB | 7234 MB -> 7234 MB |
| 全局 GPU 显存 | 451 MB -> 7691 MB | 7691 MB -> 7691 MB |

前 12 行 OCR 文本：

```text
GREE
SPLITAIR CONDITIONER
INDOOR
LINN
Model
GWH18AAD-K6DNA2E/I
Rated
Voltage
220-240V~
Heating
Capacity
5.20kW
```

观察：

- GPU 常驻后速度可接受，单张完整实拍图约 1.2s。
- 显存占用很高，单个 PaddleOCR 进程约 7.2GB。
- 识别仍有错误，例如 `UNIT` 被识别成 `LINN`。

### PaddleOCR CPU

配置：

```text
OCR_DEVICE=cpu
```

结果：

| 指标 | 第 1 次：初始化 + 推理 | 第 2 次：常驻推理 |
|---|---:|---:|
| 耗时 | 13.063 s | 10.143 s |
| OCR boxes | 44 | 44 |
| 文本行数 | 44 | 44 |
| 进程 RSS | 52.0 MB -> 2339.2 MB | 2339.2 MB -> 2525.1 MB |
| Python 进程 GPU 显存 | 0 MB -> 0 MB | 0 MB -> 0 MB |
| 全局 GPU 显存 | 451 MB -> 453 MB | 453 MB -> 453 MB |

前 12 行 OCR 文本：

```text
GREE
SPLITAIR CONDITIONER
INDOOR
UNIT
Model
GWH18AAD-K6DNA2E/I
Cerr
Voltage
220-240V~
Heating
Capacity
5.20kW
```

观察：

- CPU 几乎不占 GPU，但单张完整图常驻推理也需要约 10s，不适合作为在线主路径。
- CPU 结果和 GPU 结果不完全一致；CPU 读对了 `UNIT`，但把 `Rated` 读成了 `Cerr`。
- CPU 内存占用也不低，常驻约 2.3GB 到 2.5GB。

## 多 OCR 的资源影响

如果引入多个 OCR，要区分三种方式。

### 方式一：多个 GPU OCR 全量并行

不建议作为默认路径。

按本次 PaddleOCR GPU 基准估算，一个完整 OCR 引擎约占 7.2GB 显存。RTX 4090 约 24GB，两个类似规模 OCR 引擎可能接近 14GB 到 16GB，再叠加 layout 模型、VLM、图形比对和缓存，很容易出现显存紧张。

延迟方面，如果并行跑多个 OCR，单图墙钟时间可能接近最慢 OCR，但显存压力最大；如果串行跑，显存可控一些，但延迟会叠加。

### 方式二：一个主 OCR + 一个 CPU/轻量 OCR 兜底

可作为实验方向，但不适合整图默认启用。

CPU PaddleOCR 单图约 10s，太慢；如果只对少数低置信字段 crop 运行，延迟可能可接受。更合理的是找轻量 OCR 或 ONNX OCR 做字段级补充，而不是再跑一次完整图片 OCR。

### 方式三：一个主 OCR + 多尺度/局部重识别

最推荐优先做。

同一个 OCR 引擎已经常驻时，对字段 crop 做放大、二值化、锐化、多尺度识别，通常比引入另一个完整 OCR 模型更省资源。这个方案不会额外常驻一套大模型，主要增加的是少量局部 crop 推理时间。

## 当前建议

### 1. 先修复当前 OCR 环境兼容问题

当前 `.venv` 直接初始化 PaddleOCR 会因为 LangChain 路径变更失败。后续要么：

- 固定与 PaddleX 3.4.0 兼容的 LangChain 版本；
- 或在项目侧增加兼容处理；
- 或确认 PaddleX / PaddleOCR 是否有新版本修复该导入路径。

这个问题不解决，端到端 OCR 指标无法稳定复现。

### 2. 建立字段级 OCR 评测集

至少先人工标注 20 张实拍图，每张记录关键字段：

- `model_number`
- `voltage`
- `frequency`
- `heating_capacity`
- `cooling_capacity`
- `air_volume`
- `weight`
- `noise`
- `mfg_date`
- `barcode`
- 关键字段标签名，例如 `Rated Voltage`、`Manufactured Date`

评估时不要只看整段文本相似度，要看字段级 exact match、数字字段 exact match、单位是否保留、OCR 框是否能定位。

### 3. 默认只保留一个主 OCR

在当前 4090 环境下，PaddleOCR GPU 单进程约 7.2GB 显存。默认再常驻第二个 GPU OCR 不划算。

短期建议：

```text
主路径：PaddleOCR GPU
兜底：字段级 crop 多尺度重识别
低置信/冲突：VLM 或第二 OCR 只处理局部 crop
```

### 4. 把模板位置约束用起来

项目有模板，这是比通用 OCR 更强的先验。建议把文字识别从“整图自由识别”逐步改成：

```text
整图 OCR
-> 模板字段框映射到实拍图
-> 字段 crop 放大
-> 字段级 OCR / 多尺度 OCR
-> 正则格式校验
-> 低置信字段进入复核
```

这样比直接引入多个完整 OCR 更稳，也更省资源。

### 5. VLM 只能补洞，不能当最终事实源

VLM 适合用于：

- OCR 漏字段时读局部 crop；
- 判断两个候选值哪个更像图上文字；
- 输出低置信字段的辅助建议。

VLM 不适合直接输出最终完整标签 JSON。最终值应优先来自 OCR 框、字段 crop 和格式校验。特别是型号、条码、日期、频率、电压等字段，不能让 VLM 自动纠正常识值。

### 6. 模型训练暂不作为第一优先级

目前只有三张模板，不适合直接训练 OCR 模型。更现实的做法是：

- 用模板生成合成字段样本；
- 训练或调参字段级后处理/纠错；
- 等真实拍照样本积累到几十到几百张后，再考虑 OCR 微调。

## 下一步可执行计划

1. 增加一个轻量 OCR benchmark 脚本，固定输出冷启动耗时、常驻耗时、RSS、GPU 显存、字段级识别结果。
2. 做字段级标注表，先用 20 张真实图评估现有 PaddleOCR。
3. 实现字段 crop 重识别，优先覆盖型号、条码、电压、频率、日期。
4. 再决定是否引入第二 OCR；第二 OCR 默认只跑低置信局部 crop，不跑整图。
