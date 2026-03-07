# 格力标签检测系统

基于 OCR + VLM (视觉大模型) 的空调标签差异检测工具，可自动对比模板 PDF 与实拍标签照片的文字与图形差异。

## 核心流程

```
模板 PDF + 实拍图 → 预处理 → unified_detection.py → 文字比对 + 图形比对 → 结果输出
```

## 模块结构

```
project/
├── unified_detection.py      # 🔹 唯一主入口
├── schemas.py                # 公共数据模型 (AirConditionerLabel)
├── config.py                 # 统一配置
├── extract_lable.py          # PDF 模板图片提取
├── preprocessing/            # 图像预处理
│   ├── border.py             #   边框检测与裁剪
│   ├── perspective.py        #   透视矫正
│   └── pipeline.py           #   预处理流水线
├── services/                 # 公共服务
│   ├── ocr_service.py        #   PaddleOCR 封装
│   └── vlm_service.py        #   Qwen3-VL (Ollama) 封装
├── layout_region_comparison.py  # PP-DocLayoutV3 区域检测与比对
├── vlm_comparator.py         # VLM 比对兼容接口
├── mask_text_regions.py      # 文字区域掩盖工具
└── (实验脚本，标记为 experimental/deprecated)
```

## 环境依赖

### Python 依赖

```bash
pip install -r requirements.txt
```

### 模型依赖

| 模型 | 用途 | 安装方式 |
|------|------|---------|
| Qwen3-VL:8b | VLM 图形比对 + 结构化提取 | `ollama pull qwen3-vl:8b` |
| PaddleOCR | 文字识别 | `pip install paddleocr` 自动下载 |
| PP-DocLayoutV3 | 布局区域检测 | `pip install paddlex` 自动下载 |

### 启动 Ollama

```bash
ollama serve  # 启动服务
ollama pull qwen3-vl:8b  # 下载模型
```

## 运行方式

```bash
python unified_detection.py --pdf 600004075219-01.pdf --target test.jpg
```

### 参数说明

- `--pdf`：模板 PDF 文件路径（默认：`600004075219-01.pdf`）
- `--target`：实拍标签图片路径（默认：`test.jpg`）

## 输出说明

运行后在 `results/unified/` 目录下生成：

| 文件 | 说明 |
|------|------|
| `final_result.json` | 完整检测结果（含文字+图形比对） |
| `text_comparison.xlsx` | 文字字段对比表 |
| `visualization_diff.jpg` | 差异可视化标注图 |
| `template_preprocessed.jpg` | 预处理后的模板图 |
| `target_preprocessed.jpg` | 预处理后的实拍图 |
| `graphic_comparison/` | 图形区域对比详情 |

## 配置

可通过环境变量或修改 `config.py` 调整：

- `OLLAMA_API_BASE`：Ollama 服务地址（默认 `http://localhost:11434`）
- `OLLAMA_MODEL`：VLM 模型名称（默认 `qwen3-vl:8b`）
- `VLM_TIMEOUT`：VLM 请求超时秒数（默认 120）
- `LLM_MAX_RETRIES`：LLM 提取重试次数（默认 3）

## 已知问题

- VLM 输出不稳定，偶尔返回非 JSON 格式，已有关键词兜底和重试机制
- 图形比对强依赖 VLM，VLM 解析失败时缺少传统方法回退
- 实拍图片透视矫正依赖四角点检测，复杂背景下可能失败
