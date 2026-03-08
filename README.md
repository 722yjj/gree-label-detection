# 格力标签检测系统

基于 OCR + VLM 的空调标签差异检测工具，用于对比模板 PDF 与实拍标签图片在文字和图形上的差异。

## 项目入口

当前根入口文件为 `main.py`：

```bash
python main.py --pdf samples/pdfs/600004075219-01.pdf --target samples/images/produce/type1/600004075219_1.jpg
```

实际主流程位于 `label_detection/workflows/unified.py`，`main.py` 只是根目录 CLI 包装器。

## 核心流程

```text
模板 PDF + 实拍图
  -> 模板提取 / 图像预处理
  -> OCR 文字识别
  -> LLM 结构化提取
  -> 布局区域检测与图形对比
  -> 汇总输出 JSON / Excel / 可视化结果
```

## 当前预处理说明

目标图预处理采用两级策略：

1. 优先尝试透视矫正，先定位标签主体，再做透视变换。
2. 如果透视矫正失败，回退到黑框检测和边缘检测裁剪。

透视矫正当前已移除“浅色区域检测”分支，避免把整图背景、黄底纸、反光区域误判为标签主体。

## 项目结构

```text
project/
├── main.py                       # 根入口文件
├── AGENT_CHANGELOG.md            # agent 关键改动追溯记录
├── README.md
├── requirements.txt
├── label_detection/              # 业务代码
│   ├── core/                     # 配置与兼容层
│   ├── extraction/               # PDF 模板提取
│   ├── matching/                 # OCR / 布局 / 图形匹配
│   ├── models/                   # 数据模型导出
│   ├── preprocessing/            # 图像预处理
│   ├── services/                 # OCR / VLM 服务封装
│   ├── workflows/                # 主流程编排
│   └── schema.py                 # 标签结构化字段定义
├── samples/
│   ├── images/                   # 示例模板图 / 实拍图
│   ├── pdfs/                     # 示例模板 PDF
│   ├── reference/                # 参考提取结果
│   └── spreadsheets/             # 示例表格
├── tests/                        # 测试目录
└── results/                      # 运行输出目录
```

## 环境依赖

安装 Python 依赖：

```bash
pip install -r requirements.txt
```

模型和服务依赖：

| 组件 | 用途 | 说明 |
|------|------|------|
| Ollama + `qwen3-vl:8b` | VLM 图形比对、结构化提取 | 需要本地启动 Ollama 服务 |
| PaddleOCR | 文字识别 | 首次运行会下载模型 |
| PP-DocLayoutV3 / PaddleX | 布局区域检测 | 首次运行会下载模型 |

启动 Ollama：

```bash
ollama serve
ollama pull qwen3-vl:8b
```

## 运行参数

`main.py` 支持以下参数：

- `--pdf`：模板 PDF 路径
- `--target`：实拍图片路径
- `--output-dir`：结果输出目录

默认值定义在 `label_detection/core/config.py` 中。

## 示例输入

仓库内可直接使用的样例包括：

- 模板 PDF：`samples/pdfs/600004075219-01.pdf`
- 实拍图片：`samples/images/produce/type1/600004075219_1.jpg`
- 参考图片与文本：`samples/reference/`

## 输出结果

默认输出目录为 `results/unified/`，常见产物包括：

| 文件 / 目录 | 说明 |
|-------------|------|
| `final_result.json` | 完整检测结果 |
| `text_comparison.xlsx` | 文字字段对比表 |
| `visualization_diff.jpg` | 差异可视化结果 |
| `template_preprocessed.jpg` | 模板预处理结果 |
| `target_preprocessed.jpg` | 实拍图预处理结果 |
| `target_corners_detected.jpg` | 透视角点调试图 |
| `target_perspective_corrected.jpg` | 透视矫正结果 |
| `graphic_comparison/` | 图形区域对比结果 |

## 关键模块

- `label_detection/workflows/unified.py`：整合主流程
- `label_detection/preprocessing/pipeline.py`：模板图 / 实拍图预处理入口
- `label_detection/preprocessing/perspective.py`：标签主体定位与透视矫正
- `label_detection/matching/layout.py`：图形区域检测与匹配
- `label_detection/services/ocr_service.py`：OCR 服务封装
- `label_detection/services/vlm_service.py`：VLM 服务封装

## 变更追溯

项目中的关键改动记录在 `AGENT_CHANGELOG.md`。  
如果后续继续由不同的 code agent 修改，建议按统一模板追加记录，至少包含：

- 修改背景
- 修改文件
- 行为变化
- 验证情况
- 风险和待验证项

## 当前已知风险

- 当前命令环境未完成一次稳定的端到端自动验证，部分结论仍依赖静态检查和已有输出目录结果。
- 预处理 fallback 仍以最大轮廓裁剪为主，复杂背景下仍可能需要进一步增强。
- 图形比对仍较依赖 VLM 能力，模型输出不稳定时可能需要人工复核。
