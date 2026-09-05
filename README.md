# 标签检测系统

基于 OCR + VLM 的空调标签差异检测工具，用于对比模板文件与实拍标签图片在文字和图形上的差异。模板文件支持 PDF 和图片输入。

## 项目入口

当前根入口文件为 `main.py`：

```bash
.venv/bin/python main.py --template samples/pdfs/600004075219-01.pdf --target samples/images/produce/type1/600004075219_1.jpg
```

实际主流程位于 `label_detection/workflows/unified.py`，`main.py` 只是根目录 CLI 包装器。

桌面端统一启动命令为：

```bash
.venv/bin/python -m desktop_app.main
```

当前仓库约定：

- 如果仓库根目录已经存在 `.venv`，后续命令统一使用 `.venv/bin/python ...`
- 不再把 `uv run python ...` 作为默认启动写法，避免新会话里出现两套命令混用

## 核心流程

```text
模板文件(PDF/图片) + 实拍图
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
├── pyproject.toml
├── uv.lock
├── requirements.txt
├── desktop_app/                  # PySide6 桌面端骨架
├── docs/                         # 开发方案与补充文档
├── label_detection/              # 业务代码
│   ├── core/                     # 配置与兼容层
│   ├── extraction/               # 模板输入解析 / PDF 提取
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

推荐使用 `uv` 管理本地开发环境：

```bash
uv venv
uv sync --extra desktop
```

这套 `uv` 配置当前固定在 Python 3.12。完成同步后，推荐统一使用项目虚拟环境里的 Python：

```bash
.venv/bin/python -m desktop_app.main
```

如果只运行 CLI，也统一使用：

```bash
.venv/bin/python main.py --template samples/pdfs/600004075219-01.pdf --target samples/images/produce/type1/600004075219_1.jpg
```

兼容旧方式时，仍可使用：

```bash
pip install -r requirements.txt
```

当前服务器侧原有 `conda run -n ocr ...` 流程暂不改动；新引入的 `uv` 主要用于本地新版本开发。

模型和服务依赖：

| 组件 | 用途 | 说明 |
|------|------|------|
| Ollama + `qwen3.5:9b` | VLM 图形比对、结构化提取 | 默认后端，需要本地启动 Ollama 服务 |
| OpenAI-compatible 服务 / vLLM | VLM 图形比对、结构化提取 | 设置 `LLM_PROVIDER=vllm` 或 `LLM_PROVIDER=openai-compatible` 后使用 |
| PaddleOCR | 文字识别 | 首次运行会下载模型 |
| PP-DocLayoutV3 / PaddleX | 布局区域检测 | 首次运行会下载模型 |

启动 Ollama：

```bash
ollama serve
ollama pull qwen3.5:9b
```

使用 vLLM / OpenAI-compatible 服务示例：

```bash
scripts/start_vllm.sh

export LLM_PROVIDER=vllm
export VLLM_API_BASE=http://127.0.0.1:8000/v1
export VLLM_MODEL_PATH=/home/jnu/models/Qwen3.6-27B-int4-AutoRound
export VLLM_SERVED_MODEL_NAME=qwen3.6-27b-int4
export VLLM_MODEL=qwen3.6-27b-int4
```

当前服务器上的一键脚本默认使用独立环境 `/home/jnu/venvs/vllm`，加载
`/home/jnu/models/Qwen3.6-27B-int4-AutoRound`，并以 OpenAI-compatible 接口
`http://127.0.0.1:8000/v1` 对外提供模型 `qwen3.6-27b-int4`。启动前可先执行
`scripts/start_vllm.sh check` 只检查环境、CUDA、模型文件和端口状态；停止脚本启动的服务可执行
`scripts/start_vllm.sh stop`。

更多命令和默认配置见 `docs/vllm_startup.md`。

如果 `.venv` 缺失或依赖不完整，先执行 `uv sync --extra desktop`，再回到上面的 `.venv/bin/python ...` 启动方式。

## 运行参数

`main.py` 支持以下参数：

- `--template`：模板文件路径，支持 PDF 或图片
- `--pdf`：`--template` 的兼容别名
- `--target`：实拍图片路径
- `--output-dir`：结果输出目录

默认值定义在 `label_detection/core/config.py` 中。

## 示例输入

仓库内可直接使用的样例包括：

- 模板 PDF：`samples/pdfs/600004075219-01.pdf`
- 模板图片：`samples/images/original/type1/1.png`
- 实拍图片：`samples/images/produce/type1/600004075219_1.jpg`
- 参考图片与文本：`samples/reference/`

## 批量样本测试

批量测试入口：

```bash
.venv/bin/python scripts/batch_run_samples.py --dry-run --pair-mode all
```

默认行为：

- 模板目录扫描 `samples/pdfs/`
- 实拍目录扫描 `samples/images/produce/`
- 按文件名中的主编码自动配对
- 名称中的 `-01`、`_1`、`_2` 等后缀会被视为同一编码的不同变体
- 默认输出目录为 `results/batch_samples/`
- 每个 case 只保留 `result.json` 和 `visualization_diff.jpg`
- 批量根目录额外生成 `summary.json`

常用命令：

```bash
# 只预览将要运行的配对，不真正执行
.venv/bin/python scripts/batch_run_samples.py --dry-run --pair-mode all

# 同一编码下，所有模板变体 × 所有实拍变体，全部运行
.venv/bin/python scripts/batch_run_samples.py --pair-mode all

# 同一编码下，只选一个最优模板变体，再配对全部实拍变体
.venv/bin/python scripts/batch_run_samples.py --pair-mode best-template

# 只运行指定编码的所有组合
.venv/bin/python scripts/batch_run_samples.py --code 600004075219 --pair-mode all
```

参数说明：

- `--dry-run`：只显示将要运行的 case 列表，不执行检测
- `--pair-mode all`：同编码下的所有模板变体与所有实拍变体做全组合
- `--pair-mode best-template`：每个编码只选一个优先模板，适合节省时间和算力
- `--code 600004075219`：只跑指定编码，适合单独回归某类样本

推荐用法：

1. 先执行 `--dry-run` 确认配对是否符合预期。
2. 大批量正式跑时优先考虑 `--pair-mode best-template`。
3. 某个编码需要复查时，再用 `--code <编码> --pair-mode all` 精确重跑。

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
