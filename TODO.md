# TODO

## 当前状态

- 已完成目录重构：根目录只保留 `unified_detection.py` 作为主入口，业务代码已迁入 `label_detection/`
- 已整理样例文件：示例输入和参考产物已迁入 `samples/`
- 已删除旧的重复脚本和兼容壳文件

## 本次未完成项

- 未执行单元测试
- 未执行端到端主流程
- 未验证 OCR / VLM / 布局模型相关依赖是否完整可用

## 未验证原因

- 当前本地环境未搭建项目运行依赖
- 当前不计划在本地直接运行该项目

## 后续如果需要运行，优先检查

1. 安装 Python 依赖：`pip install -r requirements.txt`
2. 确认基础库可用：`numpy`、`opencv-python`、`scipy`、`pandas`、`openpyxl`、`requests`、`PyMuPDF`
3. 确认模型侧依赖可用：`paddleocr`、`paddlex`
4. 确认 Ollama 已启动，并已拉取 `qwen3-vl:8b`
5. 确认默认样例路径存在：
   - `samples/pdfs/600004075219-01.pdf`
   - `samples/images/test.jpg`

## 建议的后续验证顺序

1. 先跑纯逻辑测试：`python -m pytest tests/ -q`
2. 再跑一次主入口：
   `python unified_detection.py --pdf samples/pdfs/600004075219-01.pdf --target samples/images/test.jpg`
3. 最后人工检查输出目录 `results/unified/` 中的：
   - `final_result.json`
   - `text_comparison.xlsx`
   - `visualization_diff.jpg`

## 需要特别注意

- 当前代码结构已经切到包内导入，后续新增模块请优先放到 `label_detection/` 下，不要再把业务脚本散落到根目录
- `label_detection/matching/layout.py` 与 `label_detection/services/vlm_service.py` 已改为尽量延迟加载重依赖，但真正运行时仍然需要安装对应库
- `unified_detection.py` 现在只是根入口包装器，实际主流程在 `label_detection/workflows/unified.py`
