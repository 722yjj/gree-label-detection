# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

格力空调标签差异检测系统：基于 OCR + VLM（Vision Language Model）对比模板文件与实拍标签图片在文字和图形上的差异。模板支持 PDF 和图片输入。

## 常用命令

```bash
# 安装依赖
pip install -r requirements.txt

# 启动 Ollama 服务（运行前必须）
ollama serve
ollama pull qwen3-vl:8b

# 运行主流程
python main.py --template samples/pdfs/600004075219-01.pdf --target samples/images/produce/type1/600004075219_1.jpg

# 运行测试
pytest tests/

# 运行单个测试文件
pytest tests/test_vlm_detection.py

# VLM 目标检测测试脚本
python scripts/run_vlm_object_detection.py --help

# VLM 布局区域对比脚本（参数写在文件顶部，直接运行）
python scripts/test_vlm_layout_regions.py
```

## 架构概览

入口 `main.py` 调用 `label_detection/workflows/unified.py::main()`，核心流程分五步：

```
模板文件(PDF/图片) + 实拍图
  1. 预处理：模板提取(PDF红框检测/直接用图片) + 目标图透视矫正(失败回退黑框/边缘裁剪)
  2. 文字检测：PaddleOCR → 标签类型推断 → Ollama+Qwen3-VL 结构化提取 → 字段对比输出Excel
  3. 图形区域检测与对比：PP-DocLayoutV3 区域检测 → 匈牙利算法匹配 → VLM 视觉对比
  4. 可视化：在图上标注文字/图形差异（红框/黄框）
  5. 汇总：输出 JSON + Excel + 可视化图片
```

## 关键模块职责

| 模块路径 | 职责 |
|----------|------|
| `label_detection/workflows/unified.py` | 主流程编排（~1024行），包含 `run_unified_detection()` |
| `label_detection/core/config.py` | 全局配置（Ollama地址、模型名、OCR语言、阈值等），通过环境变量覆盖 |
| `label_detection/schema.py` | Pydantic v2 数据模型：`AirConditionerLabel`(标准14字段) / `CompactSpecLabel`(简约7字段) |
| `label_detection/preprocessing/pipeline.py` | 模板图/目标图预处理入口 |
| `label_detection/preprocessing/perspective.py` | 透视矫正与标签主体定位（候选四边形评分筛选） |
| `label_detection/extraction/pdf.py` | PDF 红色虚线框检测提取（PyMuPDF） |
| `label_detection/extraction/template_source.py` | 模板输入解析（PDF 或图片） |
| `label_detection/matching/layout.py` | PP-DocLayoutV3 区域检测 + 匈牙利匹配 + 未匹配区域恢复 |
| `label_detection/services/ocr_service.py` | PaddleOCR 封装 |
| `label_detection/services/vlm_service.py` | Ollama VLM 区域对比封装 |
| `label_detection/services/vlm_detection.py` | 独立 VLM 目标检测（实验性） |

## 运行时依赖

- **Ollama + qwen3-vl:8b**：VLM 推理，需本地启动服务
- **PaddleOCR**：文字识别，首次运行自动下载模型
- **PP-DocLayoutV3 / PaddleX**：布局区域检测，首次运行自动下载模型

## 设计约定

- 全局单例懒加载：OCR 引擎、LLM 实例、布局检测器通过模块级 `get_xxx()` 函数延迟初始化
- 配置通过 `label_detection/core/config.py` 集中管理，支持 `.env` 和环境变量覆盖
- `AGENT_CHANGELOG.md` 记录所有 code agent 的关键改动，修改核心逻辑后需追加记录
- 默认输出目录 `results/unified/`
- 项目使用中文日志输出和中文注释

## 注意事项

- 透视矫正已移除"浅色区域检测"分支，当前策略：深色前景聚合 → 边缘检测 → 黑框检测回退
- 区域匹配使用多因子代价函数（中心距离、面积比、宽高比、IoU）
- VLM 输出不稳定，图形对比结果可能需人工复核
- 标签类型由 `infer_label_kind()` 根据 OCR 文本自动推断
