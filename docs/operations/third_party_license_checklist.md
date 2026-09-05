# 第三方许可证交付清单

闭源商业交付前需要确认并随包保存第三方许可证文本、版权声明和模型许可证。

| 组件 | 用途 | 交付注意 |
|------|------|----------|
| PySide6 / Qt for Python | 桌面 UI | 重点确认 LGPLv3 动态链接合规、用户替换 Qt 库能力，或采购 Qt 商业授权。参考 Qt for Python Commercial Distribution 和 Qt Licensing。 |
| Qwen3 / Qwen 系列模型 | 本地 VLM/LLM 推理 | 随包保存模型仓库 LICENSE、NOTICE、模型卡和使用限制。 |
| vLLM | OpenAI-compatible 推理服务 | 随包保存 vLLM LICENSE 和依赖许可证清单。 |
| PaddlePaddle / PaddleOCR / PaddleX | OCR 与布局检测 | 随包保存 Paddle 相关 LICENSE、NOTICE 和模型许可证。 |
| OpenCV、NumPy、SciPy、Pillow 等 Python 依赖 | 图像处理和数值计算 | 从锁文件生成依赖清单，并保存许可证文本或 SPDX 摘要。 |

参考链接：

- https://doc.qt.io/qtforpython-6.5/commercial/index.html
- https://doc.qt.io/qt-6/licensing.html
- https://github.com/QwenLM/Qwen3
- https://github.com/vllm-project/vllm
- https://github.com/PaddlePaddle/Paddle

建议交付包增加：

```text
THIRD_PARTY_NOTICES/
├── python-packages.json
├── qt/
├── qwen/
├── vllm/
└── paddle/
```
