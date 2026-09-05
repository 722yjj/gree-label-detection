# Sensitive Text Promotion (兜底提权) 配置指南

## 1. 背景与问题描述

在全图比对算法（Traditional Full-Image Diff）中，针对极小且高密度的文字区域差异（`micro_text_candidate`），系统曾经内置了一套强制提权兜底规则（`promote_sensitive_text_line_discard`）。

**原逻辑行为：**
如果一个小面积的像素差异刚好落在了 PDF 模板提取出的文本行范围内，并且该文本行包含易混淆字符（如 `1, l, I, 0, O, S, 5, Z, 2, 8, G, 6`）或该差异非常小且密集，那么即使视觉大模型（VLM）判定该差异只是噪点或压缩伪影（原始判定为 `discard`），系统也会出于防漏检的防御性目的，强制将其提权为 `keep`（抛出异常）。

在最终的 `result.json` 中，这类差异会包含如下信息：
- `vlm_decision`: "keep"
- `vlm_original_decision`: "discard"
- `vlm_promoted_reason`: "sensitive_pdf_text_line_micro_diff"

**产生的问题：**
在实际产线环境中，由于实拍图经常存在因打印质量、墨水深浅、摄像头对焦等引起的正常噪点或轻微模糊。VLM 大模型已经具备良好的判断能力，能正确识别出这些不是真实的印刷缺陷。强制提权逻辑虽然防止了极端情况下的漏检，但也造成了大量的误报（False Positives），严重影响了用户的体验。

## 2. 更新与配置说明

为了解决误报过高的问题，现在该强制提权兜底策略默认 **处于关闭状态**。

如果您的特定检测场景对微小字符的漏报容忍度极低，且能够接受较高的误报率，可以通过设置系统环境变量来重新开启该功能。

### 环境变量开关

- **变量名**: `ENABLE_SENSITIVE_TEXT_PROMOTION`
- **默认值**: `false` （不启用）
- **启用方法**: 将该环境变量设置为 `true`, `1`, `yes`, 或 `on` 即可开启强制兜底提权机制。

### 开启示例

在 Linux 终端中启动时：

```bash
export ENABLE_SENSITIVE_TEXT_PROMOTION=true
uv run python desktop_app/main.py
```

或者在桌面端对应的服务启动脚本/配置文件中配置该环境变量即可。

## 3. 影响范围

此配置仅影响 `label_detection/workflows/traditional_full_image_diff.py` 内部的 `promote_sensitive_text_line_discard` 函数行为。
关闭提权后，系统将完全尊重视觉大模型（VLM）对于微小文本框的判决。如果 VLM 认为是噪点（`discard`），系统将不会再把它改成 `keep`。
