# PP-DocLayoutV3 HuggingFace/PyTorch GPU POC 追溯记录

记录日期：2026-04-30
当前主机：`spark-b34f`
项目目录：`/home/jnu/projects/gree-label-detection`

目标：评估在当前 `aarch64 + NVIDIA GB10 + CUDA 13` 主机上，绕开 Paddle GPU wheel 缺失问题，使用 HuggingFace Transformers + PyTorch CUDA 跑 `PP-DocLayoutV3` 是否可行，以及是否可能比当前 PaddleX CPU 版更快。

## 背景

当前主流程中的 PP-DocLayoutV3 调用位于：

```text
label_detection/matching/layout.py
```

调用方式：

```python
create_predictor(
    model_name="PP-DocLayoutV3",
    threshold=...,
    device=LAYOUT_DEVICE,
)
```

配置默认请求：

```text
LAYOUT_DEVICE=gpu:0
```

但当前项目 `.venv` 中 Paddle 是 CPU 版：

```text
arch: aarch64
paddle: 3.2.1
paddle.is_compiled_with_cuda(): False
paddle device: cpu
```

因此项目会打印：

```text
[Paddle] PP-DocLayoutV3 请求使用 gpu:0，但当前 Paddle 不是 CUDA 版本，回退到 CPU
```

本机 dry-run 查询 `paddlepaddle-gpu` 结果：

```text
paddlepaddle-gpu has no wheels with a matching platform tag ... aarch64
available platforms: x86_64, win_amd64
```

结论：当前主机上不能直接通过 pip 给项目 `.venv` 安装官方 Paddle GPU wheel。

## 隔离目录

本次 POC 不修改项目 `.venv`，也不接入主流程。

创建的隔离目录：

```text
/home/jnu/venvs/gree-layout-hf-gpu
/home/jnu/models/layout
/home/jnu/models/layout/hf_home
/home/jnu/projects/gree-label-detection/results/layout_hf_gpu_poc
```

当前空间占用：

```text
/home/jnu/venvs/gree-layout-hf-gpu  4.9G
/home/jnu/models/layout             128M
```

## 安装命令

创建 venv：

```bash
mkdir -p /home/jnu/venvs /home/jnu/models/layout/pp-doclayoutv3-hf results/layout_hf_gpu_poc
~/.local/bin/uv venv /home/jnu/venvs/gree-layout-hf-gpu --python 3.12
```

安装 PyTorch CUDA 13 nightly：

```bash
~/.local/bin/uv pip install \
  --python /home/jnu/venvs/gree-layout-hf-gpu/bin/python \
  torch \
  --index-url https://download.pytorch.org/whl/nightly/cu130
```

安装 HuggingFace / 图像依赖：

```bash
~/.local/bin/uv pip install \
  --python /home/jnu/venvs/gree-layout-hf-gpu/bin/python \
  transformers accelerate safetensors pillow requests \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

补装缺失依赖：

```bash
~/.local/bin/uv pip install \
  --python /home/jnu/venvs/gree-layout-hf-gpu/bin/python \
  torchvision \
  --index-url https://download.pytorch.org/whl/nightly/cu130

~/.local/bin/uv pip install \
  --python /home/jnu/venvs/gree-layout-hf-gpu/bin/python \
  opencv-python-headless \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 关键包版本

```text
torch                  2.13.0.dev20260429+cu130
torchvision            0.27.0.dev20260429+cu130
transformers           5.7.0
accelerate             1.13.0
safetensors            0.7.0
opencv-python-headless 4.13.0.92
pillow                 12.2.0
numpy                  2.4.4
cuda-toolkit           13.0.2
nvidia-cudnn-cu13      9.20.0.48
nvidia-cublas          13.1.1.3
```

## CUDA 验证

验证命令：

```bash
/home/jnu/venvs/gree-layout-hf-gpu/bin/python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('cuda_version', torch.version.cuda)
if torch.cuda.is_available():
    print('device_count', torch.cuda.device_count())
    print('device_name', torch.cuda.get_device_name(0))
    print('capability', torch.cuda.get_device_capability(0))
PY
```

结果：

```text
torch 2.13.0.dev20260429+cu130
cuda_available True
cuda_version 13.0
device_count 1
device_name NVIDIA GB10
capability (12, 1)
```

## 模型加载验证

模型：

```text
PaddlePaddle/PP-DocLayoutV3_safetensors
```

加载方式：

```python
from transformers import AutoImageProcessor, AutoModelForObjectDetection

processor = AutoImageProcessor.from_pretrained("PaddlePaddle/PP-DocLayoutV3_safetensors")
model = AutoModelForObjectDetection.from_pretrained(
    "PaddlePaddle/PP-DocLayoutV3_safetensors"
).to("cuda").eval()
```

首次加载结果：

```text
processor_loaded: 1.852s
model_cpu_loaded: 19.401s
model_to_cuda: 0.444s
process real time: 24.29s
model_class: PPDocLayoutV3ForObjectDetection
num_labels: 25
```

后续缓存加载到 GPU：

```text
LOAD_TO_CUDA_SECONDS 2.8591
```

注意：

- 首次冷加载包含 HuggingFace 下载和模型初始化，不代表常驻服务推理耗时。
- 如果每次主流程都启动子进程并重新加载模型，会因为冷加载而变慢。
- 如果做常驻 layout service，冷加载可以摊薄。

## 当前 PaddleX CPU 基线

测试图片：

```text
results/rapidocr_tensorrt_full_run_600004075219_7/debug/preprocess/template_preprocessed.png
results/rapidocr_tensorrt_full_run_600004075219_7/debug/preprocess/target_preprocessed.png
```

当前 PaddleX CPU 实测：

```text
模型初始化: 0.971s

第 1 轮:
template_preprocessed.png: 2.579s
target_preprocessed.png:   2.427s
两张合计: 5.006s

第 2 轮 warm:
template_preprocessed.png: 2.190s
target_preprocessed.png:   2.239s
两张合计: 4.429s

第 3 轮 warm:
template_preprocessed.png: 2.161s
target_preprocessed.png:   2.358s
两张合计: 4.519s
```

估算：

```text
单图 PP-DocLayout CPU: 约 2.2s - 2.6s
每个 case 两张图: 约 4.4s - 5.0s
首次进程额外初始化: 约 1.0s
```

当前 PaddleX CPU 输出标签分布：

```text
template_preprocessed.png total 6 labels {'table': 1, 'text': 3, 'image': 2}
target_preprocessed.png   total 11 labels {'doc_title': 2, 'text': 5, 'table': 1, 'image': 3}
```

## HF/PyTorch GPU 计时

同样两张图片，模型已加载并常驻 GPU 后：

```text
warmup_pair total: 0.7594s

PAIR_ROUND 1 total=0.1210s pre=0.0256s h2d=0.0004s fwd=0.0854s post=0.0095s
PAIR_ROUND 2 total=0.1233s pre=0.0263s h2d=0.0006s fwd=0.0873s post=0.0090s
PAIR_ROUND 3 total=0.1212s pre=0.0250s h2d=0.0004s fwd=0.0860s post=0.0097s
PAIR_ROUND 4 total=0.1246s pre=0.0272s h2d=0.0004s fwd=0.0881s post=0.0089s
PAIR_ROUND 5 total=0.1111s pre=0.0260s h2d=0.0004s fwd=0.0758s post=0.0089s
```

单图：

```text
template_preprocessed.png:
  total=0.0476s
  total=0.0452s
  total=0.0433s

target_preprocessed.png:
  total=0.0658s
  total=0.0711s
  total=0.0714s
```

估算：

```text
常驻 HF/PyTorch GPU 单图: 约 0.04s - 0.07s
常驻 HF/PyTorch GPU 两图 batch: 约 0.11s - 0.13s
```

## 输出差异观察

HF/PyTorch GPU 输出标签分布：

```text
template_preprocessed.png total 7 labels:
{'image': 2, 'table': 1, 'text': 3, 'vision_footnote': 1}

target_preprocessed.png total 15 labels:
{'doc_title': 2, 'image': 4, 'paragraph_title': 1, 'table': 1, 'text': 7}
```

与 PaddleX CPU 输出相比：

```text
template:
  PaddleX CPU: total 6, image 2
  HF GPU:      total 7, image 2

target:
  PaddleX CPU: total 11, image 3
  HF GPU:      total 15, image 4
```

结论：

- `image` 类区域数量并不完全一致。
- HF 输出还有 `polygon_points`，当前项目主流程只吃 `coordinate` 形式，需要适配。
- 不能只按速度替换，必须验证最终图形匹配和差异标注是否保持一致。

## 初步结论

当前主机上，HF/PyTorch GPU 路线是可行的：

```text
PyTorch CUDA 可用
PP-DocLayoutV3 safetensors 可加载
模型可放到 GB10 GPU
warm 后两图 batch 推理约 0.11s - 0.13s
```

和当前 PaddleX CPU 相比：

```text
PaddleX CPU 两图 layout: 约 4.4s - 5.0s
HF GPU 两图 layout warm: 约 0.11s - 0.13s
```

如果做成常驻服务，layout 阶段理论上有明显提速空间。

但不能直接替换主流程：

- HF 输出区域数量和 PaddleX CPU 不完全一致。
- 需要实现 label 映射、score 阈值、box/polygon 转换。
- 需要跑完整图形匹配回归，而不仅是模型推理计时。
- 如果每个 case 都重启进程并加载模型，冷加载会抵消收益。

## 2026-04-30 主流程可选后端接入

已按“默认不变、显式启用”的方式接入主流程。

新增配置：

```text
LAYOUT_BACKEND=paddlex
LAYOUT_BACKEND=hf-pytorch-gpu
HF_LAYOUT_PYTHON=/home/jnu/venvs/gree-layout-hf-gpu/bin/python
HF_LAYOUT_MODEL_ID=PaddlePaddle/PP-DocLayoutV3_safetensors
HF_LAYOUT_DEVICE=cuda
HF_LAYOUT_TIMEOUT=120
HF_LAYOUT_HF_HOME=/home/jnu/models/layout/hf_home
```

默认仍是：

```text
LAYOUT_BACKEND=paddlex
```

实现文件：

```text
label_detection/core/config.py
label_detection/matching/layout.py
label_detection/workflows/unified.py
tests/test_layout_backend.py
```

实现方式：

- `detect_layout_regions()` 增加后端分发。
- `paddlex` 后端沿用原 PaddleX CPU/GPU 逻辑。
- `hf-pytorch-gpu` 后端启动一个常驻 sidecar worker。
- sidecar worker 在隔离 Python 环境中加载 HuggingFace PP-DocLayoutV3，并将模型放到 `cuda`。
- 主进程通过 JSON line 协议发送图片路径和阈值。
- worker 返回兼容当前主流程的区域结构：

```python
{
    "label": label,
    "score": score,
    "coordinate": [x1, y1, x2, y2],
    "polygon_points": [...],  # 可选保留，当前主流程主要使用 coordinate
}
```

主流程结果中新增追溯字段：

```text
final_result.json -> graphic_comparison.layout_backend
```

### 单点后端验证

命令：

```bash
env \
  LAYOUT_BACKEND=hf-pytorch-gpu \
  HF_LAYOUT_PYTHON=/home/jnu/venvs/gree-layout-hf-gpu/bin/python \
  HF_LAYOUT_HF_HOME=/home/jnu/models/layout/hf_home \
  HF_LAYOUT_TIMEOUT=120 \
  .venv/bin/python - <<'PY'
from time import perf_counter
from label_detection.matching.layout import detect_layout_regions, get_layout_backend_name
paths = [
    'results/rapidocr_tensorrt_full_run_600004075219_7/debug/preprocess/template_preprocessed.png',
    'results/rapidocr_tensorrt_full_run_600004075219_7/debug/preprocess/target_preprocessed.png',
]
print('backend', get_layout_backend_name())
t0 = perf_counter()
for path in paths:
    ts = perf_counter()
    regions = detect_layout_regions(path)
    print(path, 'seconds', round(perf_counter() - ts, 4), 'regions', len(regions), 'image_regions', sum(1 for r in regions if r['label'] == 'image'))
print('total', round(perf_counter() - t0, 4))
PY
```

结果：

```text
backend hf-pytorch-gpu
template_preprocessed.png seconds 5.0785 regions 7 image_regions 2
target_preprocessed.png seconds 0.1482 regions 15 image_regions 4
total 5.2267
```

说明：

- 第一张包含 worker 初始化后首次 CUDA forward/warm-up。
- 第二张已经复用同一个 worker 和 GPU 模型。
- 批量运行时，同一 Python 进程内会继续复用 worker。

### 完整主流程验证

命令：

```bash
/usr/bin/time -p env \
  LAYOUT_BACKEND=hf-pytorch-gpu \
  HF_LAYOUT_PYTHON=/home/jnu/venvs/gree-layout-hf-gpu/bin/python \
  HF_LAYOUT_HF_HOME=/home/jnu/models/layout/hf_home \
  HF_LAYOUT_TIMEOUT=120 \
  OCR_BACKEND=rapidocr-tensorrt \
  RAPIDOCR_PYTHON=/home/jnu/venvs/gree-ocr-trt/bin/python \
  RAPIDOCR_TRT_CACHE_DIR=/home/jnu/models/ocr/tensorrt-engines/gb10 \
  RAPIDOCR_MODEL_TYPE=mobile \
  PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
  NO_PROXY=localhost,127.0.0.1,::1 \
  no_proxy=localhost,127.0.0.1,::1 \
  .venv/bin/python main.py \
    --template samples/pdfs/600004075219.pdf \
    --target samples/images/real/600004075219_7.jpg \
    --output-dir results/hf_layout_gpu_full_run_600004075219_7 \
    --output-mode debug
```

结果：

```text
real 50.83s
success: True
OCR backend: rapidocr-tensorrt
layout backend: hf-pytorch-gpu
text fields: 22/22
graphic comparison: 5 matched/resolved, 1 mismatched
verdict: 文字一致，图形存在差异 (1 处不匹配, 0 个未恢复)
```

输出：

```text
results/hf_layout_gpu_full_run_600004075219_7/final_result.json
results/hf_layout_gpu_full_run_600004075219_7/visualization_diff.jpg
results/hf_layout_gpu_full_run_600004075219_7/debug/text_comparison.xlsx
```

观察：

- 主流程可完整跑通。
- `final_result.json` 已记录 `graphic_comparison.layout_backend = hf-pytorch-gpu`。
- HF layout 输出比 PaddleX CPU 多出部分 image 区域，因此当前图形恢复/匹配路径会有轻微变化。
- 单例结果仍为 1 处图形不匹配，但匹配/恢复计数与 PaddleX CPU 结果不完全一致。
- 这次单例总耗时不代表批量 warm 后收益，仍受到 LLM/VLM、OCR sidecar、HF layout 首次 warm-up 影响。

### 已执行检查

```bash
.venv/bin/python -m py_compile label_detection/matching/layout.py label_detection/core/config.py label_detection/workflows/unified.py
.venv/bin/python -m pytest tests/test_layout_backend.py tests/test_unified_output_mode.py
```

结果：

```text
4 passed
```

## 2026-04-30 批量验证：real 样本

命令：

```bash
/usr/bin/time -p env \
  LAYOUT_BACKEND=hf-pytorch-gpu \
  HF_LAYOUT_PYTHON=/home/jnu/venvs/gree-layout-hf-gpu/bin/python \
  HF_LAYOUT_HF_HOME=/home/jnu/models/layout/hf_home \
  HF_LAYOUT_TIMEOUT=120 \
  OCR_BACKEND=rapidocr-tensorrt \
  RAPIDOCR_PYTHON=/home/jnu/venvs/gree-ocr-trt/bin/python \
  RAPIDOCR_TRT_CACHE_DIR=/home/jnu/models/ocr/tensorrt-engines/gb10 \
  RAPIDOCR_MODEL_TYPE=mobile \
  PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
  NO_PROXY=localhost,127.0.0.1,::1 \
  no_proxy=localhost,127.0.0.1,::1 \
  .venv/bin/python scripts/batch_run_samples.py \
    --template-root samples/pdfs \
    --target-root samples/images/real \
    --output-dir results/hf_layout_gpu_real_batch \
    --pair-mode best-template
```

结果：

```text
输出目录: results/hf_layout_gpu_real_batch
summary: results/hf_layout_gpu_real_batch/summary.json
case count: 17
success/failure/skipped: 17/0/0
external wall time: real 634.31s
case duration sum/avg/min/max: 633.14s / 37.24s / 32.16s / 45.17s
OCR backend: rapidocr-tensorrt
layout backend: hf-pytorch-gpu (通过本次命令环境指定)
文字字段: 351/382 = 91.88%
图形不匹配合计: 20
未恢复合计: 1
```

与上一批 `RapidOCR TensorRT + PaddleX CPU layout` 对比：

```text
旧结果目录: results/rapidocr_tensorrt_real_batch
旧 external wall time: real 620.10s
旧 case duration sum/avg/min/max: 619.70s / 36.45s / 32.74s / 43.53s
旧文字字段: 351/382 = 91.88%
旧图形不匹配合计: 17
旧未恢复合计: 0

HF layout GPU 批量 wall time: +14.21s (+2.29%)
HF layout GPU case duration sum: +13.44s
HF layout GPU 图形不匹配: +3
HF layout GPU 未恢复: +1
```

逐样本差异集中在 `600004075219` 这一组：

```text
600004075219_1: graph +1, time +1.64s
600004075219_2: graph +1, time +2.93s
600004075219_5: unrecovered +1, time +3.77s
600004075219_6: graph +1, time +4.02s
600004078454_*: 图形不匹配数量基本与 PaddleX CPU layout 一致，耗时有小幅上下波动
```

观察：

- HF/PyTorch GPU 的 layout 单次热推理很快，但完整批量没有变快；主耗时仍在 LLM/VLM、OCR sidecar、预处理和图形恢复判定。
- HF layout 与 PaddleX CPU 的后处理输出不完全一致，`600004075219` 的实拍图更容易多出 `image` 区域，导致图形侧多报。
- 目前不建议把 `LAYOUT_BACKEND=hf-pytorch-gpu` 设为默认。更稳妥的做法是保留可选后端，先给 HF 后端加专属区域过滤/合并策略，再做回归。

批量摘要字段补充：

- `label_detection/batch_samples.py` 的瘦身版 `result.json` 已补充透传 `layout_backend`、`template_regions_total_count`、`target_regions_total_count` 以及拆分/过滤区域统计。
- 注意：本次已跑出的 `results/hf_layout_gpu_real_batch/*/result.json` 是补字段前生成的，命令环境和本 trace 可用于追溯；后续重跑会写入新增字段。

已执行检查：

```bash
.venv/bin/python -m pytest tests/test_batch_samples.py tests/test_layout_backend.py
```

结果：

```text
8 passed
```

## 建议下一步

1. 保留当前 PaddleX CPU 作为默认后端。
2. 给 `LAYOUT_BACKEND=hf-pytorch-gpu` 增加后端专属区域过滤/合并策略，重点处理 `600004075219` 多检的实拍 `image` 区域。
3. 过滤策略完成后重跑 `samples/images/real` 批量回归，对比：

```text
layout region count
image region count
split region count
graphic matched/mismatched
visualization_diff.jpg
end-to-end time
```

4. 若批量结果稳定，再考虑桌面端/CLI 配置入口。

## 彻底清理

删除本次 POC 创建的隔离环境和模型缓存：

```bash
rm -rf /home/jnu/venvs/gree-layout-hf-gpu
rm -rf /home/jnu/models/layout
rm -rf /home/jnu/projects/gree-label-detection/results/layout_hf_gpu_poc
rm -rf /home/jnu/projects/gree-label-detection/results/hf_layout_gpu_full_run_600004075219_7
rm -rf /home/jnu/projects/gree-label-detection/results/hf_layout_gpu_real_batch
```

不会影响：

```text
项目 .venv
PaddleX CPU 模型缓存 /home/jnu/.paddlex/official_models/PP-DocLayoutV3
RapidOCR/TensorRT POC /home/jnu/venvs/gree-ocr-trt
RapidOCR/TensorRT engine /home/jnu/models/ocr
Ollama
MVS 相机配置
```

如果需要清理 uv 的全局下载缓存，需单独评估，不建议盲删，因为可能影响其它 uv 环境复用下载包。
