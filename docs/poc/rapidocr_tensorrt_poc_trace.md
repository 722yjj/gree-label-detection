# RapidOCR + PP-OCRv5 + TensorRT POC 追溯记录

记录日期：2026-04-29
当前主机：`spark-b34f`
当前分支：`experiment/graphic-diff-mask`
目标：在不污染当前项目环境和其它项目的前提下，评估 `RapidOCR + PP-OCRv5 + TensorRT` 作为 PaddleOCR GPU 不可用时的替代 OCR 路线。

## 背景

当前项目主流程仍使用 PaddleOCR / PaddleX。新机 DGX/GB10 上已能跑完整检查流程，但 OCR / PP-DocLayout 当前实际落在 CPU 上。

已观察到的原因：

- 主机架构是 `aarch64 / arm64`。
- GPU 是 `NVIDIA GB10`，驱动 `580.142`，CUDA Driver `13.0`。
- 本机 CUDA toolkit 为 `13.0.88`。
- 当前项目 `.venv` 中安装的是 CPU 版 `paddlepaddle==3.2.1`。
- `paddle.is_compiled_with_cuda()` 返回 `False`。
- 项目运行时会提示 PaddleOCR / PP-DocLayout 请求 `gpu:0`，但当前 Paddle 不是 CUDA 版本，回退到 CPU。

## 已执行调查

### 1. 本机基础环境

命令：

```bash
uname -m
nvidia-smi
nvcc --version
```

结论：

```text
arch: aarch64
GPU: NVIDIA GB10
Driver: 580.142
CUDA driver: 13.0
CUDA toolkit: 13.0.88
```

### 2. 当前项目 Python 运行时

命令：

```bash
.venv/bin/python - <<'PY'
mods = ['paddle', 'torch', 'onnxruntime', 'tensorrt', 'cv2', 'paddleocr']
for m in mods:
    try:
        mod = __import__(m)
        print(m, getattr(mod, '__version__', '?'))
        if m == 'paddle':
            print('paddle.is_compiled_with_cuda=', mod.is_compiled_with_cuda())
        if m == 'torch':
            print('torch.cuda.is_available=', mod.cuda.is_available())
    except Exception as exc:
        print(m, type(exc).__name__, exc)
PY
```

结果摘要：

```text
paddle: installed 3.2.1
paddle.is_compiled_with_cuda = False
torch: not installed
onnxruntime: not installed
tensorrt: not installed
cv2: installed 4.10.0
```

说明：

- `paddleocr` 直接导入会受 LangChain 兼容问题影响；项目内已有兼容层处理主流程导入。
- 这不改变 Paddle 是 CPU 版的事实。

### 3. ONNX Runtime 可安装性

命令：

```bash
~/.local/bin/uv pip install --python .venv/bin/python --dry-run onnxruntime-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple
~/.local/bin/uv pip install --python .venv/bin/python --dry-run onnxruntime -i https://pypi.tuna.tsinghua.edu.cn/simple
```

结果：

- `onnxruntime-gpu`：当前 `linux/aarch64 + Python 3.12` 没有匹配 wheel，解析失败。
- `onnxruntime` CPU：可安装。

结论：

ONNX Runtime GPU 不是当前裸机 Python 环境的首选路线。CPU 版可作为保底 OCR 后端或快速对比基线。

### 4. PyTorch CUDA 13 可安装性

命令：

```bash
~/.local/bin/uv pip install --python .venv/bin/python --dry-run torch --index-url https://download.pytorch.org/whl/nightly/cu130
```

结果摘要：

```text
torch==2.13.0.dev20260429+cu130
triton==3.7.0+...
nvidia-cudnn-cu13
nvidia-nccl-cu13
...
```

结论：

PyTorch CUDA 13 nightly 在当前 `aarch64` 平台可解析到 wheel。若后续 RapidOCR TensorRT 不顺，PyTorch 后端 OCR 是可验证的第二路线。

### 5. RapidOCR 可安装性

命令：

```bash
~/.local/bin/uv pip install --python .venv/bin/python --dry-run rapidocr -i https://pypi.tuna.tsinghua.edu.cn/simple
```

结果摘要：

```text
rapidocr==3.8.1
```

结论：

RapidOCR 主包可在当前平台安装。RapidOCR 文档显示 PP-OCRv5 支持 `onnxruntime`、`openvino`、`paddle`、`torch`、`mnn`、`tensorrt` 等后端，其中 TensorRT 后端要求 `rapidocr>=3.7.0`。

参考：

- RapidOCR 模型列表：https://rapidai.github.io/RapidOCRDocs/main/model_list/
- TensorRT 安装方式说明：https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html
- PaddleOCR 高性能推理说明：https://www.paddleocr.ai/latest/en/version3.x/deployment/high_performance_inference.html

### 6. TensorRT 系统包可用性

命令：

```bash
apt-cache policy tensorrt python3-libnvinfer libnvinfer10 libnvinfer-dev
```

结果摘要：

NVIDIA SBSA apt 源中存在 `arm64` TensorRT 包，例如：

```text
tensorrt 10.14.x + cuda13.0
tensorrt 10.15.x + cuda13.1
tensorrt 10.16.x + cuda13.2
python3-libnvinfer
libnvinfer10
libnvinfer-dev
```

结论：

本机系统层面具备安装 TensorRT 的条件，但 apt 安装是系统级变更，需要单独批准和记录。

## 关键决策

### 决策 1：不在当前 `.venv` 中直接安装实验依赖

当前项目 `.venv` 已承载桌面端、PaddleOCR CPU、MVS 相机、Ollama 联调等能力。RapidOCR + TensorRT 仍处于 POC 阶段，不应直接装进 `.venv`。

隔离建议：

```text
/home/jnu/venvs/gree-ocr-trt
/home/jnu/models/ocr/ppocrv5
/home/jnu/models/ocr/tensorrt-engines/gb10
/home/jnu/projects/gree-label-detection/results/ocr_trt_poc
```

其中：

- venv 放在 repo 外，避免 Git 状态污染。
- 模型和 TensorRT engine 放在 repo 外，避免大文件和机器相关 engine 混入项目。
- benchmark 输出放 `results/`，该目录已被 `.gitignore` 忽略。

### 决策 2：TensorRT apt 包必须单独确认后再装

TensorRT 的 Debian/RPM 安装是系统级安装，固定安装位置通常在 `/usr`，需要 root 权限，并且同一机器上通常只维护一个 TensorRT minor 版本。

风险：

- 可能影响其它项目中已有或未来使用 TensorRT 的环境。
- TensorRT engine 与 GPU、TensorRT 版本、模型 shape 强相关，不应跨机器复用。

控制方式：

- 安装前记录 `apt-cache policy`。
- 固定 CUDA 13.0 匹配版本优先，避免随 apt 安装 CUDA 13.2 组件。
- 安装后记录 `dpkg -l | rg 'tensorrt|nvinfer|nvonnxparser'`。
- POC 不改 `/etc/profile`、`~/.bashrc`、项目 `.venv`。

### 决策 3：POC 通过前不替换主流程

主流程仍保持：

```text
PaddleOCR CPU + 当前文字结构化逻辑
```

POC 通过后，才增加可配置 OCR 后端：

```bash
OCR_BACKEND=paddleocr
OCR_BACKEND=rapidocr-tensorrt
OCR_BACKEND=rapidocr-onnxruntime-cpu
```

默认仍用现有后端，避免线上流程突然改变。

## POC 验证标准

### 必须验证

1. RapidOCR 能加载 PP-OCRv5 模型。
2. TensorRT 后端能在 GB10 上初始化。
3. 对固定样本输出可追溯 OCR 框和文本。
4. 和现有 PaddleOCR CPU 对比：
   - 单图耗时
   - 字段级准确率
   - 是否漏识别关键字段
   - 数字、单位、型号、条码是否保持 exact match
5. TensorRT engine 文件位置可控，且不写入项目 tracked 文件。

### 固定样本

优先用当前项目已有样本：

```text
samples/pdfs/600004075219.pdf
samples/images/real/600004075219_1.jpg
samples/images/real/600004075219_2.jpg
samples/images/real/600004075219_7.jpg
```

选择理由：

- 已跑过完整流程。
- 这组图有实际 OCR 误识别案例，例如 `Weight`、`frequency`、`air_volume` 等字段。
- 第 7 张文字结果曾达到 `24/24`，可作为较好质量参考。

## 建议执行步骤

### 阶段 0：只建隔离目录

```bash
mkdir -p /home/jnu/venvs
mkdir -p /home/jnu/models/ocr/ppocrv5
mkdir -p /home/jnu/models/ocr/tensorrt-engines/gb10
mkdir -p /home/jnu/projects/gree-label-detection/results/ocr_trt_poc
```

### 阶段 1：创建独立 uv 环境

```bash
cd /home/jnu/projects/gree-label-detection
~/.local/bin/uv venv /home/jnu/venvs/gree-ocr-trt --python 3.12
```

### 阶段 2：先安装 RapidOCR，不装 TensorRT

```bash
~/.local/bin/uv pip install --python /home/jnu/venvs/gree-ocr-trt/bin/python -i https://pypi.tuna.tsinghua.edu.cn/simple rapidocr
~/.local/bin/uv pip install --python /home/jnu/venvs/gree-ocr-trt/bin/python -i https://pypi.tuna.tsinghua.edu.cn/simple onnxruntime
```

目标：

- 确认 RapidOCR import 成功。
- 确认模型配置 API。
- 先用非 TensorRT 后端跑通最小样例。

### 阶段 3：TensorRT 安装前检查

```bash
apt-cache policy tensorrt python3-libnvinfer libnvinfer10 libnvinfer-dev
dpkg -l | rg 'tensorrt|nvinfer|nvonnxparser' || true
```

只有在确认版本后，才执行 apt 安装。

### 阶段 4：TensorRT 后端验证

目标输出：

```text
results/ocr_trt_poc/
  env.txt
  rapidocr_import.txt
  benchmark_600004075219_1.json
  benchmark_600004075219_2.json
  benchmark_600004075219_7.json
```

每个 benchmark JSON 至少记录：

```json
{
  "image": "...",
  "backend": "rapidocr-tensorrt",
  "model": "PP-OCRv5",
  "elapsed_seconds": 0.0,
  "texts": [],
  "boxes": [],
  "field_observations": {}
}
```

## 实际执行记录

### 2026-04-29 隔离环境和 CPU 基线

已创建隔离目录：

```text
/home/jnu/venvs/gree-ocr-trt
/home/jnu/models/ocr/ppocrv5
/home/jnu/models/ocr/tensorrt-engines/gb10
/home/jnu/projects/gree-label-detection/results/ocr_trt_poc
```

已用 `uv` 在 repo 外创建独立 Python 3.12 环境，并通过清华源安装：

```text
rapidocr==3.8.1
onnxruntime==1.25.1
numpy==2.4.4
opencv-python==4.13.0.92
```

实际命令：

```bash
~/.local/bin/uv venv /home/jnu/venvs/gree-ocr-trt --python 3.12
~/.local/bin/uv pip install --python /home/jnu/venvs/gree-ocr-trt/bin/python -i https://pypi.tuna.tsinghua.edu.cn/simple rapidocr
~/.local/bin/uv pip install --python /home/jnu/venvs/gree-ocr-trt/bin/python -i https://pypi.tuna.tsinghua.edu.cn/simple onnxruntime
```

已确认 RapidOCR 导出当前需要的配置枚举：

```text
RapidOCR
EngineType.TENSORRT
EngineType.ONNXRUNTIME
OCRVersion.PPOCRV5
```

首次运行时 RapidOCR 自动下载并缓存了 PP-OCR 模型到隔离 venv 的 site-packages 中：

```text
/home/jnu/venvs/gree-ocr-trt/lib/python3.12/site-packages/rapidocr/models/ch_PP-OCRv5_det_mobile.onnx  4.6M
/home/jnu/venvs/gree-ocr-trt/lib/python3.12/site-packages/rapidocr/models/ch_PP-OCRv5_rec_mobile.onnx   16M
/home/jnu/venvs/gree-ocr-trt/lib/python3.12/site-packages/rapidocr/models/ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx 995K
```

已在不安装 TensorRT 的前提下，先用 `RapidOCR + PP-OCRv5 mobile + ONNXRuntime CPU` 跑通三张固定样本。输出文件：

```text
results/ocr_trt_poc/env_rapidocr_onnxruntime_cpu.json
results/ocr_trt_poc/rapidocr_ppocrv5_onnxruntime_cpu_baseline.json
```

耗时摘要：

```text
init_elapsed_seconds: 0.115
600004075219_1.jpg: wall 1.517s, rapidocr 1.384s, 46 text boxes
600004075219_2.jpg: wall 1.689s, rapidocr 1.584s, 62 text boxes
600004075219_7.jpg: wall 1.488s, rapidocr 1.393s, 60 text boxes
```

识别质量初步观察：

- `600004075219_1.jpg` 能识别 `GREE`、`SPLITAIR CONDITIONER INDOOR UNIT`、`GWH18AAD-K6DNA2E/I`、`Rated Voltage`、`220-240V~` 等关键文本。
- `600004075219_2.jpg` 和 `_7.jpg` 中存在字段粘连和大小写误识别，例如 `WeiGht`、`4.60kWjWeight`、`46dB(A) Serial No.`。
- 这说明 RapidOCR 输出可用，但替换主流程前仍需要字段级后处理和准确率对比，不能只按是否有文字框判断。

### 2026-04-29 主流程预处理后图像基线

注意：主流程不是直接把原始实拍图送入 OCR，而是先执行：

```text
PDF 模板解析 -> 模板裁剪
实拍图 -> 标签四角点检测 -> 透视矫正 -> target_preprocessed.png
```

因此又用当前项目预处理函数生成了真实 OCR 输入：

```text
results/ocr_trt_poc/preprocessed_check/template_preprocess/template_preprocessed.png
results/ocr_trt_poc/preprocessed_check/600004075219_1/target_preprocessed.png
results/ocr_trt_poc/preprocessed_check/600004075219_2/target_preprocessed.png
results/ocr_trt_poc/preprocessed_check/600004075219_7/target_preprocessed.png
```

RapidOCR 在这些预处理后图像上的输出：

```text
template_preprocessed.png: wall 0.816s, 33 text boxes
600004075219_1 target_preprocessed.png: wall 1.139s, 37 text boxes
600004075219_2 target_preprocessed.png: wall 0.932s, 36 text boxes
600004075219_7 target_preprocessed.png: wall 1.074s, 42 text boxes
```

输出和可视化：

```text
results/ocr_trt_poc/preprocessed_check/rapidocr_visualizations/index.html
results/ocr_trt_poc/preprocessed_check/rapidocr_visualizations/rapidocr_ppocrv5_preprocessed_onnxruntime_cpu.json
```

字段抽取初步观察：

- 预处理后输入明显比原始图更接近主流程真实情况，单张 OCR 耗时约 1 秒。
- RapidOCR 输出框可以转换成现有主流程使用的 `(poly, text, confidence)` 结构。
- 现有 `extract_field_labels_from_ocr_boxes()` 可直接消费 RapidOCR 转换后的框，并能命中多数标签字段。
- 仍存在字段分裂或粘连，例如 `AIR CONDITIONERINDOOR`、`220-240V~Heating`、`SerialNo.`、`WeiGht`。
- 规则抽取在 RapidOCR 文本上能识别核心字段，例如型号、电压、频率、制热/制冷量、重量、噪音、日期、条码；但 `air_volume` 在这批文本中未被规则抽取出来，后续需要依赖 LLM 图文抽取或增强规则。

替换判断：

- 现在可以进入“可配置 OCR 后端”接入验证。
- 不建议直接把默认 OCR 从 PaddleOCR 切到 RapidOCR。
- 需要先以 `OCR_BACKEND=rapidocr-*` 旁路模式跑完整批量回归，确认文字字段和差异框不退化。

### 2026-04-29 可选后端接入和完整流程验证

已按“默认不变、RapidOCR 显式启用”的方式接入主流程：

```text
默认: OCR_BACKEND=paddleocr
可选: OCR_BACKEND=rapidocr-onnxruntime-cpu
预留: OCR_BACKEND=rapidocr-tensorrt
```

实现边界：

- 项目 `.venv` 未安装 RapidOCR / ONNXRuntime。
- RapidOCR 仍在隔离环境 `/home/jnu/venvs/gree-ocr-trt` 中。
- 主流程通过 `RAPIDOCR_PYTHON=/home/jnu/venvs/gree-ocr-trt/bin/python` 调用 sidecar 子进程。
- RapidOCR 输出被转换为现有 `(poly, text, confidence)` OCR 框结构，后续字段抽取、框匹配、图形比对流程不变。
- `final_result.json` 中新增 `text_detection.ocr_backend`，用于追溯实际 OCR 后端。

已执行验证：

```bash
.venv/bin/python -m pytest tests/test_ocr_service.py tests/test_unified_output_mode.py
```

结果：

```text
4 passed
```

完整流程命令：

```bash
env \
  OCR_BACKEND=rapidocr-onnxruntime-cpu \
  RAPIDOCR_PYTHON=/home/jnu/venvs/gree-ocr-trt/bin/python \
  PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
  NO_PROXY=localhost,127.0.0.1,::1 \
  no_proxy=localhost,127.0.0.1,::1 \
  .venv/bin/python main.py \
    --template samples/pdfs/600004075219.pdf \
    --target samples/images/real/600004075219_7.jpg \
    --output-dir results/rapidocr_full_run_600004075219_7 \
    --output-mode debug
```

完整流程结果：

```text
real time: 42.81s
OCR backend: rapidocr-onnxruntime-cpu
OCR text boxes: template 33, target 42
text fields: 22/22 matched
graphic comparison: 4 matched, 1 mismatched, 0 needs review
verdict: 文字一致，图形存在差异 (1 处不匹配, 0 个未恢复)
```

输出目录：

```text
results/rapidocr_full_run_600004075219_7/final_result.json
results/rapidocr_full_run_600004075219_7/visualization_diff.jpg
results/rapidocr_full_run_600004075219_7/debug/text_comparison.xlsx
results/rapidocr_full_run_600004075219_7/debug/preprocess/
results/rapidocr_full_run_600004075219_7/debug/graphic_comparison/
```

### 2026-04-29 real 样本批量验证

已用 RapidOCR 可选后端跑完整 `samples/images/real` 批量样本。命令：

```bash
/usr/bin/time -p env \
  OCR_BACKEND=rapidocr-onnxruntime-cpu \
  RAPIDOCR_PYTHON=/home/jnu/venvs/gree-ocr-trt/bin/python \
  PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
  NO_PROXY=localhost,127.0.0.1,::1 \
  no_proxy=localhost,127.0.0.1,::1 \
  .venv/bin/python scripts/batch_run_samples.py \
    --template-root samples/pdfs \
    --target-root samples/images/real \
    --output-dir results/rapidocr_real_batch \
    --pair-mode best-template
```

批量范围：

```text
templates: samples/pdfs
targets: samples/images/real
matched codes: 600004075219, 600004078454
planned cases: 17
success: 17
failure: 0
```

时间：

```text
real 617.69s
user 1122.82s
sys 161.13s
average case duration from summary: 36.31s
min case duration: 32.68s
max case duration: 40.50s
```

输出：

```text
results/rapidocr_real_batch/summary.json
results/rapidocr_real_batch/<code>/<case_id>/result.json
results/rapidocr_real_batch/<code>/<case_id>/visualization_diff.jpg
```

文字匹配概览：

```text
600004075219_1: 20/22
600004075219_2: 20/22
600004075219_3: 20/22
600004075219_4: 21/22
600004075219_5: 20/22
600004075219_6: 19/22
600004075219_7: 22/22
600004075219_8: 21/22
600004075219_9: 21/22
600004078454_1: 21/23
600004078454_2: 20/23
600004078454_3: 21/23
600004078454_4: 20/23
600004078454_5: 19/23
600004078454_6: 19/23
600004078454_7: 20/23
600004078454_8: 20/23
```

观察：

- RapidOCR 后端能稳定跑完整批量流程，无运行失败。
- 主要文字差异集中在 OCR 粘连、错字和单位符号，例如 `air_volume`、`mfg_date`、`voltage`、`weight` 及若干 label 字段。
- 图形侧每个 case 至少有 1 处不匹配，原因来自当前图形区域检测/恢复逻辑，不完全由 OCR 后端决定。

### 2026-04-29 TensorRT 用户级安装与验证

由于当前 shell 无法免密 sudo：

```text
sudo -n true
sudo: a password is required
```

本次没有执行系统 apt 安装，改为在隔离 RapidOCR venv 内安装 TensorRT Python wheel。这样不会修改系统 `/usr` 下的 TensorRT 包，也不会污染项目 `.venv`。

安装目标：

```text
Python env: /home/jnu/venvs/gree-ocr-trt
engine dir: /home/jnu/models/ocr/tensorrt-engines/gb10
```

已安装关键包：

```text
rapidocr               3.8.1
onnxruntime            1.25.1
tensorrt               10.16.1.11
tensorrt-cu13          10.16.1.11
tensorrt-cu13-bindings 10.16.1.11
tensorrt-cu13-libs     10.16.1.11
cuda-python            13.2.0
cuda-bindings          13.2.0
numpy                  2.4.4
opencv-python          4.13.0.92
```

TensorRT / CUDA 验证：

```text
tensorrt 10.16.1.11
cudaGetDeviceCount: cudaSuccess, 1
device: NVIDIA GB10
compute capability: 12.1
```

生成的 TensorRT engine：

```text
/home/jnu/models/ocr/tensorrt-engines/gb10/ch_PP-OCRv5_det_mobile_sm121_fp16.engine  9.9M
/home/jnu/models/ocr/tensorrt-engines/gb10/ch_PP-OCRv5_rec_mobile_sm121_fp16.engine   11M
```

兼容性问题：

- `Det + Cls + Rec` 全部启用 TensorRT 时，RapidOCR 默认 cls TensorRT profile 与 PP-OCRv5 cls 模型输入高度不一致。
- 报错核心是 cls profile 高度为 `48`，但模型输入需要 `80`。
- 当前默认关闭 `Cls`：`Det=TensorRT`、`Rec=TensorRT`、`Cls=disabled`。
- 关闭方式不只是设置 `Global.use_cls=false`；由于 RapidOCR 初始化阶段仍会构造 `TextClassifier`，项目 sidecar 默认将 `TextClassifier` 替换为空实现，避免加载 cls 模型/session。
- 如后续确实需要方向分类，可显式设置 `RAPIDOCR_USE_CLS=1`；在 `rapidocr-tensorrt` 下仍会让 cls 使用 ONNXRuntime，规避 cls TensorRT engine 构建失败。

主流程接入方式已更新：

```bash
env \
  OCR_BACKEND=rapidocr-tensorrt \
  RAPIDOCR_PYTHON=/home/jnu/venvs/gree-ocr-trt/bin/python \
  RAPIDOCR_TRT_CACHE_DIR=/home/jnu/models/ocr/tensorrt-engines/gb10 \
  RAPIDOCR_MODEL_TYPE=mobile \
  .venv/bin/python main.py \
    --template samples/pdfs/600004075219.pdf \
    --target samples/images/real/600004075219_7.jpg \
    --output-dir results/rapidocr_tensorrt_full_run_600004075219_7 \
    --output-mode debug
```

完整流程验证结果：

```text
real time: 43.03s
OCR backend: rapidocr-tensorrt
OCR text time in workflow: 25.37s
text fields: 22/22 matched
graphic comparison: 4 matched, 1 mismatched, 0 needs review
verdict: 文字一致，图形存在差异 (1 处不匹配, 0 个未恢复)
```

输出：

```text
results/rapidocr_tensorrt_full_run_600004075219_7/final_result.json
results/rapidocr_tensorrt_full_run_600004075219_7/visualization_diff.jpg
results/rapidocr_tensorrt_full_run_600004075219_7/debug/text_comparison.xlsx
```

直接 OCR 探针结果：

```text
preprocessed target: 600004075219_7
RapidOCR init: 0.571s
single OCR wall time: 0.225s
RapidOCR internal elapse: 0.113s
recognized boxes: 42
```

说明：

- 当前 `rapidocr-tensorrt` 确认已经使用 GPU 构建并运行 TensorRT engine。
- 完整流程总耗时没有明显下降，是因为当前主流程仍包含 PP-DocLayout CPU、LLM/VLM 调用、图形比对等步骤，OCR 不再是唯一耗时项。
- 系统 apt TensorRT 仍未安装；这次是用户级、隔离 venv 安装。
- 已跑 `samples/images/real` 的 17 张 TensorRT 批量回归。
- 未修改项目 `.venv`。
- 未把 RapidOCR 设为默认 OCR 后端。

### 2026-04-30 TensorRT real 样本批量验证

已用 `rapidocr-tensorrt` 跑完整 `samples/images/real` 批量样本。命令：

```bash
/usr/bin/time -p env \
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
    --output-dir results/rapidocr_tensorrt_real_batch \
    --pair-mode best-template
```

批量范围：

```text
templates: samples/pdfs
targets: samples/images/real
matched codes: 600004075219, 600004078454
planned cases: 17
success: 17
failure: 0
```

时间：

```text
real 620.10s
user 694.44s
sys 145.76s
case duration sum from summary: 619.70s
average case duration from summary: 36.45s
min case duration: 32.74s
max case duration: 43.53s
text_detection time average: 20.05s
```

输出：

```text
results/rapidocr_tensorrt_real_batch/summary.json
results/rapidocr_tensorrt_real_batch/<code>/<case_id>/result.json
results/rapidocr_tensorrt_real_batch/<code>/<case_id>/visualization_diff.jpg
```

文字匹配概览：

```text
600004075219_1: 20/22
600004075219_2: 20/22
600004075219_3: 20/22
600004075219_4: 21/22
600004075219_5: 20/22
600004075219_6: 19/22
600004075219_7: 22/22
600004075219_8: 21/22
600004075219_9: 21/22
600004078454_1: 21/23
600004078454_2: 21/23
600004078454_3: 22/23
600004078454_4: 21/23
600004078454_5: 20/23
600004078454_6: 20/23
600004078454_7: 21/23
600004078454_8: 21/23
```

与 `rapidocr-onnxruntime-cpu` 批量基线对比：

```text
TRT total text match:  351/382 = 91.88%
ONNX total text match: 344/382 = 90.05%
TRT text fields delta: +7
TRT wall time:  620.10s
ONNX wall time: 617.69s
```

观察：

- TensorRT 批量稳定性通过：17 个 case 全部成功。
- `600004075219` 组的字段匹配与 ONNX CPU 基本一致。
- `600004078454` 组相对 ONNX CPU 多匹配 7 个字段；这可能来自 TensorRT 后端输出差异，也可能受到 LLM 抽取非完全确定性的影响，需要固定抽取策略后再下最终结论。
- 总耗时没有改善，原因是当前统计中的 `text_detection time` 包含 OCR 子进程启动、OCR、LLM 结构化抽取；完整流程还包含 PP-DocLayout CPU 和 VLM 图形判定。
- 直接 OCR 探针已经证明 det/rec TensorRT 本身很快，但主流程现在没有以常驻 OCR service 形式复用 engine。
- 每个 case 仍有 1 处图形不匹配，和 OCR 后端关系不大，主要来自当前图形区域检测/恢复判定。

## 回滚边界

### Python POC 回滚

如果只执行到独立 venv 和模型下载，回滚为：

```bash
rm -rf /home/jnu/venvs/gree-ocr-trt
rm -rf /home/jnu/models/ocr/ppocrv5
rm -rf /home/jnu/models/ocr/tensorrt-engines/gb10
rm -rf /home/jnu/projects/gree-label-detection/results/ocr_trt_poc
```

不会影响：

- 当前项目 `.venv`
- 桌面端
- MVS 相机配置
- Ollama
- PaddleOCR CPU 路径

### TensorRT apt 回滚

如果安装了系统 TensorRT，需要先记录安装包列表，再按包名回滚。安装前后建议保存：

```bash
dpkg -l | rg 'tensorrt|nvinfer|nvonnxparser|nvidia-cudnn' > results/ocr_trt_poc/dpkg_tensorrt_before.txt
dpkg -l | rg 'tensorrt|nvinfer|nvonnxparser|nvidia-cudnn' > results/ocr_trt_poc/dpkg_tensorrt_after.txt
```

卸载必须谨慎，不应盲目 `autoremove`，避免误删 CUDA 组件。

## 当前状态

已完成：

- 确认 PaddleOCR 当前只能 CPU。
- 确认 `onnxruntime-gpu` 无当前平台 wheel。
- 确认 `onnxruntime` CPU 可安装。
- 确认 PyTorch CUDA 13 nightly 可解析到 `aarch64` wheel。
- 确认 RapidOCR 可安装。
- 确认系统 apt 源中存在 `arm64` TensorRT 包。
- 明确 POC 隔离目录和回滚边界。
- 已创建 `/home/jnu/venvs/gree-ocr-trt`。
- 已在隔离环境安装 RapidOCR 和 ONNXRuntime CPU。
- 已下载并加载 PP-OCRv5 mobile ONNX 模型。
- 已对 3 张固定样本完成 CPU 基线 benchmark。
- 已接入可选 OCR 后端 `rapidocr-onnxruntime-cpu` 和 `rapidocr-tensorrt`。
- 已用 `rapidocr-onnxruntime-cpu` 跑完整单图流程和 `samples/images/real` 批量样本。
- 已在隔离环境安装 TensorRT Python wheel。
- 已生成 PP-OCRv5 mobile det/rec TensorRT FP16 engine。
- 已验证 `rapidocr-tensorrt` 单图完整流程可运行。
- 已用 `rapidocr-tensorrt` 跑完整 `samples/images/real` 批量样本，17/17 成功。

未执行：

- 未安装系统 TensorRT。
- 未解决 cls TensorRT profile 高度不匹配问题；当前默认关闭 cls，如显式打开则 cls 仍使用 ONNXRuntime。
- 未验证 PP-OCRv5 server TensorRT 精度和耗时。

下一步建议：

1. 对比 PaddleOCR CPU、RapidOCR ONNX CPU、RapidOCR TensorRT 三组字段级差异。
2. 如果 PP-OCRv5 mobile 精度仍明显低于 PaddleOCR，再验证 PP-OCRv5 server + TensorRT。
3. 若目标是提速，需要把 RapidOCR sidecar 改成常驻服务，并进一步处理 PP-DocLayout CPU 和 VLM 耗时。
4. 只有当需要系统级 TensorRT runtime 时，再单独执行 apt 安装并记录 `dpkg` 包状态。
