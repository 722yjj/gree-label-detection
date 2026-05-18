# 桌面端本机一键启动

本实验只处理当前 Linux 本机的启动体验，不改 OCR、VLM、布局检测和统一检测主流程。

## 命令行一键启动

在项目根目录运行：

```bash
scripts/start_desktop.sh
```

桌面图标默认会以 `LLM_PROVIDER=vllm` 启动，启动桌面端前会先调用：

```bash
scripts/start_vllm.sh
```

这样会自动检查并启动当前项目约定的本机 vLLM 服务，然后把 `LLM_PROVIDER`、`VLLM_API_BASE`、
`OPENAI_COMPATIBLE_MODEL` 等变量导入桌面端进程。vLLM 的默认参数和常用命令见
`docs/vllm_startup.md`。

关闭桌面端后，脚本会先等待桌面主进程退出，再延迟 30 分钟停止本机 vLLM。
如果这段时间内重新打开桌面端，旧的延迟停止计划会被取消并重新计时。
可用 `VLLM_DESKTOP_SHUTDOWN_DELAY=秒数` 调整这个延迟。

只做启动前检查：

```bash
scripts/start_desktop.sh --check
```

如果希望把 Ollama 不可访问也视为启动失败：

```bash
scripts/start_desktop.sh --check --strict-services
```

启动日志写入：

```text
results/desktop_app/startup.log
```

## 安装桌面图标

在项目根目录运行：

```bash
scripts/install_desktop_launcher.sh
```

脚本会基于当前 checkout 路径生成：

```text
~/.local/share/applications/label-detection.desktop
~/Desktop/label-detection.desktop
```

如果当前系统没有 `~/Desktop` 目录，只安装应用菜单入口。

## 环境要求

启动脚本使用项目虚拟环境：

```text
.venv/bin/python
```

如果 `.venv` 不存在，先在项目根目录运行：

```bash
uv sync --extra desktop
```

检查脚本会验证：

- 当前解释器是否为 `.venv/bin/python`
- `desktop_app` 和 `label_detection` 是否可导入
- PySide6、OpenCV、PaddleOCR/PaddlePaddle 或 RapidOCR 后端是否可用
- `results/desktop_app` 是否可写
- 当前 `LLM_PROVIDER` 对应的 Ollama 或 OpenAI-compatible/vLLM 服务是否可连接

如果 `LLM_PROVIDER=vllm`，启动脚本会在检查前调用：

```bash
scripts/start_vllm.sh
```

如果只是执行 `scripts/start_desktop.sh --check`，脚本只运行 `scripts/start_vllm.sh check`，
不会加载大模型。

如果 `LLM_PROVIDER` 保持默认 `ollama`，启动脚本会在检查前尝试自动启动本机 Ollama：

```bash
ollama serve
```

自动启动只针对 `OLLAMA_API_BASE` 为本机地址的情况，例如 `http://localhost:11434` 或 `http://127.0.0.1:11434`。如果系统桌面环境的 `PATH` 不包含 Ollama，脚本会优先查找常见路径，例如 `~/.local/bin/ollama`。

可用环境变量：

```bash
OLLAMA_AUTOSTART=0 scripts/start_desktop.sh
OLLAMA_BIN=/custom/path/ollama scripts/start_desktop.sh
OLLAMA_START_TIMEOUT=30 scripts/start_desktop.sh
```

vLLM 自动启动可用环境变量：

```bash
VLLM_AUTOSTART=0 LLM_PROVIDER=vllm scripts/start_desktop.sh
VLLM_MAX_MODEL_LEN=32768 LLM_PROVIDER=vllm scripts/start_desktop.sh
VLLM_GPU_MEMORY_UTILIZATION=0.35 LLM_PROVIDER=vllm scripts/start_desktop.sh
VLLM_DESKTOP_SHUTDOWN_DELAY=300 LLM_PROVIDER=vllm scripts/start_desktop.sh
```

Ollama 启动日志写入：

```text
results/desktop_app/ollama.log
```

vLLM 启动日志写入：

```text
results/vllm/server.log
```

延迟关闭状态文件写入：

```text
results/vllm/desktop-session.token
results/vllm/desktop-watchdog.pid
```
