# 桌面端本机一键启动

本实验只处理当前 Linux 本机的启动体验，不改 OCR、VLM、布局检测和统一检测主流程。

## 命令行一键启动

在项目根目录运行：

```bash
scripts/start_desktop.sh
```

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
- Ollama 服务是否可连接

启动脚本会在检查前尝试自动启动本机 Ollama：

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

Ollama 启动日志写入：

```text
results/desktop_app/ollama.log
```
