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
~/.local/share/applications/gree-label-detection.desktop
~/Desktop/gree-label-detection.desktop
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

Ollama 默认只作为警告项。这样桌面端仍可先启动，实际检测时如果服务不可用，再由业务流程报错。
