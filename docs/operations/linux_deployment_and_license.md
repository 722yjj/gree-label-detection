# Linux 部署包与离线授权

第一版交付包固定为 Linux 本机部署，目录结构如下：

```text
gree-label-detection/
├── app/                 # 项目代码
├── venvs/app/           # 项目 Python 环境
├── venvs/vllm/          # vLLM Python 环境
├── models/              # 随包模型权重
├── config/deploy.env    # 部署路径和服务参数
├── licenses/license.json
└── bin/
    ├── install.sh
    ├── check.sh
    └── start_desktop.sh
```

当前 checkout 也兼容这些入口：

```bash
bin/check.sh
bin/start_desktop.sh --check
```

## 本次实现范围

本次改动把当前服务器上的可运行项目，推进到“可迁移交付包”的第一版骨架：

- 新增 `bin/install.sh`、`bin/check.sh`、`bin/start_desktop.sh` 作为交付版固定入口。
- 新增 `config/deploy.env`，统一保存部署根目录、应用目录、结果目录、授权文件、vLLM 环境、模型路径和模型服务参数。
- `scripts/start_desktop.sh`、`scripts/start_vllm.sh` 和 Python 配置都会读取部署配置，不再只能依赖当前 checkout 的路径。
- 新增 `label_detection.license` 模块，提供机器码采集、离线许可证验签和命令行工具。
- 新增 `scripts/issue_license.py`，用于在签发机用 Ed25519 私钥签发试用/正式授权。
- 新增 `scripts/collect_target_machine.py`，用于客户机硬件/系统采集和迁移可行性判断。
- 桌面端、CLI、批处理入口都增加授权检查；授权无效时不允许运行检测。
- 桌面端在授权无效时仍可打开，只显示机器码、授权状态、到期时间或失败原因、授权文件路径，方便客户反馈机器码。
- `licenses/license.json` 已加入 `.gitignore`，避免把客户授权文件提交进仓库。

## 目标机采集

先在客户 Linux 机器运行：

```bash
.venv/bin/python scripts/collect_target_machine.py > target-machine.json
```

其中 `machine_fingerprint` 用于签发离线许可证；系统版本、CPU 架构、GPU/驱动/CUDA、磁盘、内存、摄像头和 USB 输入设备信息用于决定是否能复用同一部署包。

采集工具不会安装依赖、不会复制项目、不会修改目标机配置。它只读取系统文件和执行只读命令，例如：

- `/etc/os-release`
- `uname`
- `nvidia-smi`
- `nvcc --version`
- `lscpu`
- `free -h`
- `lsblk`
- `lsusb`
- `/dev/input`、`/dev/video*`

## 机器验证原理

授权绑定的核心是 `machine_fingerprint`。它不是直接保存硬件明文，而是把多个本机稳定字段清洗、排序、序列化后做 SHA-256 哈希，再截取前 128 bit 作为显示机器码。

当前采集字段包括：

| 字段 | 来源 | 说明 |
|------|------|------|
| `machine_id` | `/etc/machine-id` | Linux 系统安装级 ID，重装系统可能变化。 |
| `dmi_product_uuid` | `/sys/class/dmi/id/product_uuid` | 主机/主板 UUID，部分机器可能缺失或是占位值。 |
| `dmi_board_serial` | `/sys/class/dmi/id/board_serial` | 主板序列号，部分厂商会写占位值。 |
| `dmi_product_serial` | `/sys/class/dmi/id/product_serial` | 整机序列号，部分机器可能缺失。 |
| `mac_addresses` | `/sys/class/net/*/address` | 只取物理网卡 MAC，排除 `lo` 和明显虚拟网卡。 |
| `root_disk_serial` | `findmnt` + `lsblk` | 根盘所在磁盘序列号，换盘可能变化。 |
| `platform_machine` | `platform.machine()` | CPU 架构，例如 `x86_64`、`aarch64`。 |
| `processor` | `platform.processor()` | CPU/平台描述，作为辅助字段。 |
| `hostname` | `socket.gethostname()` | 只在详情里展示，不参与哈希，避免改主机名导致授权失效。 |

生成规则：

1. 读取本机字段。
2. 去掉空值、常见占位值，例如 `unknown`、`default string`、全零 UUID。
3. 排除 `hostname`，因为主机名太容易变。
4. 要求至少 3 个可信字段可用；不足时不生成有效机器码。
5. 按字段名排序，生成稳定 JSON。
6. 对 JSON 做 SHA-256。
7. 取前 32 个十六进制字符，按 8 位分组显示，例如：

```text
F681F80E-2C33864C-93608FBC-2D0B68A9
```

这种方式的目的不是绝对防破解，而是满足商业试用控制：

- 不把完整硬件信息写进授权文件。
- 单个字段变化时能通过重新签发解决。
- 多字段组合比单独绑定 MAC 更稳。
- Python 代码仍可被逆向；如果后续要提高绕过成本，可以把授权核心迁移到 native 扩展。

## 授权签发

客户机打印机器码：

```bash
.venv/bin/python -m label_detection.license fingerprint
```

签发 30 天试用授权：

```bash
LICENSE_PRIVATE_KEY_PEM="$(cat /secure/path/license_ed25519_private.pem)" \
.venv/bin/python scripts/issue_license.py \
  --machine <fingerprint> \
  --days 30 \
  --customer <name> \
  --output licenses/license.json
```

验证授权：

```bash
.venv/bin/python -m label_detection.license verify --license licenses/license.json
```

软件只内置公钥。私钥只保存在签发机，不进入客户部署包。

签发和验证逻辑：

1. 客户机生成机器码。
2. 签发机创建不含 `signature` 的许可证 JSON。
3. 签发机按字段名排序，把许可证 JSON 做规范化序列化。
4. 用 Ed25519 私钥对规范化内容签名。
5. 把 Base64 签名写入 `signature` 字段。
6. 客户机软件用内置公钥验签。
7. 验签通过后，再检查机器码是否一致、授权是否过期、版本是否支持。

验签覆盖除 `signature` 之外的所有许可证字段，因此客户修改 `customer`、`expires_at`、`features`、`machine_fingerprint` 等任意字段都会导致签名无效。

## 运行检查

交付机上执行：

```bash
bin/check.sh && bin/start_desktop.sh --check
```

授权无效或过期时，检测入口会拒绝运行。桌面端仍会打开，并显示机器码、授权状态、到期时间或失败原因、授权文件路径。

当前做了两层检查：

- 启动脚本层：`bin/start_desktop.sh` / `scripts/start_desktop.sh` 会先运行 `python -m label_detection.license verify`。`--check` 模式下授权失败直接返回失败；正常桌面启动时授权失败会跳过模型服务自启，让 UI 尽快显示机器码。
- Python 业务层：CLI 主流程、批处理、桌面检测服务在真正运行检测前再次调用 `require_valid_license()`，避免绕过启动脚本直接调用 Python 入口。

## 许可证字段

`license.json` 包含：

- `license_id`
- `customer`
- `machine_fingerprint`
- `issued_at`
- `expires_at`
- `features`
- `version`
- `signature`

机器码由 `/etc/machine-id`、DMI UUID/序列号、物理网卡 MAC、根磁盘序列、CPU/平台信息等字段组合后哈希生成。字段允许缺失，但最少需要 3 个可信字段。

示例结构：

```json
{
  "version": 1,
  "license_id": "trial-xxxxxxxxxxxx",
  "customer": "customer-name",
  "machine_fingerprint": "F681F80E-2C33864C-93608FBC-2D0B68A9",
  "issued_at": "2026-05-20T00:00:00+00:00",
  "expires_at": "2026-06-19T00:00:00+00:00",
  "features": ["desktop", "cli", "batch"],
  "signature": "<base64-ed25519-signature>"
}
```

## 迁移判断

目标机能否“一整个文件夹直接跑”，取决于 CPU 架构、系统 ABI、GPU 驱动/CUDA、Python 和二进制依赖是否匹配。

当前开发机是：

```text
Ubuntu 24.04 + aarch64 + NVIDIA GB10 + CUDA 13.0 + Python 3.12
```

已检查的 4090 机器是：

```text
Ubuntu 22.04.5 + x86_64 + RTX 4090 24GB + Driver CUDA 12.4 + Python 3.10
```

因此当前开发机的 `.venv` 和 `/home/jnu/venvs/vllm` 不能原样搬到 4090 使用。可复用的是代码、模型、配置模板、授权逻辑和部署目录结构；`venvs/app` 与 `venvs/vllm` 需要在 4090 上按 `x86_64 + CUDA 12.4` 重建。

对 4090 这类异构目标机，推荐使用 Docker Compose 交付：

```text
gree-label-detection-deploy/
├── compose.yaml
├── .env
├── licenses/license.json
├── models/
├── results/
└── app-data/
```

4090 已确认 Docker、NVIDIA runtime、`docker run --gpus all ... nvidia-smi` 可用，因此 Docker 方案比复制 venv 更稳。Docker 仍然依赖宿主机 NVIDIA 驱动和 `nvidia-container-runtime`，但可以固定 Python、CUDA 用户态、PyTorch/Paddle/vLLM、OpenCV/Qt 等运行环境。

## 合规注意

闭源商业交付前需要确认第三方许可证路径，尤其是 PySide6/Qt 的 LGPLv3 动态链接合规或商业授权。还需随包保留 Qwen、vLLM、PaddlePaddle 等依赖的许可证清单。

详细清单见 `docs/operations/third_party_license_checklist.md`。
