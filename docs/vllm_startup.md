# vLLM 启动说明

本文档记录当前服务器上 vLLM 的一键启动、检查、环境变量和常用调参命令。

## 当前默认配置

一键脚本：`scripts/start_vllm.sh`

| 配置项 | 默认值 |
|--------|--------|
| `VLLM_VENV` | `/home/jnu/venvs/vllm` |
| `VLLM_MODEL_PATH` | `/home/jnu/models/Qwen3.6-27B-int4-AutoRound` |
| `VLLM_SERVED_MODEL_NAME` | `qwen3.6-27b-int4` |
| `VLLM_API_BASE` | `http://127.0.0.1:8000/v1` |
| `VLLM_HOST` / `VLLM_PORT` | `0.0.0.0` / `8000` |
| `VLLM_MAX_MODEL_LEN` | `49152` |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.30` |
| `VLLM_MAX_NUM_SEQS` | `1` |
| `VLLM_MAX_NUM_BATCHED_TOKENS` | `12288` |
| `VLLM_KV_CACHE_DTYPE` | `fp8` |
| `VLLM_SPECULATIVE_CONFIG` | `{"method": "mtp", "num_speculative_tokens": 1}` |
| `VLLM_DESKTOP_SHUTDOWN_DELAY` | `1800` |

等价的 vLLM serve 命令大致是：

```bash
/home/jnu/venvs/vllm/bin/vllm serve /home/jnu/models/Qwen3.6-27B-int4-AutoRound \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name qwen3.6-27b-int4 \
  --dtype auto \
  --max-model-len 49152 \
  --gpu-memory-utilization 0.30 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 12288 \
  --kv-cache-dtype fp8 \
  --trust-remote-code \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'
```

## 检查环境

只检查 vLLM 环境、CUDA、模型目录和端口，不启动模型：

```bash
cd /home/jnu/projects/gree-label-detection
scripts/start_vllm.sh check
```

查看脚本帮助：

```bash
scripts/start_vllm.sh --help
```

## 启动和停止

后台启动 vLLM，并等待 `/v1/models` 就绪：

```bash
cd /home/jnu/projects/gree-label-detection
scripts/start_vllm.sh
```

查看状态：

```bash
scripts/start_vllm.sh status
```

查看日志：

```bash
tail -f results/vllm/server.log
```

停止由脚本启动的 vLLM 进程：

```bash
scripts/start_vllm.sh stop
```

重启：

```bash
scripts/start_vllm.sh restart
```

前台启动，适合临时排查启动报错：

```bash
scripts/start_vllm.sh foreground
```

桌面端联动的延迟关闭命令：

```bash
VLLM_DESKTOP_PID=<desktop-pid> \
VLLM_DESKTOP_SESSION_TOKEN=<token> \
scripts/start_vllm.sh delayed-stop
```

它会先等待桌面主进程退出，再按 `VLLM_DESKTOP_SHUTDOWN_DELAY` 倒计时。桌面端重新打开时会写入新的 token，
旧的延迟停止计划会跳过停止。

## 让项目使用 vLLM

在当前 shell 中导入项目需要的 LLM/VLM 环境变量：

```bash
cd /home/jnu/projects/gree-label-detection
eval "$(scripts/start_vllm.sh --print-env)"
```

等价的手动环境变量是：

```bash
export LLM_PROVIDER=vllm
export VLLM_API_BASE=http://127.0.0.1:8000/v1
export OPENAI_COMPATIBLE_API_BASE=http://127.0.0.1:8000/v1
export VLLM_API_KEY=EMPTY
export OPENAI_COMPATIBLE_API_KEY=EMPTY
export VLLM_MODEL_PATH=/home/jnu/models/Qwen3.6-27B-int4-AutoRound
export VLLM_SERVED_MODEL_NAME=qwen3.6-27b-int4
export VLLM_MODEL=qwen3.6-27b-int4
export OPENAI_COMPATIBLE_MODEL=qwen3.6-27b-int4
export TEXT_LLM_MODEL=qwen3.6-27b-int4
export GRAPHIC_VLM_MODEL=qwen3.6-27b-int4
```

启动桌面端：

```bash
.venv/bin/python -m desktop_app.main
```

运行桌面端环境检查：

```bash
LLM_PROVIDER=vllm \
VLLM_API_BASE=http://127.0.0.1:8000/v1 \
VLLM_MODEL=qwen3.6-27b-int4 \
.venv/bin/python scripts/check_desktop_env.py
```

## 批量样本测试

先启动 vLLM，再导入环境变量：

```bash
cd /home/jnu/projects/gree-label-detection
scripts/start_vllm.sh
eval "$(scripts/start_vllm.sh --print-env)"
.venv/bin/python scripts/batch_run_samples.py
```

## 临时覆盖默认参数

更省显存，降低上下文到 32768：

```bash
VLLM_MAX_MODEL_LEN=32768 \
VLLM_GPU_MEMORY_UTILIZATION=0.30 \
scripts/start_vllm.sh restart
```

只临时把显存比例改为 0.40：

```bash
VLLM_GPU_MEMORY_UTILIZATION=0.40 scripts/start_vllm.sh restart
```

如果 `0.30 + 49152` 启动时提示 KV cache 或上下文容量不足，优先尝试：

```bash
VLLM_MAX_MODEL_LEN=32768 scripts/start_vllm.sh restart
```

再考虑提高：

```bash
VLLM_GPU_MEMORY_UTILIZATION=0.35 scripts/start_vllm.sh restart
```

## 直接测试接口

查看服务模型列表：

```bash
curl http://127.0.0.1:8000/v1/models
```

发送一个最小文本请求：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer EMPTY' \
  -d '{
    "model": "qwen3.6-27b-int4",
    "messages": [
      {"role": "user", "content": "用一句话介绍你自己。"}
    ],
    "temperature": 0,
    "max_tokens": 128
  }'
```
