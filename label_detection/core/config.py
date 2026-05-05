"""Project-wide configuration."""

from pathlib import Path
from urllib.parse import urlparse
import os


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = PROJECT_ROOT / "samples"


def _env_flag(name: str, default: str) -> bool:
    value = os.getenv(name, default).strip().lower()
    return value not in {"", "0", "false", "no", "off"}


# ==================== Ollama / VLM 配置 ====================
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
TEXT_LLM_MODEL = os.getenv("TEXT_LLM_MODEL", OLLAMA_MODEL)
GRAPHIC_VLM_MODEL = os.getenv("GRAPHIC_VLM_MODEL", OLLAMA_MODEL)
LOCAL_OLLAMA_HOSTS = {"localhost", "127.0.0.1", "::1"}
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "10m")

# 文本结构化抽取请求超时（秒）
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))

# 文本结构化抽取最大生成 token 数
TEXT_NUM_PREDICT = int(os.getenv("TEXT_NUM_PREDICT", "1024"))

# VLM 请求超时（秒）
VLM_TIMEOUT = int(os.getenv("VLM_TIMEOUT", "120"))

# 图形 VLM 最大生成 token 数
GRAPHIC_NUM_PREDICT = int(
    os.getenv("GRAPHIC_NUM_PREDICT", os.getenv("VLM_NUM_PREDICT", "256"))
)

# 兼容旧配置名：现仅表示图形 VLM 的生成 token 上限
VLM_NUM_PREDICT = GRAPHIC_NUM_PREDICT

# ==================== OCR 配置 ====================
OCR_BACKEND = os.getenv("OCR_BACKEND", "paddleocr").strip().lower() or "paddleocr"
OCR_LANG = os.getenv("OCR_LANG", "ch")
OCR_USE_ANGLE_CLS = True
PADDLE_DEVICE = os.getenv("PADDLE_DEVICE", "gpu:0").strip() or "gpu:0"
OCR_DEVICE = os.getenv("OCR_DEVICE", PADDLE_DEVICE).strip() or PADDLE_DEVICE
RAPIDOCR_PYTHON = os.getenv("RAPIDOCR_PYTHON", "").strip()
RAPIDOCR_TIMEOUT = float(os.getenv("RAPIDOCR_TIMEOUT", "120"))
RAPIDOCR_MODEL_TYPE = os.getenv("RAPIDOCR_MODEL_TYPE", "mobile").strip().lower() or "mobile"
RAPIDOCR_USE_CLS = _env_flag("RAPIDOCR_USE_CLS", "0")
RAPIDOCR_TRT_CACHE_DIR = os.getenv(
    "RAPIDOCR_TRT_CACHE_DIR",
    str(PROJECT_ROOT.parent / "models" / "ocr" / "tensorrt-engines" / "gb10"),
).strip()
LAYOUT_DEVICE = os.getenv("LAYOUT_DEVICE", PADDLE_DEVICE).strip() or PADDLE_DEVICE
PADDLE_DEVICE_REQUIRED = _env_flag("PADDLE_DEVICE_REQUIRED", "0")
PADDLE_DISABLE_MODEL_SOURCE_CHECK = _env_flag(
    "PADDLE_DISABLE_MODEL_SOURCE_CHECK",
    "1",
)
PADDLE_EMPTY_CACHE_AFTER_RUN = _env_flag("PADDLE_EMPTY_CACHE_AFTER_RUN", "1")

# ==================== 布局检测配置 ====================
LAYOUT_BACKEND = os.getenv("LAYOUT_BACKEND", "paddlex").strip().lower() or "paddlex"
LAYOUT_DETECTION_THRESHOLD = float(os.getenv("LAYOUT_DETECTION_THRESHOLD", "0.3"))
LAYOUT_MODEL_NAME = "PP-DocLayoutV3"
HF_LAYOUT_PYTHON = os.getenv(
    "HF_LAYOUT_PYTHON",
    "/home/jnu/venvs/gree-layout-hf-gpu/bin/python",
).strip()
HF_LAYOUT_MODEL_ID = os.getenv(
    "HF_LAYOUT_MODEL_ID",
    "PaddlePaddle/PP-DocLayoutV3_safetensors",
).strip()
HF_LAYOUT_DEVICE = os.getenv("HF_LAYOUT_DEVICE", "cuda").strip() or "cuda"
HF_LAYOUT_TIMEOUT = float(os.getenv("HF_LAYOUT_TIMEOUT", "120"))
HF_LAYOUT_HF_HOME = os.getenv(
    "HF_LAYOUT_HF_HOME",
    str(Path.home() / "models" / "layout" / "hf_home"),
).strip()

# ==================== 区域匹配配置 ====================
REGION_MATCH_DISTANCE_THRESHOLD = 0.2

# 区域匹配代价各因子权重（用于匈牙利算法多因子匹配）
MATCH_WEIGHT_CENTER = float(os.getenv("MATCH_WEIGHT_CENTER", "0.35"))
MATCH_WEIGHT_AREA = float(os.getenv("MATCH_WEIGHT_AREA", "0.25"))
MATCH_WEIGHT_ASPECT = float(os.getenv("MATCH_WEIGHT_ASPECT", "0.20"))
MATCH_WEIGHT_IOU = float(os.getenv("MATCH_WEIGHT_IOU", "0.20"))
MATCH_COST_THRESHOLD = float(os.getenv("MATCH_COST_THRESHOLD", "0.6"))

# ==================== 重试配置 ====================
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))

# ==================== 输出目录 ====================
DEFAULT_OUTPUT_DIR = os.getenv(
    "DEFAULT_OUTPUT_DIR",
    str(PROJECT_ROOT / "results" / "unified"),
)

# ==================== 图形比对配置 ====================
# 是否默认启用 VLM 进行图形比对
USE_VLM_FOR_GRAPHIC = True

# 是否对版面模型的 image 大区域做二次拆分（默认开启，可用环境变量显式关闭）
ENABLE_IMAGE_REGION_SPLIT = os.getenv("ENABLE_IMAGE_REGION_SPLIT", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}

# VLM 输入标准化画布尺寸
VLM_CANVAS_SIZE = int(os.getenv("VLM_CANVAS_SIZE", "512"))

# VLM 请求重试次数
VLM_MAX_RETRIES = int(os.getenv("VLM_MAX_RETRIES", "2"))

# ==================== 样例文件 ====================
DEFAULT_PDF_PATH = os.getenv(
    "DEFAULT_PDF_PATH",
    str(SAMPLES_DIR / "pdfs" / "600004075219-01.pdf"),
)
DEFAULT_TARGET_PATH = os.getenv(
    "DEFAULT_TARGET_PATH",
    str(SAMPLES_DIR / "images" / "test.jpg"),
)


def get_ollama_host(api_base: str | None = None) -> str:
    """Extract the hostname from the configured Ollama base URL."""
    parsed = urlparse((api_base or OLLAMA_API_BASE).strip())
    return (parsed.hostname or "").strip().lower()


def is_local_ollama(api_base: str | None = None) -> bool:
    """Return whether the Ollama endpoint points to the local machine."""
    return get_ollama_host(api_base) in LOCAL_OLLAMA_HOSTS


def _merge_no_proxy(existing: str | None) -> str:
    entries = []
    for item in (existing or "").split(","):
        value = item.strip()
        if value and value not in entries:
            entries.append(value)

    for host in LOCAL_OLLAMA_HOSTS:
        if host not in entries:
            entries.append(host)

    return ",".join(entries)


def ensure_local_ollama_no_proxy(api_base: str | None = None) -> bool:
    """Ensure localhost Ollama traffic bypasses any configured HTTP proxy."""
    if not is_local_ollama(api_base):
        return False

    merged = _merge_no_proxy(os.getenv("NO_PROXY") or os.getenv("no_proxy"))
    os.environ["NO_PROXY"] = merged
    os.environ["no_proxy"] = merged
    return True
