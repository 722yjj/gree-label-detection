"""
统一配置模块

集中管理项目中各模块共用的配置项，避免散落在各文件中的硬编码。

STATUS: main
"""

import os

# ==================== Ollama / VLM 配置 ====================
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
# OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3-vl:8b")

# VLM 请求超时（秒）
VLM_TIMEOUT = int(os.getenv("VLM_TIMEOUT", "120"))

# VLM / LLM 最大生成 token 数
VLM_NUM_PREDICT = int(os.getenv("VLM_NUM_PREDICT", "8192"))

# ==================== OCR 配置 ====================
OCR_LANG = os.getenv("OCR_LANG", "ch")
OCR_USE_ANGLE_CLS = True

# ==================== 布局检测配置 ====================
LAYOUT_DETECTION_THRESHOLD = float(os.getenv("LAYOUT_DETECTION_THRESHOLD", "0.3"))
LAYOUT_MODEL_NAME = "PP-DocLayoutV3"

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
DEFAULT_OUTPUT_DIR = os.getenv("DEFAULT_OUTPUT_DIR", "results/unified")

# ==================== 图形比对配置 ====================
# 是否默认启用 VLM 进行图形比对
USE_VLM_FOR_GRAPHIC = True

# VLM 输入标准化画布尺寸
VLM_CANVAS_SIZE = int(os.getenv("VLM_CANVAS_SIZE", "512"))

# VLM 请求重试次数
VLM_MAX_RETRIES = int(os.getenv("VLM_MAX_RETRIES", "2"))

# ==================== 样例文件 ====================
DEFAULT_PDF_PATH = "600004075219-01.pdf"
DEFAULT_TARGET_PATH = "test.jpg"
