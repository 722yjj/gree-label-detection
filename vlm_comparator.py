"""
基于 Qwen3-VL 的视觉大模型图形对比模块

使用 Ollama API 调用模型。
核心实现已迁移至 services/vlm_service.py，本文件保留为兼容接口。

STATUS: main
"""

# 直接从服务模块导入，保持向后兼容
from services.vlm_service import (
    VLMComparator,
    get_vlm_comparator,
    compare_with_vlm,
)

import json
import cv2
import sys


# ============ 测试代码 ============
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python vlm_comparator.py <模板图片> <目标图片>")
        print("示例: python vlm_comparator.py template.jpg target.jpg")
        sys.exit(1)

    template_path = sys.argv[1]
    target_path = sys.argv[2]

    # 加载图片
    template = cv2.imread(template_path)
    target = cv2.imread(target_path)

    if template is None or target is None:
        print("错误: 无法读取图片")
        sys.exit(1)

    # 对比
    print("正在对比图片...")
    comparator = VLMComparator()
    result = comparator.compare_images(template, target)

    # 输出结果
    print("\n" + "=" * 50)
    print("对比结果:")
    print("=" * 50)
    print(json.dumps(result, ensure_ascii=False, indent=2))
