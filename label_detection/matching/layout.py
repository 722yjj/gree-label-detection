"""
基于 PP-DocLayoutV3 的区域检测对比脚本

流程：
1. 预处理：模板 PDF 提取 + 实拍图片裁剪（与 unified_detection.py 相同）
2. 区域检测：使用 PP-DocLayoutV3 检测 image 类型区域
3. 区域匹配：根据位置匹配模板和实拍的对应区域
4. 区域对比：对每对匹配区域进行 SSIM 相似度计算
5. 输出结果：对比结果 JSON + 可视化图片

STATUS: main
"""

import json
import os
from typing import Dict, List, Tuple

import numpy as np

from scipy.optimize import linear_sum_assignment

# 导入公共模块
from label_detection.core.config import (
    LAYOUT_DETECTION_THRESHOLD,
    MATCH_WEIGHT_CENTER,
    MATCH_WEIGHT_AREA,
    MATCH_WEIGHT_ASPECT,
    MATCH_WEIGHT_IOU,
    MATCH_COST_THRESHOLD,
)


# 创建 PP-DocLayoutV3 预测器（全局复用）
_layout_predictor = None


def _require_cv2():
    import cv2

    return cv2


def _get_ssim():
    from skimage.metrics import structural_similarity as ssim

    return ssim


def get_layout_predictor(threshold: float = None):
    """获取或创建 PP-DocLayoutV3 预测器"""
    global _layout_predictor
    if threshold is None:
        threshold = LAYOUT_DETECTION_THRESHOLD
    if _layout_predictor is None:
        from paddlex import create_predictor

        print("[模型] 初始化 PP-DocLayoutV3 预测器...")
        _layout_predictor = create_predictor(model_name="PP-DocLayoutV3", threshold=threshold)
    return _layout_predictor


def detect_layout_regions(image_path: str, threshold: float = 0.3) -> List[Dict]:
    """
    使用 PP-DocLayoutV3 检测图像区域
    
    Args:
        image_path: 图片路径
        threshold: 检测置信度阈值
        
    Returns:
        检测到的区域列表，每个区域包含 label, score, coordinate
    """
    predictor = get_layout_predictor(threshold)
    output = predictor.predict(image_path)
    
    regions = []
    for res in output:
        # PaddleX 返回的结果对象，JSON 格式为 {'res': {'boxes': [...]}}
        result_dict = res.json
        if isinstance(result_dict, str):
            result_dict = json.loads(result_dict)
        
        # 从 'res' 键中获取实际结果
        if 'res' in result_dict:
            result_dict = result_dict['res']
        
        for box in result_dict.get("boxes", []):
            regions.append({
                "label": box["label"],
                "score": box["score"],
                "coordinate": box["coordinate"],  # [x1, y1, x2, y2]
            })
    
    return regions


def extract_regions_by_type(regions: List[Dict], region_type: str = "image") -> List[Dict]:
    """
    从检测结果中筛选指定类型的区域
    
    Args:
        regions: 检测到的区域列表
        region_type: 要筛选的区域类型（image, table, text 等）
        
    Returns:
        筛选后的区域列表
    """
    return [r for r in regions if r["label"] == region_type]


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    计算两个矩形框的 IoU（Intersection over Union）
    
    Args:
        box1, box2: [x1, y1, x2, y2] 格式的坐标
        
    Returns:
        IoU 值 (0-1)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def normalize_coordinates(box: List[float], img_shape: Tuple[int, int]) -> List[float]:
    """
    将坐标归一化到 0-1 范围
    
    Args:
        box: [x1, y1, x2, y2] 坐标
        img_shape: (height, width) 图片尺寸
        
    Returns:
        归一化后的坐标
    """
    h, w = img_shape
    return [box[0]/w, box[1]/h, box[2]/w, box[3]/h]


def _compute_match_cost(
    box1: List[float],
    box2: List[float],
    img1_shape: Tuple[int, int],
    img2_shape: Tuple[int, int],
) -> float:
    """
    计算两个区域的多因子匹配代价

    综合考虑中心点距离、面积比、宽高比、IoU 四个因子。
    """
    norm1 = normalize_coordinates(box1, img1_shape)
    norm2 = normalize_coordinates(box2, img2_shape)

    # 中心点距离
    cx1, cy1 = (norm1[0] + norm1[2]) / 2, (norm1[1] + norm1[3]) / 2
    cx2, cy2 = (norm2[0] + norm2[2]) / 2, (norm2[1] + norm2[3]) / 2
    center_dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

    # 面积比差异
    area1 = (norm1[2] - norm1[0]) * (norm1[3] - norm1[1])
    area2 = (norm2[2] - norm2[0]) * (norm2[3] - norm2[1])
    if max(area1, area2) > 0:
        area_ratio_diff = abs(area1 - area2) / max(area1, area2)
    else:
        area_ratio_diff = 1.0

    # 宽高比差异
    w1, h1 = norm1[2] - norm1[0], norm1[3] - norm1[1]
    w2, h2 = norm2[2] - norm2[0], norm2[3] - norm2[1]
    ar1 = w1 / h1 if h1 > 0 else 0
    ar2 = w2 / h2 if h2 > 0 else 0
    if max(ar1, ar2) > 0:
        aspect_ratio_diff = abs(ar1 - ar2) / max(ar1, ar2)
    else:
        aspect_ratio_diff = 1.0

    # 归一化 IoU（在归一化坐标上计算）
    norm_iou = calculate_iou(norm1, norm2)

    cost = (
        MATCH_WEIGHT_CENTER * center_dist
        + MATCH_WEIGHT_AREA * area_ratio_diff
        + MATCH_WEIGHT_ASPECT * aspect_ratio_diff
        + MATCH_WEIGHT_IOU * (1 - norm_iou)
    )
    return float(cost)


def match_regions(
    regions1: List[Dict],
    regions2: List[Dict],
    img1_shape: Tuple[int, int],
    img2_shape: Tuple[int, int],
    cost_threshold: float = None,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """
    使用匈牙利算法对两组区域进行全局最优匹配

    多因子代价矩阵 = 中心点距离 + 面积比 + 宽高比 + (1-IoU)

    Args:
        regions1, regions2: 区域列表
        img1_shape, img2_shape: 图片尺寸 (height, width)
        cost_threshold: 代价阈值，超过则舍弃配对

    Returns:
        (matched_pairs, unmatched1, unmatched2)
        - matched_pairs: [(idx1, idx2, cost), ...]
        - unmatched1: 未匹配的 regions1 索引
        - unmatched2: 未匹配的 regions2 索引
    """
    if cost_threshold is None:
        cost_threshold = MATCH_COST_THRESHOLD

    if not regions1 or not regions2:
        return [], list(range(len(regions1))), list(range(len(regions2)))

    n, m = len(regions1), len(regions2)

    # 构造代价矩阵
    cost_matrix = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            cost_matrix[i, j] = _compute_match_cost(
                regions1[i]["coordinate"],
                regions2[j]["coordinate"],
                img1_shape,
                img2_shape,
            )

    # 匈牙利算法求全局最优匹配
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matched_pairs = []
    used1 = set()
    used2 = set()

    for i, j in zip(row_indices, col_indices):
        cost = cost_matrix[i, j]
        if cost < cost_threshold:
            matched_pairs.append((int(i), int(j), float(cost)))
            used1.add(int(i))
            used2.add(int(j))

    unmatched1 = [i for i in range(n) if i not in used1]
    unmatched2 = [j for j in range(m) if j not in used2]

    return matched_pairs, unmatched1, unmatched2


def align_images_sift(img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    使用 SIFT 特征点对齐两张图像
    
    Args:
        img1: 参考图像
        img2: 待对齐图像
        
    Returns:
        (aligned_img2, success): 对齐后的 img2 和是否成功
    """
    cv2 = _require_cv2()
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
    
    # SIFT 特征点检测
    try:
        detector = cv2.SIFT_create(nfeatures=1000)
    except:
        detector = cv2.ORB_create(nfeatures=1000)
    
    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)
    
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        # 特征点不足，直接缩放
        h, w = img1.shape[:2]
        return cv2.resize(img2, (w, h)), False
    
    # 特征点匹配
    if hasattr(detector, 'descriptorType') and detector.descriptorType() == cv2.CV_8U:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2)
    
    try:
        matches = matcher.knnMatch(des1, des2, k=2)
    except:
        h, w = img1.shape[:2]
        return cv2.resize(img2, (w, h)), False
    
    # 筛选好的匹配
    good_matches = []
    for match in matches:
        if len(match) == 2:
            m, n = match
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
    
    if len(good_matches) < 4:
        h, w = img1.shape[:2]
        return cv2.resize(img2, (w, h)), False
    
    # 计算单应性矩阵
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    
    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    
    if H is None:
        h, w = img1.shape[:2]
        return cv2.resize(img2, (w, h)), False
    
    h, w = img1.shape[:2]
    aligned = cv2.warpPerspective(img2, H, (w, h))
    
    return aligned, True


def extract_main_contours(img: np.ndarray) -> List[np.ndarray]:
    """
    提取图像中的主要轮廓
    
    Args:
        img: 输入图像
        
    Returns:
        主要轮廓列表
    """
    cv2 = _require_cv2()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    # 二值化
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # 形态学操作：去噪
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 提取轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 过滤小轮廓
    img_area = gray.shape[0] * gray.shape[1]
    min_area = img_area * 0.01  # 至少占图像 1%
    
    main_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    # 按面积排序
    main_contours.sort(key=cv2.contourArea, reverse=True)
    
    return main_contours


def compare_contours_hu(contour1: np.ndarray, contour2: np.ndarray) -> float:
    """
    使用 Hu 矩对比两个轮廓的形状相似度
    
    Hu 矩对平移、旋转、缩放不变
    
    Returns:
        相似度分数 (0-1)，越高越相似
    """
    cv2 = _require_cv2()
    # cv2.matchShapes 返回值越小越相似
    # 使用方法 I1（基于 Hu 矩）
    distance = cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I1, 0)
    
    # 转换为相似度分数 (0-1)
    # 使用指数衰减，distance=0 时相似度=1
    similarity = np.exp(-distance * 10)  # 可调整衰减系数
    
    return float(similarity)


def _compute_traditional_score(
    img1: np.ndarray,
    img2: np.ndarray,
    box1: List[float],
    box2: List[float],
) -> Dict:
    """
    快速计算传统方法形状分数（用于融合判定）

    只做 SIFT 对齐 + Hu 矩 + SSIM，不保存中间结果。
    """
    cv2 = _require_cv2()
    ssim = _get_ssim()
    x1, y1, x2, y2 = [int(v) for v in box1]
    crop1 = img1[y1:y2, x1:x2]
    x1, y1, x2, y2 = [int(v) for v in box2]
    crop2 = img2[y1:y2, x1:x2]

    if crop1.size == 0 or crop2.size == 0:
        return {"shape_score": 0.0}

    aligned_crop2, _ = align_images_sift(crop1, crop2)

    contours1 = extract_main_contours(crop1)
    contours2 = extract_main_contours(aligned_crop2)

    # Hu 矩分数
    hu_avg = 0.0
    if contours1 and contours2:
        n = min(3, len(contours1), len(contours2))
        hu_scores = [compare_contours_hu(contours1[i], contours2[i]) for i in range(n)]
        hu_avg = float(np.mean(hu_scores))

    # 凸包分数
    hull_score = 0.0
    if contours1 and contours2:
        hull1 = cv2.convexHull(np.vstack(contours1))
        hull2 = cv2.convexHull(np.vstack(contours2))
        hull_score = compare_contours_hu(hull1, hull2)

    # SSIM 分数
    ssim_score = 0.0
    try:
        g1 = cv2.cvtColor(crop1, cv2.COLOR_BGR2GRAY) if len(crop1.shape) == 3 else crop1
        g2 = cv2.cvtColor(aligned_crop2, cv2.COLOR_BGR2GRAY) if len(aligned_crop2.shape) == 3 else aligned_crop2
        _, b1 = cv2.threshold(g1, 127, 255, cv2.THRESH_BINARY)
        _, b2 = cv2.threshold(g2, 127, 255, cv2.THRESH_BINARY)
        ssim_score, _ = ssim(b1, b2, full=True)
    except Exception:
        pass

    shape_score = hu_avg * 0.5 + float(hull_score) * 0.3 + float(ssim_score) * 0.2
    return {"shape_score": float(shape_score)}


def compare_region_pair(
    img1: np.ndarray, 
    img2: np.ndarray, 
    box1: List[float], 
    box2: List[float],
    output_dir: str = None,
    pair_idx: int = 0,
    use_vlm: bool = False,
    vlm_model: str = None
) -> Dict:
    """
    对比两个区域的形状
    
    使用多种方法进行形状对比，容忍比例和角度差异：
    1. SIFT 特征点对齐
    2. 轮廓提取 + Hu 矩对比
    3. SSIM 作为辅助参考
    
    或使用 VLM (视觉大模型) 进行语义级对比。
    
    Args:
        img1, img2: 原始图像
        box1, box2: 区域坐标 [x1, y1, x2, y2]
        output_dir: 输出目录（用于保存对比可视化）
        pair_idx: 区域对索引
        use_vlm: 是否使用 VLM 进行对比
        vlm_model: VLM 模型名称 (默认 qwen3-vl:8b)
        
    Returns:
        对比结果字典
    """
    cv2 = _require_cv2()
    ssim = _get_ssim()
    # ===== VLM 对比模式 =====
    if use_vlm:
        vlm_result = None
        try:
            from label_detection.services.vlm_service import get_vlm_comparator
            comparator = get_vlm_comparator(model_name=vlm_model)
            vlm_result = comparator.compare_region_pair(
                img1, img2, box1, box2,
                output_dir=output_dir,
                pair_idx=pair_idx
            )
        except ImportError as e:
            print(f"[警告] 无法加载 VLM 模块: {e}")
            print("[警告] 回退到传统对比方法")
        except Exception as e:
            print(f"[警告] VLM 对比失败: {e}")
            print("[警告] 回退到传统对比方法")

        if vlm_result is not None:
            # 计算传统分数用于融合判定
            trad_result = _compute_traditional_score(img1, img2, box1, box2)
            shape_score = trad_result.get("shape_score", 0.5)
            vlm_decision = vlm_result.get("decision", "unknown")

            # 融合判定规则
            if shape_score >= 0.8:
                # 传统高置信度 match → 直接判 match
                vlm_result["decision"] = "match"
                vlm_result["is_match"] = True
                vlm_result["needs_review"] = False
                vlm_result["judgment_source"] = "traditional_override"
            elif shape_score <= 0.2 and vlm_decision != "match":
                # 传统高置信度 mismatch 且 VLM 未判 match → 判 mismatch
                vlm_result["decision"] = "mismatch"
                vlm_result["is_match"] = False
                vlm_result["needs_review"] = False
                vlm_result["judgment_source"] = "traditional_override"
            elif shape_score <= 0.2 and vlm_decision == "match":
                # 传统说不匹配但 VLM 说匹配 → 冲突，标记需复核
                vlm_result["needs_review"] = True
                vlm_result["judgment_source"] = "conflict"
            elif vlm_decision == "unknown":
                # VLM 无法判定，保持 unknown
                vlm_result["judgment_source"] = "vlm_unknown"
            elif vlm_decision == "mismatch" and shape_score >= 0.6:
                # VLM 说不匹配但传统说相似 → 冲突，标记需复核
                vlm_result["decision"] = "unknown"
                vlm_result["is_match"] = False
                vlm_result["needs_review"] = True
                vlm_result["judgment_source"] = "conflict"
            else:
                vlm_result["judgment_source"] = "vlm"

            vlm_result["traditional_shape_score"] = shape_score
            return vlm_result
    # 裁剪区域
    x1, y1, x2, y2 = [int(v) for v in box1]
    crop1 = img1[y1:y2, x1:x2]
    
    x1, y1, x2, y2 = [int(v) for v in box2]
    crop2 = img2[y1:y2, x1:x2]
    
    if crop1.size == 0 or crop2.size == 0:
        return {"shape_score": 0.0, "error": "Empty crop region"}
    
    result = {
        "crop1_size": list(crop1.shape[:2]),
        "crop2_size": list(crop2.shape[:2]),
    }
    
    # ===== Step 1: 特征点对齐 =====
    aligned_crop2, align_success = align_images_sift(crop1, crop2)
    result["align_success"] = align_success
    
    # ===== Step 2: 轮廓提取 =====
    contours1 = extract_main_contours(crop1)
    contours2 = extract_main_contours(aligned_crop2)
    
    result["contour_count_1"] = len(contours1)
    result["contour_count_2"] = len(contours2)
    
    # ===== Step 3: Hu 矩形状对比 =====
    if contours1 and contours2:
        # 对比最大的几个轮廓
        n_compare = min(3, len(contours1), len(contours2))
        hu_scores = []
        
        for i in range(n_compare):
            score = compare_contours_hu(contours1[i], contours2[i])
            hu_scores.append(score)
        
        avg_hu_score = np.mean(hu_scores)
        result["hu_moment_scores"] = [float(s) for s in hu_scores]
        result["hu_moment_avg"] = float(avg_hu_score)
    else:
        result["hu_moment_avg"] = 0.0
        result["hu_moment_scores"] = []
    
    # ===== Step 4: 使用 matchShapes 对比所有轮廓组合 =====
    if contours1 and contours2:
        # 合并所有轮廓为一个大轮廓（如果有多个）
        all_points_1 = np.vstack(contours1) if len(contours1) > 0 else contours1[0]
        all_points_2 = np.vstack(contours2) if len(contours2) > 0 else contours2[0]
        
        # 创建凸包来近似整体形状
        hull1 = cv2.convexHull(all_points_1)
        hull2 = cv2.convexHull(all_points_2)
        
        hull_score = compare_contours_hu(hull1, hull2)
        result["hull_shape_score"] = float(hull_score)
    else:
        result["hull_shape_score"] = 0.0
    
    # ===== Step 5: SSIM 作为辅助参考（对齐后） =====
    gray1 = cv2.cvtColor(crop1, cv2.COLOR_BGR2GRAY) if len(crop1.shape) == 3 else crop1
    gray2 = cv2.cvtColor(aligned_crop2, cv2.COLOR_BGR2GRAY) if len(aligned_crop2.shape) == 3 else aligned_crop2
    
    # 二值化
    _, binary1 = cv2.threshold(gray1, 127, 255, cv2.THRESH_BINARY)
    _, binary2 = cv2.threshold(gray2, 127, 255, cv2.THRESH_BINARY)
    
    try:
        ssim_score, _ = ssim(binary1, binary2, full=True)
        result["ssim_score"] = float(ssim_score)
    except Exception as e:
        result["ssim_score"] = 0.0
    
    # ===== 综合形状分数 =====
    # 使用加权平均：Hu矩权重较高（更适合形状对比）
    hu_weight = 0.5
    hull_weight = 0.3
    ssim_weight = 0.2
    
    shape_score = (
        result.get("hu_moment_avg", 0) * hu_weight +
        result.get("hull_shape_score", 0) * hull_weight +
        result.get("ssim_score", 0) * ssim_weight
    )
    result["shape_score"] = float(shape_score)
    
    # ===== 多级回退判定 =====
    # VLM 不可用时，基于传统方法 shape_score 给出判定
    if shape_score >= 0.7:
        result["decision"] = "match"
        result["is_match"] = True
        result["confidence"] = min(shape_score, 0.95)
        result["differences"] = []
        result["summary"] = f"传统方法判定匹配 (shape_score={shape_score:.2f})"
        result["needs_review"] = False
        result["error_type"] = None
    elif shape_score <= 0.3:
        result["decision"] = "mismatch"
        result["is_match"] = False
        result["confidence"] = min(1.0 - shape_score, 0.95)
        result["differences"] = [f"形状分数过低: {shape_score:.2f}"]
        result["summary"] = f"传统方法判定不匹配 (shape_score={shape_score:.2f})"
        result["needs_review"] = False
        result["error_type"] = None
    else:
        # 中间区域无法确定，标记需人工复核
        result["decision"] = "unknown"
        result["is_match"] = False
        result["confidence"] = 0.0
        result["differences"] = [f"形状分数不确定: {shape_score:.2f}"]
        result["summary"] = f"传统方法无法确定，建议人工复核 (shape_score={shape_score:.2f})"
        result["needs_review"] = True
        result["error_type"] = None
    
    result["judgment_source"] = "traditional"
    result["fallback_method"] = "traditional"
    
    # ===== 保存对比可视化 =====
    if output_dir:
        # 绘制轮廓对比图
        vis1 = crop1.copy()
        vis2 = aligned_crop2.copy()
        
        for contour in contours1[:3]:
            cv2.drawContours(vis1, [contour], -1, (0, 255, 0), 2)
        for contour in contours2[:3]:
            cv2.drawContours(vis2, [contour], -1, (0, 255, 0), 2)
        
        cv2.imwrite(os.path.join(output_dir, f"region_{pair_idx}_template_contours.jpg"), vis1)
        cv2.imwrite(os.path.join(output_dir, f"region_{pair_idx}_target_aligned_contours.jpg"), vis2)
    
    return result



def preprocess_template_image(template_path: str, output_dir: str = "results/layout"):
    """
    预处理模板图片：检测黑框，去除黑框外白边，裁剪到黑框内部
    """
    cv2 = _require_cv2()
    from label_detection.preprocessing import crop_to_border, find_black_border

    os.makedirs(output_dir, exist_ok=True)

    print("\n[预处理] 处理模板图片...")
    template = cv2.imread(template_path)

    if template is None:
        return None, None, f"无法读取图片: {template_path}"

    print(f"  原始尺寸: {template.shape}")

    border = find_black_border(template, output_dir, "template")

    if border is None:
        print("  未检测到明确的黑框，使用原图")
        cropped = template
    else:
        cropped = crop_to_border(template, border, padding=3)
        print(f"  裁剪后尺寸: {cropped.shape}")

    output_path = os.path.join(output_dir, "template_preprocessed.jpg")
    cv2.imwrite(output_path, cropped)
    print(f"  已保存: {output_path}")

    return cropped, output_path, None


def draw_regions(img: np.ndarray, regions: List[Dict], color: Tuple = (0, 255, 0)) -> np.ndarray:
    """
    在图像上绘制检测到的区域
    """
    cv2 = _require_cv2()
    result = img.copy()
    for i, region in enumerate(regions):
        x1, y1, x2, y2 = [int(v) for v in region["coordinate"]]
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        label = f"{region['label']}({region['score']:.2f})"
        cv2.putText(result, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return result


def run_layout_comparison(
    pdf_path: str, 
    target_image_path: str, 
    output_dir: str = "results/layout",
    region_type: str = "image",
    detection_threshold: float = 0.3
) -> Dict:
    """
    完整的区域检测对比流程
    
    Args:
        pdf_path: 模板 PDF 文件路径
        target_image_path: 实拍图片路径
        output_dir: 输出目录
        region_type: 要对比的区域类型（默认 image）
        detection_threshold: 检测置信度阈值
        
    Returns:
        对比结果字典
    """
    cv2 = _require_cv2()
    from label_detection.extraction.pdf import extract_red_box_info
    from label_detection.preprocessing import preprocess_target

    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"区域检测对比：基于 PP-DocLayoutV3 ({region_type} 类型)")
    print("=" * 60)
    
    results = {
        "success": True,
        "region_type": region_type,
        "template_regions": [],
        "target_regions": [],
        "matched_pairs": [],
        "comparison_results": [],
    }
    
    # ========== Step 1: 预处理（与 unified_detection.py 相同）==========
    print("\n" + "=" * 40)
    print("Step 1: 预处理")
    print("=" * 40)
    
    # 1.1 从 PDF 提取模板图片
    print("\n[1.1] 从 PDF 提取模板图片...")
    template_raw_path = extract_red_box_info(
        pdf_path,
        target_dpi=300,
        output_dir=os.path.join(output_dir, "template_assets"),
    )
    if not template_raw_path:
        return {"success": False, "error": "无法从 PDF 提取模板图片"}
    
    # 1.2 预处理模板（检测黑框，去除白边）
    print("\n[1.2] 预处理模板图片（去除黑框外白边）...")
    template_cropped, template_path, err = preprocess_template_image(
        template_raw_path, output_dir
    )
    if err:
        return {"success": False, "error": err}
    
    # 1.3 预处理实拍图片
    print("\n[1.3] 预处理实拍图片...")
    target_cropped, _, err = preprocess_target(target_image_path, output_dir)
    if err:
        return {"success": False, "error": err}
    
    # 保存预处理后的实拍图
    target_preprocessed_path = os.path.join(output_dir, "target_preprocessed.jpg")
    cv2.imwrite(target_preprocessed_path, target_cropped)
    
    # ========== Step 2: 区域检测 ==========
    print("\n" + "=" * 40)
    print("Step 2: PP-DocLayoutV3 区域检测")
    print("=" * 40)
    
    # 2.1 检测模板区域
    print("\n[2.1] 检测模板图片区域...")
    template_all_regions = detect_layout_regions(template_path, detection_threshold)
    template_regions = extract_regions_by_type(template_all_regions, region_type)
    print(f"  检测到 {len(template_all_regions)} 个区域，其中 {region_type} 类型 {len(template_regions)} 个")
    
    # 2.2 检测实拍区域
    print("\n[2.2] 检测实拍图片区域...")
    target_all_regions = detect_layout_regions(target_preprocessed_path, detection_threshold)
    target_regions = extract_regions_by_type(target_all_regions, region_type)
    print(f"  检测到 {len(target_all_regions)} 个区域，其中 {region_type} 类型 {len(target_regions)} 个")
    
    results["template_regions"] = template_regions
    results["target_regions"] = target_regions
    
    # 保存检测可视化
    template_vis = draw_regions(template_cropped, template_regions, (0, 255, 0))
    target_vis = draw_regions(target_cropped, target_regions, (0, 255, 0))
    cv2.imwrite(os.path.join(output_dir, "template_regions.jpg"), template_vis)
    cv2.imwrite(os.path.join(output_dir, "target_regions.jpg"), target_vis)
    
    # ========== Step 3: 区域匹配 ==========
    print("\n" + "=" * 40)
    print("Step 3: 区域匹配")
    print("=" * 40)
    
    matched_pairs, unmatched1, unmatched2 = match_regions(
        template_regions, 
        target_regions,
        template_cropped.shape[:2],
        target_cropped.shape[:2],
    )
    
    print(f"\n  匹配结果:")
    print(f"    - 成功匹配: {len(matched_pairs)} 对")
    print(f"    - 模板未匹配: {len(unmatched1)} 个")
    print(f"    - 实拍未匹配: {len(unmatched2)} 个")
    
    results["matched_pairs"] = matched_pairs
    results["unmatched_template"] = unmatched1
    results["unmatched_target"] = unmatched2
    
    # ========== Step 4: 区域对比 ==========
    print("\n" + "=" * 40)
    print("Step 4: 区域内容对比")
    print("=" * 40)
    
    comparison_results = []
    for idx, (i, j, dist) in enumerate(matched_pairs):
        region1 = template_regions[i]
        region2 = target_regions[j]
        
        print(f"\n[对比 {idx+1}] 模板区域 {i} <-> 实拍区域 {j} (距离: {dist:.4f})")
        
        result = compare_region_pair(
            template_cropped, 
            target_cropped,
            region1["coordinate"],
            region2["coordinate"],
            output_dir=output_dir,
            pair_idx=idx
        )
        result["template_idx"] = i
        result["target_idx"] = j
        result["match_distance"] = dist
        
        print(f"  对齐成功: {result.get('align_success', False)}")
        print(f"  Hu矩形状分数: {result.get('hu_moment_avg', 0):.4f}")
        print(f"  凸包形状分数: {result.get('hull_shape_score', 0):.4f}")
        print(f"  SSIM分数: {result.get('ssim_score', 0):.4f}")
        print(f"  综合形状分数: {result.get('shape_score', 0):.4f}")
        
        comparison_results.append(result)
        
        # 保存裁剪的区域图片
        x1, y1, x2, y2 = [int(v) for v in region1["coordinate"]]
        crop1 = template_cropped[y1:y2, x1:x2]
        x1, y1, x2, y2 = [int(v) for v in region2["coordinate"]]
        crop2 = target_cropped[y1:y2, x1:x2]
        
        cv2.imwrite(os.path.join(output_dir, f"region_{idx}_template.jpg"), crop1)
        cv2.imwrite(os.path.join(output_dir, f"region_{idx}_target.jpg"), crop2)
    
    results["comparison_results"] = comparison_results
    
    # ========== 输出汇总 ==========
    print("\n" + "=" * 60)
    print("检测完成！结果汇总")
    print("=" * 60)
    
    # 计算统计（使用 shape_score 替代 ssim_score）
    total_pairs = len(matched_pairs)
    if total_pairs > 0:
        avg_shape = np.mean([r.get("shape_score", 0) for r in comparison_results])
        min_shape = min([r.get("shape_score", 0) for r in comparison_results])
        avg_hu = np.mean([r.get("hu_moment_avg", 0) for r in comparison_results])
    else:
        avg_shape = 0.0
        min_shape = 0.0
        avg_hu = 0.0
    
    print(f"\n🖼️ 区域检测:")
    print(f"   - 模板 {region_type} 区域: {len(template_regions)} 个")
    print(f"   - 实拍 {region_type} 区域: {len(target_regions)} 个")
    
    print(f"\n🔗 区域匹配:")
    print(f"   - 成功匹配: {total_pairs} 对")
    print(f"   - 模板未匹配: {len(unmatched1)} 个")
    print(f"   - 实拍未匹配: {len(unmatched2)} 个")
    
    print(f"\n📊 形状相似度统计:")
    print(f"   - 平均综合形状分数: {avg_shape:.4f}")
    print(f"   - 最低综合形状分数: {min_shape:.4f}")
    print(f"   - 平均Hu矩形状分数: {avg_hu:.4f}")
    
    # 综合判定（使用 shape_score）
    region_count_match = len(template_regions) == len(target_regions)
    all_matched = len(unmatched1) == 0 and len(unmatched2) == 0
    high_similarity = avg_shape > 0.7 and min_shape > 0.5
    
    if region_count_match and all_matched and high_similarity:
        verdict = "✅ 图形区域形状一致"
    elif all_matched and avg_shape > 0.5:
        verdict = f"⚠️ 图形区域形状存在差异（平均形状分数 {avg_shape:.2f}）"
    elif all_matched:
        verdict = f"⚠️ 图形区域形状差异较大（平均形状分数 {avg_shape:.2f}，最低 {min_shape:.2f}）"
    else:
        verdict = f"❌ 图形区域不匹配（{len(unmatched1) + len(unmatched2)} 个未匹配）"
    
    print(f"\n🏷️ 综合判定: {verdict}")
    
    results["verdict"] = verdict
    results["avg_shape_score"] = avg_shape
    results["min_shape_score"] = min_shape
    results["avg_hu_score"] = avg_hu
    results["output_dir"] = output_dir
    
    # 保存结果 JSON（使用自定义 encoder 处理 numpy 类型）
    result_json_path = os.path.join(output_dir, "comparison_result.json")
    
    class NumpyEncoder(json.JSONEncoder):
        """处理 numpy 类型的 JSON encoder"""
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    print(f"\n结果已保存: {result_json_path}")
    
    return results
