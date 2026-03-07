"""
区域匹配算法单元测试

测试坐标归一化、IoU 计算、多因子代价匹配（匈牙利算法）
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from label_detection.matching.layout import (
    normalize_coordinates,
    calculate_iou,
    match_regions,
    _compute_match_cost,
)


class TestNormalizeCoordinates:
    """坐标归一化测试"""

    def test_basic(self):
        box = [100, 200, 300, 400]
        img_shape = (800, 600)
        result = normalize_coordinates(box, img_shape)
        assert result == pytest.approx([100 / 600, 200 / 800, 300 / 600, 400 / 800])

    def test_full_image(self):
        box = [0, 0, 800, 600]
        img_shape = (600, 800)
        result = normalize_coordinates(box, img_shape)
        assert result == [0.0, 0.0, 1.0, 1.0]

    def test_origin(self):
        box = [0, 0, 0, 0]
        img_shape = (100, 100)
        result = normalize_coordinates(box, img_shape)
        assert result == [0.0, 0.0, 0.0, 0.0]


class TestCalculateIoU:
    """IoU 计算测试"""

    def test_identical_boxes(self):
        box = [10, 10, 50, 50]
        assert calculate_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        box1 = [0, 0, 10, 10]
        box2 = [20, 20, 30, 30]
        assert calculate_iou(box1, box2) == 0.0

    def test_partial_overlap(self):
        box1 = [0, 0, 20, 20]
        box2 = [10, 10, 30, 30]
        assert calculate_iou(box1, box2) == pytest.approx(100 / 700)

    def test_contained_box(self):
        box1 = [0, 0, 100, 100]
        box2 = [10, 10, 50, 50]
        assert calculate_iou(box1, box2) == pytest.approx(1600 / 10000)

    def test_touching_boxes(self):
        box1 = [0, 0, 10, 10]
        box2 = [10, 0, 20, 10]
        assert calculate_iou(box1, box2) == 0.0


class TestComputeMatchCost:
    """多因子匹配代价测试"""

    def test_identical_regions_zero_cost(self):
        """完全相同的区域代价应接近 0"""
        box = [10, 10, 50, 50]
        cost = _compute_match_cost(box, box, (100, 100), (100, 100))
        assert cost < 0.05

    def test_different_position_higher_cost(self):
        """位置差异较大时代价应较高"""
        box1 = [0, 0, 20, 20]
        box2 = [80, 80, 100, 100]
        cost = _compute_match_cost(box1, box2, (100, 100), (100, 100))
        assert cost > 0.3

    def test_similar_area_different_aspect_ratio(self):
        """面积相似但宽高比不同 → 代价中等"""
        box1 = [0, 0, 40, 40]  # 正方形
        box2 = [0, 0, 80, 20]  # 长条形（面积相同）
        cost = _compute_match_cost(box1, box2, (100, 100), (100, 100))
        assert cost > 0.05  # 宽高比差异贡献

    def test_different_image_sizes_same_ratio_position(self):
        """不同图像尺寸但同比例位置 → 低代价"""
        box1 = [25, 25, 75, 75]
        box2 = [50, 50, 150, 150]
        cost = _compute_match_cost(box1, box2, (100, 100), (200, 200))
        assert cost < 0.1


class TestMatchRegions:
    """区域匹配测试（匈牙利算法）"""

    def _make_region(self, x1, y1, x2, y2):
        return {"coordinate": [x1, y1, x2, y2], "label": "image", "score": 0.9}

    def test_empty_regions(self):
        matched, um1, um2 = match_regions([], [], (100, 100), (100, 100))
        assert matched == []
        assert um1 == []
        assert um2 == []

    def test_single_pair(self):
        r1 = [self._make_region(10, 10, 50, 50)]
        r2 = [self._make_region(10, 10, 50, 50)]
        matched, um1, um2 = match_regions(r1, r2, (100, 100), (100, 100))
        assert len(matched) == 1
        assert um1 == []
        assert um2 == []

    def test_unmatched_regions(self):
        """位置差异过大导致无法匹配"""
        r1 = [self._make_region(0, 0, 10, 10)]
        r2 = [self._make_region(80, 80, 100, 100)]
        matched, um1, um2 = match_regions(
            r1, r2, (100, 100), (100, 100), cost_threshold=0.3
        )
        assert len(matched) == 0
        assert um1 == [0]
        assert um2 == [0]

    def test_multiple_pairs(self):
        r1 = [
            self._make_region(10, 10, 30, 30),
            self._make_region(60, 60, 90, 90),
        ]
        r2 = [
            self._make_region(12, 12, 32, 32),
            self._make_region(58, 58, 88, 88),
        ]
        matched, um1, um2 = match_regions(r1, r2, (100, 100), (100, 100))
        assert len(matched) == 2
        assert um1 == []
        assert um2 == []

    def test_different_image_sizes(self):
        """不同尺寸图像的区域匹配（归一化坐标）"""
        r1 = [self._make_region(50, 50, 100, 100)]
        r2 = [self._make_region(100, 100, 200, 200)]
        matched, um1, um2 = match_regions(r1, r2, (200, 200), (400, 400))
        assert len(matched) == 1

    def test_one_side_empty(self):
        r1 = [self._make_region(10, 10, 50, 50)]
        matched, um1, um2 = match_regions(r1, [], (100, 100), (100, 100))
        assert matched == []
        assert um1 == [0]
        assert um2 == []

    def test_close_centers_different_areas(self):
        """两个候选区域中心接近但面积明显不同 → 匈牙利算法应正确匹配"""
        r1 = [
            self._make_region(40, 40, 60, 60),   # 小区域
            self._make_region(35, 35, 65, 65),   # 大区域（中心相同）
        ]
        r2 = [
            self._make_region(35, 35, 65, 65),   # 大区域
            self._make_region(40, 40, 60, 60),   # 小区域
        ]
        matched, um1, um2 = match_regions(r1, r2, (100, 100), (100, 100))
        assert len(matched) == 2
        # 验证面积相似的区域被正确匹配
        for i, j, cost in matched:
            r1_area = (r1[i]["coordinate"][2] - r1[i]["coordinate"][0]) * (r1[i]["coordinate"][3] - r1[i]["coordinate"][1])
            r2_area = (r2[j]["coordinate"][2] - r2[j]["coordinate"][0]) * (r2[j]["coordinate"][3] - r2[j]["coordinate"][1])
            assert r1_area == r2_area, "面积相似的区域应被匹配在一起"

    def test_target_has_extra_region(self):
        """目标图多出一个区域 → unmatched2 正确返回"""
        r1 = [self._make_region(10, 10, 30, 30)]
        r2 = [
            self._make_region(10, 10, 30, 30),
            self._make_region(60, 60, 90, 90),
        ]
        matched, um1, um2 = match_regions(r1, r2, (100, 100), (100, 100))
        assert len(matched) == 1
        assert um1 == []
        assert len(um2) == 1

    def test_cost_threshold_filters_pairs(self):
        """代价超过阈值的配对被舍弃"""
        r1 = [self._make_region(0, 0, 20, 20)]
        r2 = [self._make_region(70, 70, 90, 90)]
        # 使用很低的阈值
        matched, um1, um2 = match_regions(
            r1, r2, (100, 100), (100, 100), cost_threshold=0.1
        )
        assert len(matched) == 0
        assert um1 == [0]
        assert um2 == [0]

    def test_match_returns_cost_not_distance(self):
        """匹配结果第三个元素是代价值（非纯距离）"""
        r1 = [self._make_region(10, 10, 50, 50)]
        r2 = [self._make_region(15, 15, 55, 55)]
        matched, _, _ = match_regions(r1, r2, (100, 100), (100, 100))
        assert len(matched) == 1
        _, _, cost = matched[0]
        assert isinstance(cost, float)
        assert 0 < cost < 1  # 代价应在合理范围
