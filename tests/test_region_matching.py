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
    denormalize_coordinates,
    calculate_iou,
    match_regions,
    _compute_match_cost,
    split_barcode_regions,
    split_composite_image_regions,
    merge_fragmented_split_regions,
    filter_split_image_regions,
    infer_corresponding_region,
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


class TestCoordinateProjection:
    def test_denormalize_coordinates(self):
        box = denormalize_coordinates([0.1, 0.2, 0.4, 0.5], (200, 100))
        assert box == pytest.approx([10.0, 40.0, 40.0, 100.0])

    def test_infer_corresponding_region_with_matched_offset(self):
        template_regions = [
            {"coordinate": [20, 20, 60, 60], "label": "image", "score": 0.9},
        ]
        target_regions = [
            {"coordinate": [30, 30, 70, 70], "label": "image", "score": 0.9},
            {"coordinate": [110, 40, 150, 80], "label": "image", "score": 0.9},
        ]
        matched_pairs = [(0, 0, 0.05)]

        inferred = infer_corresponding_region(
            target_regions[1],
            (200, 200),
            (200, 200),
            matched_pairs,
            template_regions,
            target_regions,
            source_side="target",
        )

        assert inferred is not None
        assert inferred["inferred"] is True
        assert inferred["inferred_from"] == "target"
        assert inferred["coordinate"] == [100, 30, 140, 70]


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


class TestBarcodeRegionFilter:
    def _make_region(self, x1, y1, x2, y2):
        return {"coordinate": [x1, y1, x2, y2], "label": "image", "score": 0.9}

    def test_barcode_region_is_skipped(self):
        img = np.full((220, 320, 3), 255, dtype=np.uint8)
        for x in range(165, 285, 6):
            img[95:160, x : x + 3] = 0

        region = self._make_region(150, 80, 300, 190)
        ocr_boxes = [
            (
                np.array([[165, 160], [285, 160], [285, 180], [165, 180]], dtype=np.float32),
                "600001076226",
                0.99,
            )
        ]

        comparable, skipped = split_barcode_regions([region], img, ocr_boxes)
        assert comparable == []
        assert len(skipped) == 1
        assert skipped[0]["skip_reason"] == "barcode"
        assert skipped[0]["barcode_hint"]["barcode_digits"] == "600001076226"

    def test_plain_graphic_region_is_kept(self):
        img = np.full((220, 320, 3), 255, dtype=np.uint8)
        img[70:150, 40:250] = 0

        region = self._make_region(30, 60, 260, 160)
        ocr_boxes = [
            (
                np.array([[45, 90], [240, 90], [240, 120], [45, 120]], dtype=np.float32),
                "GWH24AGD-K6DNA1C/I(WIFI)",
                0.97,
            )
        ]

        comparable, skipped = split_barcode_regions([region], img, ocr_boxes)
        assert len(comparable) == 1
        assert skipped == []

    def test_texture_only_barcode_region_is_skipped_without_digits(self):
        img = np.full((240, 360, 3), 255, dtype=np.uint8)
        for x in range(150, 300, 8):
            img[90:180, x : x + 4] = 0

        region = self._make_region(140, 80, 310, 185)
        comparable, skipped = split_barcode_regions([region], img, None)

        assert comparable == []
        assert len(skipped) == 1
        assert skipped[0]["skip_reason"] == "barcode"


class TestCompositeImageRegionSplit:
    def _make_region(self, x1, y1, x2, y2):
        return {"coordinate": [x1, y1, x2, y2], "label": "image", "score": 0.9}

    def test_splits_wide_region_into_multiple_icon_groups(self):
        img = np.full((220, 360, 3), 255, dtype=np.uint8)

        # 左侧一个图标组：两个相邻块应被合并成一个子区域。
        img[60:170, 30:70] = 0
        img[60:170, 76:116] = 0

        # 右侧另一个图标组：与左组距离明显更大，应拆成第二个子区域。
        img[55:175, 190:255] = 0

        region = self._make_region(20, 40, 270, 190)
        refined, split_parents = split_composite_image_regions([region], img)

        assert len(split_parents) == 1
        assert len(refined) == 2
        assert all(item["split_child_count"] == 2 for item in refined)
        assert refined[0]["coordinate"][2] < refined[1]["coordinate"][0]
        assert refined[0]["split_parent_coordinate"] == region["coordinate"]
        assert refined[1]["split_parent_coordinate"] == region["coordinate"]

    def test_merges_stacked_parts_of_same_icon(self):
        img = np.full((240, 420, 3), 255, dtype=np.uint8)

        # 左侧图标由上下两段组成，应该被合并成一个框。
        img[60:140, 40:110] = 0
        img[150:162, 48:102] = 0

        # 右侧第二个图标。
        img[55:175, 250:320] = 0

        region = self._make_region(20, 40, 340, 190)
        refined, split_parents = split_composite_image_regions([region], img)

        assert len(split_parents) == 1
        assert len(refined) == 2

        refined = sorted(refined, key=lambda item: item["coordinate"][0])
        left_box = refined[0]["coordinate"]
        assert left_box[1] <= 60
        assert left_box[3] >= 162

    def test_keeps_single_graphic_region_when_no_clear_split_exists(self):
        img = np.full((220, 320, 3), 255, dtype=np.uint8)
        img[70:160, 40:250] = 0

        region = self._make_region(30, 60, 260, 170)
        refined, split_parents = split_composite_image_regions([region], img)

        assert len(refined) == 1
        assert refined[0]["coordinate"] == region["coordinate"]
        assert split_parents == []


class TestSplitRegionPostFilter:
    def _make_region(self, x1, y1, x2, y2, parent_box=None, child_idx=None):
        region = {"coordinate": [x1, y1, x2, y2], "label": "image", "score": 0.9}
        if parent_box is not None:
            region["split_parent_coordinate"] = list(parent_box)
        if child_idx is not None:
            region["split_child_idx"] = child_idx
        return region

    def test_merge_fragmented_split_regions_merges_vertical_siblings(self):
        parent = [20, 40, 340, 190]
        regions = [
            self._make_region(40, 60, 110, 140, parent_box=parent, child_idx=0),
            self._make_region(48, 150, 102, 162, parent_box=parent, child_idx=1),
            self._make_region(250, 55, 320, 175, parent_box=parent, child_idx=2),
        ]

        merged = merge_fragmented_split_regions(regions)

        assert len(merged) == 2
        merged = sorted(merged, key=lambda item: item["coordinate"][0])
        assert merged[0]["coordinate"] == [40, 60, 110, 162]
        assert merged[1]["coordinate"] == [250, 55, 320, 175]

    def test_merge_fragmented_split_regions_merges_stacked_symbol_fragments(self):
        parent = [38, 527, 236, 619]
        regions = [
            self._make_region(42, 528, 94, 624, parent_box=parent, child_idx=0),
            self._make_region(113, 528, 165, 624, parent_box=parent, child_idx=1),
            self._make_region(185, 535, 227, 556, parent_box=parent, child_idx=2),
            self._make_region(175, 604, 233, 621, parent_box=parent, child_idx=3),
        ]

        merged = merge_fragmented_split_regions(regions)

        assert len(merged) == 3
        merged = sorted(merged, key=lambda item: item["coordinate"][0])
        assert merged[0]["coordinate"] == [42, 528, 94, 624]
        assert merged[1]["coordinate"] == [113, 528, 165, 624]
        assert merged[2]["coordinate"] == [175, 535, 233, 621]
        assert merged[2]["merged_child_indices"] == [2, 3]

    def test_filter_split_image_regions_skips_barcode_clusters(self):
        img = np.full((240, 360, 3), 255, dtype=np.uint8)
        for x in range(140, 310, 8):
            img[92:138, x : x + 4] = 0

        parent = [130, 88, 320, 142]
        regions = [
            self._make_region(142, 92, 154, 138, parent_box=parent, child_idx=0),
            self._make_region(178, 92, 190, 138, parent_box=parent, child_idx=1),
            self._make_region(214, 92, 226, 138, parent_box=parent, child_idx=2),
            self._make_region(250, 92, 262, 138, parent_box=parent, child_idx=3),
        ]

        kept, skipped = filter_split_image_regions(regions, img, None)

        assert kept == []
        assert len(skipped) == 4
        assert all(item["skip_reason"] == "barcode_cluster" for item in skipped)

    def test_filter_split_image_regions_skips_compact_right_side_barcode_cluster(self):
        img = np.full((260, 420, 3), 255, dtype=np.uint8)
        for x in range(280, 388, 7):
            img[110:150, x : x + 3] = 0

        parent = [250, 104, 395, 152]
        regions = [
            self._make_region(262, 108, 320, 150, parent_box=parent, child_idx=0),
            self._make_region(324, 110, 356, 150, parent_box=parent, child_idx=1),
            self._make_region(360, 112, 392, 148, parent_box=parent, child_idx=2),
        ]

        kept, skipped = filter_split_image_regions(regions, img, None)

        assert kept == []
        assert len(skipped) == 3
        assert all(item["skip_reason"] == "barcode_cluster" for item in skipped)

    def test_filter_split_image_regions_keeps_graphic_next_to_barcode_cluster(self):
        img = np.full((240, 500, 3), 255, dtype=np.uint8)
        img[104:146, 274:326] = 0
        img[116:134, 260:340] = 0
        for x in range(350, 486, 7):
            img[96:152, x : x + 3] = 0

        parent = [250, 88, 492, 160]
        regions = [
            self._make_region(260, 96, 340, 152, parent_box=parent, child_idx=0),
            self._make_region(350, 96, 430, 152, parent_box=parent, child_idx=1),
            self._make_region(440, 96, 488, 152, parent_box=parent, child_idx=2),
        ]

        kept, skipped = filter_split_image_regions(regions, img, None)

        assert len(kept) == 1
        assert kept[0]["coordinate"] == [260, 96, 340, 152]
        assert len(skipped) == 2
        assert all(item["skip_reason"] == "barcode_cluster" for item in skipped)

    def test_filter_split_image_regions_skips_thin_sliver_child(self):
        img = np.full((220, 360, 3), 255, dtype=np.uint8)
        parent = [20, 40, 270, 190]
        regions = [
            self._make_region(40, 60, 110, 150, parent_box=parent, child_idx=0),
            self._make_region(130, 158, 240, 178, parent_box=parent, child_idx=1),
        ]

        kept, skipped = filter_split_image_regions(regions, img, None)

        assert len(kept) == 1
        assert kept[0]["coordinate"] == [40, 60, 110, 150]
        assert len(skipped) == 1
        assert skipped[0]["skip_reason"] == "thin_sliver"
