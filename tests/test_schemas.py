"""
schemas 模块单元测试

测试 AirConditionerLabel 的创建、序列化、字段默认值
"""

import pytest
import sys
import os

# 将项目根目录添加到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from label_detection.models.schema import AirConditionerLabel


class TestAirConditionerLabel:
    """AirConditionerLabel 数据模型测试"""

    def test_create_empty(self):
        """测试空模型创建，所有字段应为 None"""
        label = AirConditionerLabel()
        for field_name in AirConditionerLabel.model_fields:
            assert getattr(label, field_name) is None

    def test_create_with_data(self):
        """测试带数据创建"""
        label = AirConditionerLabel(
            brand="GREE",
            model_number="GWH12AAD-K6DNA2E/I",
            voltage="220-240V~",
        )
        assert label.brand == "GREE"
        assert label.model_number == "GWH12AAD-K6DNA2E/I"
        assert label.voltage == "220-240V~"
        assert label.frequency is None  # 未设置的字段

    def test_create_from_dict(self):
        """测试从字典创建（模拟 LLM 解析结果）"""
        data = {
            "brand": "GREE",
            "product_type": "SPLIT AIR CONDITIONER",
            "frequency": "50Hz",
        }
        label = AirConditionerLabel(**data)
        assert label.brand == "GREE"
        assert label.product_type == "SPLIT AIR CONDITIONER"
        assert label.frequency == "50Hz"

    def test_model_dump(self):
        """测试序列化为字典"""
        label = AirConditionerLabel(brand="GREE", voltage="220V")
        dumped = label.model_dump()
        assert isinstance(dumped, dict)
        assert dumped["brand"] == "GREE"
        assert dumped["voltage"] == "220V"
        assert dumped["frequency"] is None

    def test_model_fields_count(self):
        """测试字段数量（确保与 PROJECT_MODIFICATION_PLAN 一致）"""
        assert len(AirConditionerLabel.model_fields) == 14

    def test_ignore_extra_fields(self):
        """测试忽略未知字段（LLM 可能返回额外字段）"""
        data = {
            "brand": "GREE",
            "unknown_field": "should be ignored",
        }
        # Pydantic v2 默认忽略额外字段
        label = AirConditionerLabel(**data)
        assert label.brand == "GREE"
        assert not hasattr(label, "unknown_field")

    def test_equality_comparison(self):
        """测试两个模型对象的比较（文字对比核心逻辑）"""
        label1 = AirConditionerLabel(brand="GREE", voltage="220V")
        label2 = AirConditionerLabel(brand="GREE", voltage="220V")
        label3 = AirConditionerLabel(brand="GREE", voltage="240V")

        # 逐字段比较
        for field in AirConditionerLabel.model_fields:
            assert getattr(label1, field) == getattr(label2, field)

        assert label1.voltage != label3.voltage
