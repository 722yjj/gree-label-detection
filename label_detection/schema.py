"""Shared data models."""

import re
from typing import Optional, Type

from pydantic import BaseModel, Field


LABEL_KIND_STANDARD = "air_conditioner_standard"
LABEL_KIND_COMPACT = "compact_spec"


class AirConditionerLabel(BaseModel):
    """空调标签结构化数据模型"""

    brand: Optional[str] = Field(default=None, description="品牌，例如 GREE")
    product_type: Optional[str] = Field(
        default=None,
        description="产品类型，例如 Split Air Conditioner",
    )
    model_number: Optional[str] = Field(
        default=None,
        description="Model/机型型号，例如 GWH12...",
    )
    voltage: Optional[str] = Field(default=None, description="额定电压，例如 220-240V~")
    frequency: Optional[str] = Field(default=None, description="额定频率，例如 50Hz")
    heating_capacity: Optional[str] = Field(
        default=None,
        description="制热量/Heating Capacity",
    )
    cooling_capacity: Optional[str] = Field(
        default=None,
        description="制冷量/Cooling Capacity",
    )
    air_volume: Optional[str] = Field(default=None, description="风量/Air Flow Volume")
    weight: Optional[str] = Field(default=None, description="重量/Weight")
    noise: Optional[str] = Field(default=None, description="噪声水平/Noise Level")
    mfg_date: Optional[str] = Field(default=None, description="制造日期/Date")
    manufacturer: Optional[str] = Field(default=None, description="制造商名称")
    address: Optional[str] = Field(default=None, description="制造商地址")
    barcode: Optional[str] = Field(
        default=None,
        description="条形码下方的数字/Serial Number",
    )


class CompactSpecLabel(BaseModel):
    """紧凑型标签结构化数据模型（型号/净重/毛重/颜色/管径/冷媒/条码）"""

    model_number: Optional[str] = Field(
        default=None,
        description="Model/机型型号，例如 GWH24AGD-K6DNA1C/I(WIFI)",
    )
    net_weight: Optional[str] = Field(default=None, description="净重/N.W.")
    gross_weight: Optional[str] = Field(default=None, description="毛重/G.W.")
    color: Optional[str] = Field(default=None, description="颜色/Color")
    connection_pipes: Optional[str] = Field(
        default=None,
        description='管径/Connection Pipes，例如 1/4"/1/2"',
    )
    refrigerant: Optional[str] = Field(default=None, description="冷媒/Refrigerant")
    barcode: Optional[str] = Field(
        default=None,
        description="条形码下方的数字/Serial Number",
    )


def infer_label_kind(ocr_text: object) -> str:
    """Infer the label schema from OCR text."""
    text = re.sub(r"\s+", "", str(ocr_text or "").upper())
    compact_hits = sum(
        keyword in text
        for keyword in (
            "CONNECTIONPIPES",
            "REFRIGERANT",
            "N.W",
            "G.W",
            "NW",
            "GW",
            "COLOR",
        )
    )
    return LABEL_KIND_COMPACT if compact_hits >= 2 else LABEL_KIND_STANDARD


def get_label_model(label_kind: str) -> Type[BaseModel]:
    if label_kind == LABEL_KIND_COMPACT:
        return CompactSpecLabel
    return AirConditionerLabel
