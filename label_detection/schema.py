"""Shared data models."""

from typing import Optional

from pydantic import BaseModel, Field


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
