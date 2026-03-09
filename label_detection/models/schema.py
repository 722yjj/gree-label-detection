"""Compatibility wrapper for the shared schema module."""

from label_detection.schema import (
    AirConditionerLabel,
    CompactSpecLabel,
    LABEL_KIND_COMPACT,
    LABEL_KIND_STANDARD,
    get_label_model,
    infer_label_kind,
)

__all__ = [
    "AirConditionerLabel",
    "CompactSpecLabel",
    "LABEL_KIND_STANDARD",
    "LABEL_KIND_COMPACT",
    "infer_label_kind",
    "get_label_model",
]
