"""
app/services/segmentation/common/base_trigger.py

Abstract base for any segmentation trigger (product, customer, etc.).
Each concrete trigger decides:
  - whether enough data exists at all (gate)
  - whether enough has changed since last run (threshold)

This keeps trigger logic centralized and consistent across feature types.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TriggerDecision:
    should_run: bool
    reason:     str   # machine-readable reason code
    detail:     str   # human-readable explanation


class SegmentationTrigger(ABC):
    """
    Concrete subclasses implement evaluate() to decide whether
    a segmentation run is warranted right now.
    """

    @abstractmethod
    def evaluate(self, business_id: str) -> TriggerDecision:
        ...