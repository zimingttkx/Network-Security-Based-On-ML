"""Detection verdict data types."""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


class Action(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    LOG = "log"
    CHALLENGE = "challenge"


class ThreatLevel(str, Enum):
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Verdict:
    """Single detection verdict from one detector."""

    action: Action
    confidence: float  # 0.0 - 1.0
    threat_level: ThreatLevel = ThreatLevel.SAFE
    reason: str = ""
    detector: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "action": self.action.value,
            "confidence": self.confidence,
            "threat_level": self.threat_level.value,
            "reason": self.reason,
            "detector": self.detector,
            "metadata": self.metadata,
        }
