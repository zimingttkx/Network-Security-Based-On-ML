"""
Network Intrusion Prevention Engine.

Components:
- RuleEngine:       IP/protocol/rate-limit fast filtering (microseconds)
- KitsuneDetector:  AfterImage 115-dim feature extraction + KitNET anomaly detection (NDSS'18)
- LucidDetector:    1D CNN DDoS flow detection (IEEE TNSM 2020)
- DetectionPipeline: Multi-stage chain with short-circuit on BLOCK verdict
"""

from networksecurity.engine.detector import BaseDetector, PacketInfo
from networksecurity.engine.pipeline import DetectionPipeline
from networksecurity.engine.rule_engine import RuleEngine
from networksecurity.engine.verdict import Action, ThreatLevel, Verdict

__all__ = [
    "Action",
    "BaseDetector",
    "DetectionPipeline",
    "PacketInfo",
    "RuleEngine",
    "ThreatLevel",
    "Verdict",
]
