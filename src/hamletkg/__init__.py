"""HAMLET-KG research prototype."""

from .models import (
    AtomicEvent,
    CaseGraph,
    Entity,
    GraphRelation,
    RiskEvent,
    RiskPattern,
)
from .materializer import IndexedWindowPatternMatcher, PatternMatcher
from .similarity import FusedGraphCaseMatcher, HierarchicalCaseMatcher, RiskCaseIndex

__all__ = [
    "AtomicEvent",
    "CaseGraph",
    "Entity",
    "GraphRelation",
    "RiskEvent",
    "RiskPattern",
    "PatternMatcher",
    "IndexedWindowPatternMatcher",
    "HierarchicalCaseMatcher",
    "FusedGraphCaseMatcher",
    "RiskCaseIndex",
]

__version__ = "0.2.0"
