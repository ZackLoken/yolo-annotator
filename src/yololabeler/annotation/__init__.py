"""The annotation domain package: the per-image document and the engine that
edits it.
"""

from .engine import AnnotationEngine
from .tab import AnnotateTab

__all__ = ["AnnotateTab", "AnnotationEngine"]
