"""A task-source layer: self-generated, retrieved and teacher-distilled behind one interface."""
from .base import Provenance, SourceResult, TaskRecord, TaskSource
from .registry import ContaminationFilter, SharedNoveltyBuffer
from .pipeline import SourceStats, TaskPipeline
from .similarity import similarity, token_jaccard, char_cosine

__all__ = ["Provenance", "SourceResult", "TaskRecord", "TaskSource", "ContaminationFilter",
           "SharedNoveltyBuffer", "SourceStats", "TaskPipeline", "similarity",
           "token_jaccard", "char_cosine"]
