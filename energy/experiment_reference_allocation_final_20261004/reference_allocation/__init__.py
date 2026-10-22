"""Shared, model-agnostic analysis for reference-allocation experiments."""

from .metrics import absolute_metrics, build_pairs, same_composition_metrics

__all__ = ["absolute_metrics", "build_pairs", "same_composition_metrics"]
