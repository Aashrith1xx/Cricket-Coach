"""Metric computation, split by discipline.

Importing this package does not import the discipline modules, so the worker can load only
what a given clip needs.
"""

__all__ = ["batting", "bowling"]
