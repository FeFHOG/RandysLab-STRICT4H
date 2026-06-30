"""Compatibility layer for the former autoquant package name.

New code should import from ``randyslab``.
"""

from randyslab import __version__

__all__ = ["__version__"]
