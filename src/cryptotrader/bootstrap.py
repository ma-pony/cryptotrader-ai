"""Runtime bootstrap public surface.

The executable assembly lives in :mod:`cryptotrader.runtime`; this module has
no legacy configuration or per-mode cycle builder.
"""

from cryptotrader.runtime import Runtime, build_runtime

__all__ = ["Runtime", "build_runtime"]
