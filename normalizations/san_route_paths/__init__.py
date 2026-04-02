"""san_route_paths — explicit registry for route path implementations.

Canonical paths:
    none               — no routing (handled in SANRouteNorm, not here)
    local_transport    — monotonic PL spline (LocalTransportPath)
    residual_content   — dilated-TCN additive delta (ResidualContentPath)
    alignment          — monotonic temporal warping (AlignmentPath)
    gating             — 4-expert soft-gating bank (GatingPath)

Aliases (for backward compatibility):
    local_value_parameter → local_transport
"""
from __future__ import annotations

from .alignment import AlignmentPath
from .gating import GatingPath
from .local_transport import LocalTransportPath
from .residual_content import ResidualContentPath

_REGISTRY: dict[str, type] = {
    "local_transport":  LocalTransportPath,
    "residual_content": ResidualContentPath,
    "alignment":        AlignmentPath,
    "gating":           GatingPath,
}

_ALIASES: dict[str, str] = {
    "local_value_parameter": "local_transport",
}


def build_route_path(route_path: str, **kwargs):
    """Return a constructed route path instance.

    Resolves aliases before lookup.  Passes all kwargs to the path constructor;
    each path accepts (**kwargs) to silently absorb unneeded parameters.

    Args:
        route_path: Canonical name or alias.
        **kwargs:   Forwarded to the path constructor (pred_len, enc_in, …).
    """
    canonical = _ALIASES.get(route_path, route_path)
    if canonical not in _REGISTRY:
        raise ValueError(
            f"Unknown route_path '{route_path}'. "
            f"Registered: {list(_REGISTRY)}  Aliases: {list(_ALIASES)}"
        )
    return _REGISTRY[canonical](**kwargs)


__all__ = [
    "LocalTransportPath",
    "ResidualContentPath",
    "AlignmentPath",
    "GatingPath",
    "build_route_path",
]
