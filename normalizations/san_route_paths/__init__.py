"""san_route_paths — explicit registry for route path implementations.

Canonical paths:
    none                    — no routing (handled in SANRouteNorm, not here)
    local_transport         — monotonic PL spline (LocalTransportPath)
    residual_content        — dilated-TCN additive delta (ResidualContentPath)
    alignment               — monotonic temporal warping (AlignmentPath)
    gating                  — state-conditioned 2-branch gate over level/residual (GatingPath)
    lp_state_correction     — Base-family denorm-state correction (LPStateCorrection)
                              must be paired with route_state="lp_state"
    slope_state_correction  — patch-affine denorm-state correction (SlopeStateCorrection)
                              must be paired with route_state="slope_state"
    timeapn_correction      — TimeAPN phase+mean correction (TimeAPNCorrection)
                              must be paired with route_state="timeapn_state"
                              actual logic lives in SANRouteNorm; this entry is a stub

Aliases (for backward compatibility):
    local_value_parameter → local_transport
"""
from __future__ import annotations

from .alignment import AlignmentPath
from .gating import GatingPath
from .local_transport import LocalTransportPath
from .lp_state_correction import LPStateCorrection
from .residual_content import ResidualContentPath
from .slope_state_correction import SlopeStateCorrection
from .timeapn_correction import TimeAPNCorrection

_REGISTRY: dict[str, type] = {
    "local_transport":      LocalTransportPath,
    "residual_content":     ResidualContentPath,
    "alignment":            AlignmentPath,
    "gating":               GatingPath,
    "lp_state_correction":  LPStateCorrection,
    "slope_state_correction": SlopeStateCorrection,
    "timeapn_correction":   TimeAPNCorrection,
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
    "LPStateCorrection",
    "SlopeStateCorrection",
    "TimeAPNCorrection",
    "build_route_path",
]
