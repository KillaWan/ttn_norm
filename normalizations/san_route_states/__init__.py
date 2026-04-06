"""san_route_states — explicit registry for route state implementations."""
from __future__ import annotations

from .base import RouteStateBase
from .dlogsigma import DLogSigmaState
from .lp_state import LPState
from .nu import NuState
from .omega_spec import OmegaSpecState
from .slope_state import SlopeState

_REGISTRY = {
    "nu": NuState,
    "dlogsigma": DLogSigmaState,
    "omega_spec": OmegaSpecState,
    "lp_state": LPState,
    "slope_state": SlopeState,
}


def build_route_state(route_state: str) -> RouteStateBase:
    """Return a constructed route state instance.

    Args:
        route_state: One of the registered state names.
    """
    if route_state not in _REGISTRY:
        raise ValueError(
            f"Unknown route_state '{route_state}'. Registered: {list(_REGISTRY)}"
        )
    return _REGISTRY[route_state]()


__all__ = [
    "RouteStateBase",
    "NuState",
    "DLogSigmaState",
    "OmegaSpecState",
    "LPState",
    "SlopeState",
    "build_route_state",
]
