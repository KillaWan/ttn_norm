"""san_route_paths — explicit registry for route path implementations."""
from __future__ import annotations

from .affine_residual import AffineResidualPath

_REGISTRY = {
    "affine_residual": AffineResidualPath,
}


def build_route_path(route_path: str, **kwargs):
    """Return a constructed route path instance.

    Args:
        route_path: One of the registered path names.
        **kwargs:   Passed to the path constructor.
    """
    if route_path not in _REGISTRY:
        raise ValueError(
            f"Unknown route_path '{route_path}'. Registered: {list(_REGISTRY)}"
        )
    return _REGISTRY[route_path](**kwargs)


__all__ = ["AffineResidualPath", "build_route_path"]
