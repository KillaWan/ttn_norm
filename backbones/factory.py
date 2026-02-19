import os
import sys
from typing import Any


def _ensure_fan_on_path() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    fan_root = os.path.join(root, "FAN")
    if fan_root not in sys.path:
        sys.path.append(fan_root)


def build_backbone(name: str, **kwargs: Any):
    _ensure_fan_on_path()
    from torch_timeseries import models as fan_models

    if not hasattr(fan_models, name):
        raise ValueError(f"Unknown FAN backbone: {name}")
    return getattr(fan_models, name)(**kwargs)
