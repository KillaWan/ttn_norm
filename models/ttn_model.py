from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ttn_norm.backbones import build_backbone
from ttn_norm.models.local_tf_norm.local_norm import LocalTFNorm, LocalTFNormState
from ttn_norm.normalizations import (
    DishTS,
    FAN,
    No,
    RevIN,
    SAN,
    SANRouteNorm,
    SANPhasePatchFilterNorm,
    SANPhaseResidualGainNorm,
    PointMeanPhaseNorm,
)


class TTNModel(nn.Module):
    def __init__(
        self,
        backbone_type: str,
        backbone_kwargs: dict,
        norm_model: Optional[nn.Module] = None,
        is_former: Optional[bool] = None,
        effective_label_len: Optional[int] = None,
    ):
        super().__init__()
        self.backbone_type = backbone_type
        self.fm = build_backbone(backbone_type, **backbone_kwargs)
        self.nm = norm_model if norm_model is not None else nn.Identity()
        if is_former is None:
            former_names = {"timesnet", "informer", "autoformer", "fedformer", "koopa"}
            name = backbone_type.lower()
            self.is_former = "former" in name or name in former_names
        else:
            self.is_former = is_former
        if effective_label_len is not None:
            self.effective_label_len = effective_label_len
        else:
            self.effective_label_len = min(
                backbone_kwargs.get("label_len", 0),
                backbone_kwargs.get("seq_len", backbone_kwargs.get("label_len", 0)),
            )
        self._last_state: Optional[LocalTFNormState] = None

    def normalize(
        self,
        batch_x: torch.Tensor,
        dec_inp: Optional[torch.Tensor] = None,
    ):
        nm = self.nm
        if isinstance(nm, LocalTFNorm):
            # Returns (residual, state); dec_inp rescaled by same instance stats.
            batch_x, state = nm.normalize(batch_x, return_state=True)
            self._last_state = state
            if dec_inp is not None and state.mean is not None and state.std is not None:
                dec_inp = (dec_inp - state.mean) / state.std

        elif isinstance(nm, (RevIN, DishTS)):
            # normalize() returns (normalized_x, normalized_dec_inp)
            batch_x, dec_inp = nm.normalize(batch_x, dec_inp)

        elif isinstance(nm, (SAN, SANRouteNorm, SANPhasePatchFilterNorm, SANPhaseResidualGainNorm, PointMeanPhaseNorm)):
            # normalize() stores pred stats internally, returns normalized tensor.
            batch_x = nm.normalize(batch_x)

        elif isinstance(nm, FAN):
            # FAN: normalize() returns a single tensor.
            batch_x = nm.normalize(batch_x)

        elif isinstance(nm, No):
            # Pass-through — no transformation.
            pass

        else:
            # Generic fallback: assume nm(batch_x) → tensor (e.g. nn.Identity).
            batch_x = nm(batch_x)

        return batch_x, dec_inp

    def denormalize(self, pred: torch.Tensor) -> torch.Tensor:
        nm = self.nm
        if isinstance(nm, LocalTFNorm):
            return nm.denormalize(pred, state=self._last_state)

        elif isinstance(nm, (RevIN, DishTS, FAN)):
            return nm.denormalize(pred)

        elif isinstance(nm, (SAN, SANRouteNorm, SANPhasePatchFilterNorm, SANPhaseResidualGainNorm, PointMeanPhaseNorm)):
            # Uses internally stored pred stats from last normalize() call.
            return nm.denormalize(pred)

        elif isinstance(nm, No):
            return pred

        elif hasattr(nm, "denormalize"):
            return nm.denormalize(pred)

        return pred

    def forward(
        self,
        batch_x: torch.Tensor,
        batch_x_enc: Optional[torch.Tensor] = None,
        dec_inp: Optional[torch.Tensor] = None,
        dec_inp_enc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_norm, _ = self.normalize(batch_x, dec_inp=None)
        if self.is_former:
            label_len = self.effective_label_len
            pred_len = dec_inp_enc.shape[1] - label_len if dec_inp_enc is not None else getattr(self.fm, "pred_len", batch_x.shape[1])
            dec_label = x_norm[:, -label_len:, :]
            dec_future = torch.zeros(
                x_norm.size(0), pred_len, x_norm.size(-1),
                device=x_norm.device, dtype=x_norm.dtype,
            )
            dec_inp_norm = torch.cat([dec_label, dec_future], dim=1)
            pred = self.fm(x_norm, batch_x_enc, dec_inp_norm, dec_inp_enc)
        else:
            pred = self.fm(x_norm)
        pred = self.denormalize(pred)
        return pred
