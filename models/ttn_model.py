from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ttn_norm.backbones import build_backbone
from ttn_norm.models.local_tf_norm.local_norm import LocalTFNorm, LocalTFNormState


class TTNModel(nn.Module):
    def __init__(
        self,
        backbone_type: str,
        backbone_kwargs: dict,
        norm_model: Optional[nn.Module] = None,
        is_former: Optional[bool] = None,
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
        self._last_state: Optional[LocalTFNormState] = None

    def normalize(
        self,
        batch_x: torch.Tensor,
        dec_inp: Optional[torch.Tensor] = None,
    ):
        if isinstance(self.nm, LocalTFNorm):
            batch_x, state = self.nm.normalize(batch_x, return_state=True)
            self._last_state = state
        else:
            batch_x = self.nm(batch_x)
        return batch_x, dec_inp

    def denormalize(self, pred: torch.Tensor) -> torch.Tensor:
        if isinstance(self.nm, LocalTFNorm):
            return self.nm.denormalize(pred, state=self._last_state)
        if hasattr(self.nm, "denormalize"):
            return self.nm.denormalize(pred)
        return pred

    def forward(
        self,
        batch_x: torch.Tensor,
        batch_x_enc: Optional[torch.Tensor] = None,
        dec_inp: Optional[torch.Tensor] = None,
        dec_inp_enc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_x, dec_inp = self.normalize(batch_x, dec_inp=dec_inp)
        if self.is_former:
            pred = self.fm(batch_x, batch_x_enc, dec_inp, dec_inp_enc)
        else:
            pred = self.fm(batch_x)
        pred = self.denormalize(pred)
        return pred
