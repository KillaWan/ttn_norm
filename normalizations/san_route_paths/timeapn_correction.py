"""TimeAPNCorrection — compatibility stub for the timeapn_correction route path.

Taxonomy note
-------------
This file exists only to satisfy the route_path builder interface (build_route_path).
It has no trainable parameters and performs no computation.

Role in the timeapn_correction + timeapn_state combination
-----------------------------------------------------------
The actual TimeAPN logic is implemented entirely inside SANRouteNorm, analogous
to how lp_state_correction is handled.

SANRouteNorm.normalize():
    Computes sliding-mean normalization, FFT-based phase and amplitude extraction,
    runs timeapn_mean_predictor / timeapn_phase_predictor / timeapn_amp_predictor,
    and mixes with alpha_in gate:
        x_used = x_san_norm + alpha_in * (x_new_norm - x_san_norm)

SANRouteNorm.denormalize():
    Applies FFT-based phase compensation to the backbone output y_norm,
    adds back the predicted future mean, and gates with alpha_out:
        y_final = y_san + alpha_out * (y_new - y_san)
    When alpha_out == 0 the result is exact pure-SAN baseline.

SANRouteNorm.compute_route_state_loss():
    Returns weighted sum of timeapn_mean_loss + timeapn_phase_loss +
    timeapn_amp_loss + timeapn_recon_loss.

This file is a retained compatibility stub.  No trainable parameters live here.
"""
from __future__ import annotations

import torch.nn as nn


class TimeAPNCorrection(nn.Module):
    """Minimal structured TimeAPN path stub — no trainable parameters.

    Carries no computation.  The actual TimeAPN logic lives in SANRouteNorm.
    """

    def __init__(self, pred_len: int = 0, enc_in: int = 0, sigma_min: float = 1e-3, **kwargs):
        super().__init__()
        # No trainable parameters.

    def get_route_diagnostics(self) -> dict:
        return {}
