# -------------------------------------------------------------------
#  modules/predictor_loss.py
# -------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
from typing import Dict

# Geodesic Distance Loss (for axis) 
def geodesic_loss_report(pred_axis: torch.Tensor, gt_axis: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    pred_axis_norm = F.normalize(pred_axis, dim=-1)
    gt_axis_norm = F.normalize(gt_axis, dim=-1)
    cosine_sim = F.cosine_similarity(pred_axis_norm, gt_axis_norm, dim=-1)
    clamped_cosine = torch.clamp(torch.abs(cosine_sim), 0.0, 1.0 - eps)
    return torch.acos(clamped_cosine) 

def compute_predictor_loss(
    pred_type_logits: torch.Tensor or None,
    pred_axis: torch.Tensor,
    pred_origin: torch.Tensor, 
    gt_type: torch.Tensor,
    gt_axis: torch.Tensor,
    gt_origin: torch.Tensor,
    type_weight: float = 1.0,
    axis_weight: float = 1.0, 
    origin_weight: float = 1.0, 
    origin_l1_beta: float = 0.001 # 1mm = 0.001 
) -> Dict[str, torch.Tensor]:
    
    if pred_type_logits is not None:
        loss_type = F.cross_entropy(pred_type_logits, gt_type)
    else:
        loss_type = torch.tensor(0.0, device=pred_axis.device)
    
    loss_axis_raw_per_sample = geodesic_loss_report(pred_axis, gt_axis)
    loss_axis = loss_axis_raw_per_sample.mean()
    
    loss_origin = F.smooth_l1_loss(pred_origin, gt_origin, beta=origin_l1_beta)
    
    report_origin_l1 = F.l1_loss(pred_origin, gt_origin)
    
    total_loss = (
        type_weight * loss_type +
        axis_weight * loss_axis +
        origin_weight * loss_origin
    )
    
    return {
        'total_loss': total_loss,
        'type_loss': loss_type,
        'axis_loss': loss_axis,
        'origin_loss': loss_origin, 
    }