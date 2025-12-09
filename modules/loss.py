# -------------------------------------------------------------------
#  modules/loss.py
# -------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
import chamfer3D.dist_chamfer_3D
from modules.dual_quaternion import (
    dual_quaternion_norm, quaternion_mul, 
    quaternion_conjugate, quaternion_to_matrix
)

chamfer_dist_calculator = None

def _apply_dual_quaternion(points, qr, qd):
    B, N, _ = points.shape
    T = qr.shape[1]
    
    R = quaternion_to_matrix(qr) # [B, T, 3, 3]
    qr_conj = quaternion_conjugate(qr)
    t_q = 2.0 * quaternion_mul(qd, qr_conj)
    t = t_q[..., 1:] # [B, T, 3]

    R_flat = R.reshape(B * T, 3, 3)
    t_flat = t.reshape(B * T, 1, 3)
    
    points_expanded = points.unsqueeze(1).expand(-1, T, -1, -1) # [B, T, N, 3]
    points_flat = points_expanded.reshape(B * T, N, 3)
    
    rotated_flat = torch.bmm(points_flat, R_flat.transpose(1, 2))
    transformed_flat = rotated_flat + t_flat
    
    return transformed_flat.view(B, T, N, 3)

def _chamfer_distance_loss(pred_points, gt_points, mask=None):
    global chamfer_dist_calculator
    B, T, N, _ = pred_points.shape
    
    pred_flat = pred_points.reshape(B * T, N, 3)
    gt_flat = gt_points.reshape(B * T, N, 3)
    
    if chamfer_dist_calculator is None:
        chamfer_dist_calculator = chamfer3D.dist_chamfer_3D.chamfer_3DDist()
        
    dist1, dist2, _, _ = chamfer_dist_calculator(pred_flat, gt_flat)
    
    if mask is not None:
        mask_flat = mask.unsqueeze(1).expand(-1, T, -1).reshape(B * T, N).float()
        mask_sum = mask_flat.sum(dim=1) + 1e-8
        
        dist1 = dist1 * mask_flat
        dist2 = dist2 * mask_flat
        
        loss1_per_sample = dist1.sum(dim=1) / mask_sum
        loss2_per_sample = dist2.sum(dim=1) / mask_sum
    else:
        loss1_per_sample = dist1.mean(dim=1)
        loss2_per_sample = dist2.mean(dim=1)
        
    cd_loss = (loss1_per_sample + loss2_per_sample).mean()
    return cd_loss

def compute_mesh_recon_loss(pred_qr, pred_qd, gt_qr, gt_qd, initial_mesh, part_mask, first_step_weight=2.0):
    pred_mesh = _apply_dual_quaternion(initial_mesh, pred_qr, pred_qd)
    with torch.no_grad():
        gt_mesh = _apply_dual_quaternion(initial_mesh, gt_qr, gt_qd)
        
    diff = pred_mesh - gt_mesh
    
    if part_mask is not None:
        mask_expanded = part_mask.unsqueeze(1).unsqueeze(-1)
        diff = diff * mask_expanded
        
    per_frame_loss = torch.mean(diff ** 2, dim=(2, 3))
    
    per_frame_loss[:, 0] *= first_step_weight
    
    return torch.mean(per_frame_loss)

def compute_quaternion_loss(pred_qr, gt_qr, first_step_weight=2.0):
    pred_qr = F.normalize(pred_qr, p=2, dim=-1)
    gt_qr = F.normalize(gt_qr, p=2, dim=-1)
    
    dot_product = torch.sum(pred_qr * gt_qr, dim=-1)
    dot_product = torch.clamp(dot_product, -1.0 + 1e-7, 1.0 - 1e-7) 
    
    qr_loss_per_frame = (1.0 - dot_product) / 2.0 
    qr_loss_per_frame = qr_loss_per_frame ** 2
    
    qr_loss_per_frame[:, 0] *= first_step_weight
    
    return torch.mean(qr_loss_per_frame)

def compute_translation_loss(pred_qd, pred_qr, gt_qd, gt_qr, first_step_weight=2.0):
    pred_qr_conj = quaternion_conjugate(pred_qr)
    pred_t_q = 2.0 * quaternion_mul(pred_qd, pred_qr_conj)
    pred_t = pred_t_q[..., 1:]
    
    gt_qr_conj = quaternion_conjugate(gt_qr)
    gt_t_q = 2.0 * quaternion_mul(gt_qd, gt_qr_conj)
    gt_t = gt_t_q[..., 1:]
    
    qd_loss_per_frame = torch.sum((pred_t - gt_t)**2, dim=-1)
    qd_loss_per_frame[:, 0] *= first_step_weight
    
    return torch.mean(qd_loss_per_frame)

def extract_rotation_axis_safe(qr, eps=1e-6):
    w = qr[..., 0:1]
    xyz = qr[..., 1:]
    
    # sin(theta/2)
    sin_half_theta = torch.sqrt(torch.clamp(1.0 - w**2, min=eps))
    
    # axis = xyz / sin(theta/2)
    axis = xyz / (sin_half_theta + eps)
    axis = F.normalize(axis, p=2, dim=-1)
    
    valid_mask = (sin_half_theta.squeeze(-1) > 0.01)
    return axis, valid_mask


def enhanced_dualquat_vae_loss(
    pred_qr: torch.Tensor, pred_qd: torch.Tensor, gt_qr: torch.Tensor, gt_qd: torch.Tensor, 
    mu: torch.Tensor, logvar: torch.Tensor, joint_type: torch.Tensor, joint_axis: torch.Tensor, 
    joint_origin: torch.Tensor, initial_mesh: torch.Tensor, part_mask: torch.Tensor,
    kl_weight: float = 0.01, cd_weight: float = 30.0, mesh_recon_weight: float = 10.0, 
    quat_recon_weight: float = 0.2, constraint_weight: float = 100.0, 
    qd_zero_weight: float = 50.0, qr_identity_weight: float = 10.0, 
    free_bits: float = 48.0, 
) -> Dict[str, torch.Tensor]:

    reduction: str = 'mean'
    B, T = pred_qr.shape[0], pred_qr.shape[1]
    device = pred_qr.device
    
    mesh_recon_loss = compute_mesh_recon_loss(pred_qr, pred_qd, gt_qr, gt_qd, initial_mesh, part_mask, first_step_weight=2.0)
    qr_recon_loss = compute_quaternion_loss(pred_qr, gt_qr, first_step_weight=2.0)
    qd_recon_loss = compute_translation_loss(pred_qd, pred_qr, gt_qd, gt_qr, first_step_weight=2.0)
    
    total_recon_loss = (
        mesh_recon_weight * mesh_recon_loss + 
        quat_recon_weight * (qr_recon_loss + qd_recon_loss)
    )
    
    pred_mesh = _apply_dual_quaternion(initial_mesh, pred_qr, pred_qd)
    with torch.no_grad():
        gt_mesh = _apply_dual_quaternion(initial_mesh, gt_qr, gt_qd)
    cd_loss = _chamfer_distance_loss(pred_mesh, gt_mesh, mask=part_mask)
    
    is_revolute_expanded = (joint_type == 0).float().unsqueeze(1).expand(-1, T)
    is_prismatic_expanded = (joint_type == 1).float().unsqueeze(1).expand(-1, T)
    is_revolute_1d = (joint_type == 0).float()
    is_prismatic_1d = (joint_type == 1).float()
    
    # --- (Revolute/Prismatic) ---
    pred_rot_axis, valid_mask = extract_rotation_axis_safe(pred_qr)
    gt_axis_norm = F.normalize(joint_axis.unsqueeze(1).expand(-1, T, -1), p=2, dim=-1)

    axis_dot = torch.sum(pred_rot_axis * gt_axis_norm, dim=-1).abs()
    loss_axis_alignment = (1.0 - axis_dot) * valid_mask.float()
    axis_cross = torch.cross(pred_rot_axis, gt_axis_norm, dim=-1)
    loss_axis_perpendicularity = torch.sum(axis_cross**2, dim=-1) * valid_mask.float()
    loss_revolute_dir = (loss_axis_alignment * 2.0 + loss_axis_perpendicularity * 0.5)

    pred_qr_conj_phys = quaternion_conjugate(pred_qr)
    pred_t_q_phys = 2.0 * quaternion_mul(pred_qd, pred_qr_conj_phys)
    pred_t_vec_phys = pred_t_q_phys[..., 1:]
    translation_cross_product = torch.cross(pred_t_vec_phys, gt_axis_norm, dim=-1)
    loss_prismatic_dir = torch.sum(translation_cross_product**2, dim=-1)

    constraint_loss_per_frame = (loss_revolute_dir * is_revolute_expanded + loss_prismatic_dir * is_prismatic_expanded)
    axis_constraint_loss = torch.mean(constraint_loss_per_frame)
    
    qd_zero_loss = torch.mean(pred_t_vec_phys ** 2, dim=(1, 2))
    qd_zero_loss = torch.mean(qd_zero_loss * is_revolute_1d)
    
    identity_qr = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).view(1, 1, 4).expand(B, T, -1)
    dot_product_ident = torch.abs(torch.sum(pred_qr * identity_qr, dim=-1))
    dot_product_ident = torch.clamp(dot_product_ident, -1.0 + 1e-7, 1.0 - 1e-7)
    geodesic_dist_ident = 2.0 * torch.acos(dot_product_ident)
    qr_identity_loss = torch.mean(geodesic_dist_ident ** 2, dim=1)
    qr_identity_loss = torch.mean(qr_identity_loss * is_prismatic_1d)

    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss_raw = torch.sum(kl_per_dim, dim=1)
    
    kl_loss_per_sample = kl_loss_raw
    if free_bits > 0:
        free_bits_tensor = torch.full_like(kl_loss_per_sample, free_bits)
        kl_loss_freebits = torch.max(torch.zeros_like(kl_loss_per_sample), 
                                     kl_loss_per_sample - free_bits_tensor)
        kl_loss = kl_loss_freebits.mean()
    else:
        kl_loss = kl_loss_per_sample.mean()

    
    total_loss = (
        total_recon_loss +
        cd_weight * cd_loss +
        constraint_weight * axis_constraint_loss +
        qd_zero_weight * qd_zero_loss +
        qr_identity_weight * qr_identity_loss +
        kl_weight * kl_loss
    )
    
    
    if reduction == 'mean':
        return {
            'total_loss': total_loss,
            'mesh_recon_loss': mesh_recon_loss,
            'qr_recon_loss': qr_recon_loss,
            'qd_recon_loss': qd_recon_loss,
            'recon_loss': total_recon_loss,
            'cd_loss': cd_loss,
            'constraint_loss': axis_constraint_loss,
            'qd_zero_loss': qd_zero_loss,
            'qr_identity_loss': qr_identity_loss,
            'static_loss': torch.tensor(0.0).to(device),
            'hinge_loss': torch.tensor(0.0).to(device),
            'kl_loss': kl_loss,
            'kl_loss_raw': kl_loss_raw.mean(), 
        }
    else:
        raise ValueError(f"Unsupported reduction: {reduction}. Only 'mean' is supported.")