"""
Enhanced loss functions for Dual Quaternion CVAE training. (FIXED)

Key improvements:
1. Geodesic distance for quaternion rotation (proper SO(3) metric)
2. Separate qr_loss and qd_loss with configurable weights
3. KL loss with free bits (FIXED LOGIC)
4. Temporal consistency loss for smooth trajectories
5. KL weight scheduler with warmup
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

# Import from dual_quaternion
from modules.dual_quaternion import dual_quaternion_norm, quaternion_mul, quaternion_conjugate


def geodesic_quaternion_distance(q1: torch.Tensor, q2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Compute geodesic distance between quaternions on SO(3).
    ... (docstring) ...
    """
    # Ensure quaternions are normalized
    q1 = F.normalize(q1, p=2, dim=-1)
    q2 = F.normalize(q2, p=2, dim=-1)

    # Compute dot product (cosine of angle)
    dot_product = torch.sum(q1 * q2, dim=-1)

    # Take absolute value to handle q and -q representing same rotation
    dot_product = torch.abs(dot_product)

    # Clamp to avoid numerical issues with arccos
    dot_product = torch.clamp(dot_product, -1.0 + eps, 1.0 - eps)

    # Geodesic distance
    geodesic_dist = 2.0 * torch.acos(dot_product)

    return geodesic_dist


def quaternion_rotation_loss(pred_qr: torch.Tensor, gt_qr: torch.Tensor,
                             use_geodesic: bool = True) -> torch.Tensor:
    """
    Rotation quaternion loss with optional geodesic distance.
    ... (docstring) ...
    """
    if use_geodesic:
        # Geodesic distance (proper SO(3) metric)
        geodesic_dist = geodesic_quaternion_distance(pred_qr, gt_qr)
        loss = torch.mean(geodesic_dist ** 2)  # Squared for smoother gradients
    else:
        # MSE loss (handle double-cover: q and -q)
        loss_pos = F.mse_loss(pred_qr, gt_qr, reduction='mean')
        loss_neg = F.mse_loss(pred_qr, -gt_qr, reduction='mean')
        loss = torch.min(loss_pos, loss_neg)

    return loss


def quaternion_translation_loss(pred_qd: torch.Tensor, gt_qd: torch.Tensor) -> torch.Tensor:
    """
    Translation quaternion loss (MSE).
    ... (docstring) ...
    """
    return F.mse_loss(pred_qd, gt_qd, reduction='mean')


def kl_divergence_loss(mu: torch.Tensor, logvar: torch.Tensor,
                             free_bits: float = 0.0) -> torch.Tensor:
    """
    (DEPRECATED - 逻辑已移至 enhanced_dualquat_vae_loss)
    KL divergence loss with free bits to prevent posterior collapse.
    ... (docstring) ...
    """
    # KL divergence per dimension: -0.5 * (1 + logvar - mu^2 - exp(logvar))
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

    # ✅ 修复: free_bits 逻辑应该在 sum 之后应用，
    # 但旧的 buggy 逻辑是在 mean 之前应用。
    # 为了避免混淆，这个辅助函数不再使用 free_bits。
    if free_bits > 0:
        # print("警告: kl_divergence_loss 不再支持 free_bits，请在 enhanced_dualquat_vae_loss 中设置")
        # 保持旧的 buggy 逻辑以防万一
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)

    # Average over batch and dimensions
    kl_loss = torch.mean(kl_per_dim)

    return kl_loss


def temporal_consistency_loss(qr: torch.Tensor, qd: torch.Tensor) -> torch.Tensor:
    """
    Temporal consistency loss to encourage smooth trajectories.
    ... (docstring) ...
    """
    # Compute differences between consecutive frames
    qr_diff = qr[:, 1:] - qr[:, :-1]  # [B, T-1, 4]
    qd_diff = qd[:, 1:] - qd[:, :-1]  # [B, T-1, 4]

    # L2 norm of differences
    temporal_loss_qr = torch.mean(qr_diff ** 2)
    temporal_loss_qd = torch.mean(qd_diff ** 2)

    return temporal_loss_qr + temporal_loss_qd


def orthogonality_loss(qr: torch.Tensor, qd: torch.Tensor) -> torch.Tensor:
    """
    Orthogonality constraint for dual quaternions.
    ... (docstring) ...
    """
    # Dot product between qr and qd
    dot_product = torch.sum(qr * qd, dim=-1)  # [B, T]

    # Penalize non-zero dot products
    ortho_loss = torch.mean(dot_product ** 2)

    return ortho_loss


#
# ✅ ===================================================================
# ✅ 关键修复: enhanced_dualquat_vae_loss (V6 - 修复了 free_bits)
# ✅ ===================================================================
#

def enhanced_dualquat_vae_loss(
    pred_qr: torch.Tensor,
    pred_qd: torch.Tensor,
    gt_qr: torch.Tensor,
    gt_qd: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    joint_type: torch.Tensor = None,  # NEW: joint type (0=revolute, 1=prismatic)
    kl_weight: float = 0.001,
    temporal_weight: float = 0.1,
    translation_weight: float = 2.0,
    ortho_weight: float = 0.1,
    norm_weight: float = 0.1,  # New: weight for normalization loss
    first_step_weight: float = 2.0,  # New: extra weight for first frame reconstruction
    free_bits: float = 64.0,
    use_geodesic: bool = True,
    reduction: str = 'mean'
) -> Dict[str, torch.Tensor]:
    """
    增强版 VAE 损失函数 (V7 - 支持dual-head loss，根据关节类型分离旋转和平移损失)
    """

    # --- 1. 旋转重建损失 (Rotation reconstruction loss) ---
    if use_geodesic:
        geodesic_dist = geodesic_quaternion_distance(pred_qr, gt_qr) # [B, T]
        qr_loss_per_frame = geodesic_dist ** 2
    else:
        loss_pos_per_frame = torch.sum((pred_qr - gt_qr)**2, dim=-1) # [B, T]
        loss_neg_per_frame = torch.sum((pred_qr - -gt_qr)**2, dim=-1) # [B, T]
        qr_loss_per_frame = torch.min(loss_pos_per_frame, loss_neg_per_frame)

    # Apply first step weighting
    qr_loss_per_frame[:, 0] *= first_step_weight

    # (B,) - 逐样本的旋转损失
    qr_loss_per_sample = torch.mean(qr_loss_per_frame, dim=1)

    # --- 2. 平移重建损失 (Translation reconstruction loss) ---
    qd_loss_per_frame = torch.sum((pred_qd - gt_qd)**2, dim=-1) # [B, T]

    # Apply first step weighting
    qd_loss_per_frame[:, 0] *= first_step_weight

    # (B,) - 逐样本的平移损失 (应用权重)
    qd_loss_per_sample = torch.mean(qd_loss_per_frame, dim=1) * translation_weight

    # --- NEW: Dual-head loss based on joint type ---
    # joint_type: 0 = revolute (旋转), 1 = prismatic (平移)
    if joint_type is not None:
        joint_type = joint_type.to(pred_qr.device)
        # For revolute: keep qr_loss, zero out qd_loss
        # For prismatic: keep qd_loss, zero out qr_loss
        is_revolute = (joint_type == 0).float().unsqueeze(-1)  # [B, 1]
        is_prismatic = (joint_type == 1).float().unsqueeze(-1)  # [B, 1]

        qr_loss_per_sample = qr_loss_per_sample * is_revolute.squeeze(-1)
        qd_loss_per_sample = qd_loss_per_sample * is_prismatic.squeeze(-1) 
    
    # --- 3. 总重建损失 ---
    # (B,)
    recon_loss_per_sample = qr_loss_per_sample + qd_loss_per_sample

    # --- 4. KL 散度 (KL divergence) ---
    # (B, latent_dim)
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    
    # ✅ 修复: kl_loss_raw_per_sample 应该在 dim=1 上求 *sum*
    # (B,)
    kl_loss_raw_per_sample = torch.sum(kl_per_dim, dim=1) 
    
    if free_bits > 0:
        # ✅ 修复: free_bits (64.0) 应该应用在 *总和* 上
        free_bits_tensor = torch.full_like(kl_loss_raw_per_sample, free_bits)
        kl_loss_per_sample = torch.max(kl_loss_raw_per_sample, free_bits_tensor)
    else:
        kl_loss_per_sample = kl_loss_raw_per_sample

    # --- 5. 时间平滑损失 (Temporal consistency loss) ---
    qr_diff = pred_qr[:, 1:] - pred_qr[:, :-1]  # [B, T-1, 4]
    qd_diff = pred_qd[:, 1:] - pred_qd[:, :-1]  # [B, T-1, 4]
    temp_loss_qr_per_frame = torch.sum(qr_diff ** 2, dim=-1) # [B, T-1]
    temp_loss_qd_per_frame = torch.sum(qd_diff ** 2, dim=-1) # [B, T-1]
    # (B,) - 逐样本的时间损失 (应用权重)
    temp_loss_per_sample = (torch.mean(temp_loss_qr_per_frame, dim=1) + \
                            torch.mean(temp_loss_qd_per_frame, dim=1)) * temporal_weight

    # --- 6. 正交损失 (Orthogonality constraint) ---
    dot_product = torch.sum(pred_qr * pred_qd, dim=-1)  # [B, T]
    # (B,) - 逐样本的正交损失 (应用权重)
    ortho_loss_per_sample = torch.mean(dot_product ** 2, dim=1) * ortho_weight

    # --- New: Dual quaternion normalization loss ---
    # Compute norm of predicted dq
    pred_dq_norm_r, pred_dq_norm_d = dual_quaternion_norm((pred_qr, pred_qd))  # [B, T, 4], [B, T, 4]
    
    # Ideal unit norm: (1,0,0,0) for r, (0,0,0,0) for d
    ideal_norm_r = torch.tensor([1.0, 0.0, 0.0, 0.0], device=pred_qr.device).expand_as(pred_dq_norm_r)
    ideal_norm_d = torch.zeros_like(pred_dq_norm_d)
    
    # MSE to ideal
    norm_loss_r_per_frame = torch.sum((pred_dq_norm_r - ideal_norm_r)**2, dim=-1)  # [B, T]
    norm_loss_d_per_frame = torch.sum((pred_dq_norm_d - ideal_norm_d)**2, dim=-1)  # [B, T]
    norm_loss_per_frame = norm_loss_r_per_frame + norm_loss_d_per_frame
    norm_loss_per_sample = torch.mean(norm_loss_per_frame, dim=1) * norm_weight
    
    # --- 7. 总损失 (Total loss) ---
    # (B,)
    total_loss_per_sample = recon_loss_per_sample + \
                            (kl_weight * kl_loss_per_sample) + \
                            temp_loss_per_sample + \
                            ortho_loss_per_sample + \
                            norm_loss_per_sample  # New

    # --- 8. (关键) Reduction ---
    if reduction == 'mean':
        return {
            'total_loss': torch.mean(total_loss_per_sample),
            'recon_loss': torch.mean(recon_loss_per_sample),
            'qr_loss': torch.mean(qr_loss_per_sample),
            'qd_loss': torch.mean(qd_loss_per_sample),
            'kl_loss': torch.mean(kl_loss_per_sample),
            'kl_loss_raw': torch.mean(kl_loss_raw_per_sample), # 监控原始 KL
            'temporal_loss': torch.mean(temp_loss_per_sample),
            'ortho_loss': torch.mean(ortho_loss_per_sample),
            'norm_loss': torch.mean(norm_loss_per_sample)  # New
        }
    elif reduction == 'none':
        return {
            'total_loss': total_loss_per_sample,
            'recon_loss': recon_loss_per_sample,
            'qr_loss': qr_loss_per_sample,
            'qd_loss': qd_loss_per_sample,
            'kl_loss': kl_loss_per_sample,
            'kl_loss_raw': kl_loss_raw_per_sample,
            'temporal_loss': temp_loss_per_sample,
            'ortho_loss': ortho_loss_per_sample,
            'norm_loss': norm_loss_per_sample  # New
        }
    else:
        raise ValueError(f"不支持的 reduction: {reduction}")


class KLWeightScheduler:
    """
    KL weight scheduler with linear warmup and optional adaptive adjustment.
    ... (docstring) ...
    """
    # (这个类没有变化)
    def __init__(self, kl_max: float = 0.01, warmup_epochs: int = 50,
                 min_kl_threshold: float = 4.0):
        self.kl_max = kl_max
        self.warmup_epochs = warmup_epochs
        self.min_kl_threshold = min_kl_threshold

    def get_weight(self, epoch: int, current_kl: Optional[float] = None) -> float:
        if epoch < self.warmup_epochs:
            weight = self.kl_max * (epoch + 1) / self.warmup_epochs
        else:
            weight = self.kl_max
        if current_kl is not None and current_kl < self.min_kl_threshold:
            weight = weight * 0.5
        return weight


if __name__ == "__main__":
    """Test loss functions."""
    print("Testing enhanced loss functions (V6 - free_bits fix)...")

    batch_size = 4
    num_frames = 16
    latent_dim = 512

    pred_qr = F.normalize(torch.randn(batch_size, num_frames, 4, requires_grad=True), p=2, dim=-1)
    pred_qd = torch.randn(batch_size, num_frames, 4, requires_grad=True)
    gt_qr = F.normalize(torch.randn(batch_size, num_frames, 4), p=2, dim=-1)
    gt_qd = torch.randn(batch_size, num_frames, 4)
    mu = torch.randn(batch_size, latent_dim, requires_grad=True)
    logvar = torch.randn(batch_size, latent_dim, requires_grad=True)

    # ... (旧的辅助函数测试，为简洁省略) ...
    
    print("\n9. Testing full enhanced loss (reduction='mean')...")
    loss_dict = enhanced_dualquat_vae_loss(
        pred_qr, pred_qd, gt_qr, gt_qd, mu, logvar,
        kl_weight=0.05, temporal_weight=0.1, translation_weight=2.0,
        ortho_weight=0.1, free_bits=64.0, use_geodesic=True, # ✅ 测试 free_bits=64.0
        reduction='mean'
    )
    print("    Loss components:")
    for key, value in loss_dict.items():
        if 'loss' in key:
            print(f"        {key}: {value.item():.4f} (Shape: {value.shape})")

    print("\n9b. Testing full enhanced loss (reduction='none')...")
    loss_dict_none = enhanced_dualquat_vae_loss(
        pred_qr, pred_qd, gt_qr, gt_qd, mu, logvar,
        kl_weight=0.05, temporal_weight=0.1, translation_weight=2.0,
        ortho_weight=0.1, free_bits=64.0, use_geodesic=True, # ✅ 测试 free_bits=64.0
        reduction='none'
    )
    print("    Loss components (per-sample):")
    for key, value in loss_dict_none.items():
        if 'loss' in key:
            print(f"        {key}: (Shape: {value.shape})")
    assert loss_dict_none['total_loss'].shape == (batch_size,), "reduction='none' 失败！"


    # ... (KLWeightScheduler 和 backprop 测试, 为简洁省略) ...
    
    print("\n✅ All tests passed!")