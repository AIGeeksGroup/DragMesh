import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

# ========================================
# 编码器模块（保持不变）
# ========================================

class MultiScaleDragEncoder(nn.Module):
    """
    Enhanced drag control encoder that preserves spatial information.

    Key changes (V10):
    1. Explicitly encode relative geometry: drag_point vs joint_origin
    2. Separate encoding of magnitude and direction
    3. Output both drag features AND relative position encoding

    Rationale:
    - Rotation heavily depends on "WHERE the drag happens relative to joint"
    - Translation depends on "drag direction"
    - Previous design lost spatial relationships
    """
    def __init__(self, drag_dim=6, joint_origin_dim=3, output_dim=512):
        super().__init__()
        # ✅ CHANGE 1: Separate encoding branches for better feature separation
        self.direction_encoder = nn.Sequential(
            nn.Linear(6, 256),  # drag_point + drag_vector
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256)
        )

        # ✅ CHANGE 2: Relative position encoder (drag_point relative to joint_origin)
        self.relative_position_encoder = nn.Sequential(
            nn.Linear(3, 128),  # relative_position = drag_point - joint_origin
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 128)
        )

        # ✅ CHANGE 3: Magnitude encoder (how far and strong the drag is)
        self.magnitude_encoder = nn.Sequential(
            nn.Linear(1, 64),  # drag_vector magnitude
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        self.fusion = nn.Sequential(
            nn.Linear(256 + 128 + 64, output_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, drag_point, drag_vector, joint_origin):
        """
        Args:
            drag_point: [B, 3] start point of drag
            drag_vector: [B, 3] drag displacement
            joint_origin: [B, 3] center of rotation/translation

        Returns:
            drag_feat: [B, output_dim] enriched drag features
        """
        # ✅ Branch 1: Direction (what direction the drag goes)
        direction_input = torch.cat([drag_point, drag_vector], dim=1)  # [B, 6]
        direction_feat = self.direction_encoder(direction_input)  # [B, 256]

        # ✅ Branch 2: Relative position (where relative to joint)
        relative_pos = drag_point - joint_origin  # [B, 3]
        relative_feat = self.relative_position_encoder(relative_pos)  # [B, 128]

        # ✅ Branch 3: Magnitude (strength of interaction)
        drag_magnitude = torch.norm(drag_vector, dim=1, keepdim=True)  # [B, 1]
        magnitude_feat = self.magnitude_encoder(drag_magnitude)  # [B, 64]

        # Fusion
        combined = torch.cat([direction_feat, relative_feat, magnitude_feat], dim=1)  # [B, 448]
        drag_feat = self.fusion(combined)  # [B, output_dim]

        return drag_feat


class PointCloudEncoder(nn.Module):
    """PointNet-style encoder."""
    def __init__(self, input_dim=4, output_dim=1024):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_dim, 128, 1),
            nn.GroupNorm(16, 128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.GroupNorm(32, 256),
            nn.ReLU(),
            nn.Conv1d(256, 1024, 1),
            nn.GroupNorm(64, 1024),
            nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, output_dim),
        )
        self.point_feature_dim = 1024
        
    def forward(self, points):
        x = points.transpose(1, 2)
        point_features = self.mlp1(x)
        global_features = torch.max(point_features, dim=2)[0]
        global_features = self.mlp2(global_features)
        return global_features, point_features


class LocalFeatureSamplerV10(nn.Module):
    """
    Dual local feature sampler (V10 improvement).

    Instead of sampling only near joint_origin, now samples:
    1. Features near joint_origin (rotation center)
    2. Features near drag_point (interaction point)

    This allows model to learn how the relative geometry between
    these two points affects the rotation/translation.
    """
    def __init__(self, k=32, in_features=1024, out_features=512):
        super().__init__()
        self.k = k

        # Separate MLPs for each sampling point
        self.joint_mlp = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, out_features)
        )

        self.drag_mlp = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, out_features)
        )

        # Fusion module to combine both local features
        self.fusion = nn.Sequential(
            nn.Linear(out_features * 2, out_features),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(out_features, out_features)
        )

    def forward(self, points_xyz, point_features, joint_origin, drag_point):
        """
        Args:
            points_xyz: [B, N, 3] point cloud coordinates
            point_features: [B, C, N] encoded point features
            joint_origin: [B, 3] center of rotation
            drag_point: [B, 3] interaction point

        Returns:
            local_feat: [B, out_features] fused local features
        """
        B, C, N = point_features.shape

        # Sample near joint_origin
        joint_origin_expanded = joint_origin.unsqueeze(1)  # [B, 1, 3]
        dist_to_joint = torch.cdist(joint_origin_expanded, points_xyz)  # [B, 1, N]
        knn_joint = dist_to_joint.topk(self.k, dim=2, largest=False).indices.squeeze(1)  # [B, k]
        knn_joint_expanded = knn_joint.unsqueeze(1).expand(-1, C, -1)  # [B, C, k]
        joint_local_feats = torch.gather(point_features, dim=2, index=knn_joint_expanded)  # [B, C, k]
        joint_local_feat = torch.max(joint_local_feats, dim=2).values  # [B, C]
        joint_local_feat = self.joint_mlp(joint_local_feat)  # [B, out_features]

        # Sample near drag_point
        drag_point_expanded = drag_point.unsqueeze(1)  # [B, 1, 3]
        dist_to_drag = torch.cdist(drag_point_expanded, points_xyz)  # [B, 1, N]
        knn_drag = dist_to_drag.topk(self.k, dim=2, largest=False).indices.squeeze(1)  # [B, k]
        knn_drag_expanded = knn_drag.unsqueeze(1).expand(-1, C, -1)  # [B, C, k]
        drag_local_feats = torch.gather(point_features, dim=2, index=knn_drag_expanded)  # [B, C, k]
        drag_local_feat = torch.max(drag_local_feats, dim=2).values  # [B, C]
        drag_local_feat = self.drag_mlp(drag_local_feat)  # [B, out_features]

        # Fuse both local features
        combined = torch.cat([joint_local_feat, drag_local_feat], dim=1)  # [B, 2*out_features]
        local_feat = self.fusion(combined)  # [B, out_features]

        return local_feat


class JointConditionEncoder(nn.Module):
    """Joint condition encoder."""
    def __init__(self, joint_type_embed_dim=128, joint_feat_dim=512):
        super().__init__()
        self.joint_type_embedding = nn.Embedding(2, joint_type_embed_dim)
        self.axis_encoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )
        self.origin_encoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )
        fusion_dim = joint_type_embed_dim + 256 + 256
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, joint_feat_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(joint_feat_dim, joint_feat_dim)
        )
    
    def forward(self, joint_type, joint_axis, joint_origin):
        type_feat = self.joint_type_embedding(joint_type)
        axis_feat = self.axis_encoder(joint_axis)
        origin_feat = self.origin_encoder(joint_origin)
        combined = torch.cat([type_feat, axis_feat, origin_feat], dim=1)
        return self.fusion(combined)


# ========================================
# 主模型: Non-AR VAE with Joint-Prior Influenced Decoder
# ========================================

class DualQuaternionVAE(nn.Module):
    """
    Non-Autoregressive VAE for interactive 3D shape deformation.

    Key features:
    1. Dual-input: Initial mesh state + drag interaction
    2. Joint-aware: Separate handling of rotation vs translation
    3. Non-AR decoder: Full-sequence MLP predicts all frames at once
    4. Joint prior influence: Joint information conditions all output frames
    5. First-frame drag guidance: drag_vector only affects latent initialization
    """
    def __init__(
        self,
        mesh_feat_dim=1024,
        drag_feat_dim=512,
        latent_dim=512,
        num_frames=16,
        joint_type_embed_dim=128,
        include_drag=False,
        use_film=True
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_frames = num_frames

        self.joint_feat_dim = 512
        self.local_feat_dim = 512
        self.decoder_hidden_dim = 1024

        self.mesh_encoder = PointCloudEncoder(input_dim=4, output_dim=mesh_feat_dim)
        # ✅ V10: Enhanced drag encoder with spatial awareness
        self.drag_encoder = MultiScaleDragEncoder(output_dim=drag_feat_dim)
        self.joint_encoder = JointConditionEncoder(
            joint_type_embed_dim=joint_type_embed_dim,
            joint_feat_dim=self.joint_feat_dim
        )
        # ✅ V10: Use improved dual-point sampler
        self.local_sampler = LocalFeatureSamplerV10(
            k=32,
            in_features=1024,
            out_features=self.local_feat_dim
        )

        # VAE encoder: encodes all conditions into latent space
        self.vae_input_dim = (
            self.local_feat_dim +
            self.joint_feat_dim +
            drag_feat_dim
        )
        self.fc_mu = nn.Linear(self.vae_input_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.vae_input_dim, latent_dim)

        # Optional: FiLM conditioning for local features
        if use_film:
            self.film_scale = nn.Linear(self.joint_feat_dim, self.local_feat_dim)
            self.film_shift = nn.Linear(self.joint_feat_dim, self.local_feat_dim)

        # Non-AR decoder: full-sequence prediction with joint prior influence
        # Input: z (latent) + joint_feat (expanded to all frames) + pos_encoding
        # Per frame: latent_dim + joint_feat_dim + decoder_hidden_dim
        # Full sequence (flattened): (latent_dim + joint_feat_dim + decoder_hidden_dim) * num_frames
        per_frame_dim = latent_dim + self.joint_feat_dim + self.decoder_hidden_dim
        self.decoder_mlp = nn.Sequential(
            nn.Linear(per_frame_dim * num_frames, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_frames * 8)  # Flat output T*8
        )

        # Positional encoding: frame-wise time position
        self.pos_encoding = nn.Parameter(torch.randn(1, num_frames, self.decoder_hidden_dim))
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode_features(self, mesh, part_mask, drag_point, drag_vector,
                       joint_type, joint_axis, joint_origin):
        """Encode all input features into drag, joint, and local representations."""
        part_mask_expanded = part_mask.unsqueeze(-1)
        mesh_with_mask = torch.cat([mesh, part_mask_expanded], dim=-1)
        global_mesh_feat, point_features = self.mesh_encoder(mesh_with_mask)

        # ✅ V10: Pass joint_origin to drag encoder for spatial awareness
        drag_feat = self.drag_encoder(drag_point, drag_vector, joint_origin)

        joint_feat = self.joint_encoder(joint_type, joint_axis, joint_origin)

        # ✅ V10: Pass both joint_origin and drag_point to local sampler
        local_feat = self.local_sampler(
            mesh[..., :3],
            point_features,
            joint_origin,
            drag_point
        )

        # Optional FiLM conditioning
        if hasattr(self, 'film_scale'):
            scale = torch.sigmoid(self.film_scale(joint_feat))
            shift = self.film_shift(joint_feat)
            local_feat = local_feat * scale + shift

        return drag_feat, joint_feat, local_feat
    
    def forward(self, mesh, drag_point, drag_vector, joint_type, joint_axis,
                joint_origin, part_mask):
        """
        Forward pass during training.

        Args:
            mesh: Point cloud [B, N, 3]
            drag_point: Starting point of drag [B, 3]
            drag_vector: Drag displacement [B, 3]
            joint_type: Joint type (0=revolute, 1=prismatic) [B]
            joint_axis: Joint rotation axis [B, 3]
            joint_origin: Joint origin point [B, 3]
            part_mask: Which points are movable [B, N]

        Returns:
            pred_qr, pred_qd: Predicted dual quaternions [B, T, 4]
            mu, logvar: Latent distribution parameters
        """
        # Encode all input features
        drag_feat, joint_feat, local_feat = self.encode_features(
            mesh, part_mask, drag_point, drag_vector,
            joint_type, joint_axis, joint_origin
        )

        # Encode to latent space
        encoder_input = torch.cat([local_feat, joint_feat, drag_feat], dim=1)
        mu = self.fc_mu(encoder_input)
        logvar = self.fc_logvar(encoder_input)

        # Sample from latent distribution
        if self.training:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu

        # Non-AR decode: joint_feat influences all frames
        pred_qr, pred_qd = self.decode(z, joint_feat)

        return pred_qr, pred_qd, mu, logvar
    
    def decode(self, z, joint_feat):
        """
        Non-AR decoder: predicts all frames simultaneously.

        Args:
            z: Latent code [B, latent_dim]
            joint_feat: Joint features [B, joint_feat_dim] (influences ALL frames)

        Returns:
            pred_qr: Predicted rotation quaternions [B, T, 4]
            pred_qd: Predicted translation quaternions [B, T, 4]
        """
        batch_size = z.shape[0]

        # Expand latent and joint features to all frames
        # This ensures joint priors influence every frame in the sequence
        z_rep = z.unsqueeze(1).expand(-1, self.num_frames, -1)  # [B, T, latent_dim]
        joint_rep = joint_feat.unsqueeze(1).expand(-1, self.num_frames, -1)  # [B, T, joint_feat_dim]

        # Add learnable positional encoding
        pos = self.pos_encoding.expand(batch_size, -1, -1)  # [B, T, decoder_hidden_dim]
        if self.training:
            joint_rep = F.dropout(joint_rep, p=0.5, training=self.training)

        # Concatenate all features per frame: z + joint_prior + temporal_pos
        # [B, T, latent_dim + joint_feat_dim + decoder_hidden_dim]
        decoder_input = torch.cat([z_rep, joint_rep, pos], dim=-1)  # [B, T, input_dim]
        
        # Flatten to batch: [B, T * input_dim]
        # This processes all T frames together in a single forward pass
        decoder_input_flat = decoder_input.reshape(batch_size, -1)  # [B, T*input_dim]

        # Full-sequence MLP prediction (non-autoregressive)
        flat_output = self.decoder_mlp(decoder_input_flat)  # [B, T*8]

        # Reshape to per-frame dual quaternions
        dualquat_seq = flat_output.view(batch_size, self.num_frames, 8)  # [B, T, 8]

        # Normalize rotation quaternions
        pred_qr = F.normalize(dualquat_seq[..., :4], p=2, dim=-1)  # [B, T, 4]
        pred_qd = dualquat_seq[..., 4:]  # [B, T, 4]

        return pred_qr, pred_qd
    
    def inference(self, mesh, drag_point, drag_vector, joint_type, joint_axis,
                  joint_origin, part_mask):
        """
        Inference mode: deterministic latent code.
        """
        drag_feat, joint_feat, local_feat = self.encode_features(
            mesh, part_mask, drag_point, drag_vector,
            joint_type, joint_axis, joint_origin
        )

        encoder_input = torch.cat([local_feat, joint_feat, drag_feat], dim=1)
        mu = self.fc_mu(encoder_input)
        z = mu  # Use mean as latent code

        # Non-AR decode
        qr, qd = self.decode(z, joint_feat)

        return qr, qd

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    print(f"Model parameters: {count_parameters(model)/1e6:.2f}M")