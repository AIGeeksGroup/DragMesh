# -------------------------------------------------------------------
# 文件: modules/model_v2.py
# -------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import math


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_sinusoidal_encoding(num_frames, dim):
    position = torch.arange(num_frames, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) *
                         -(math.log(10000.0) / dim))
    pe = torch.zeros(num_frames, dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # [1, T, dim]


class MultiScaleDragEncoder(nn.Module):
    def __init__(self, drag_dim=6, joint_origin_dim=3, output_dim=512):
        super().__init__()
        self.direction_encoder = nn.Sequential(
            nn.Linear(6, 256), nn.ReLU(),
            nn.LayerNorm(256), nn.Linear(256, 256)
        )
        self.relative_position_encoder = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(),
            nn.LayerNorm(128), nn.Linear(128, 128)
        )
        self.magnitude_encoder = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 64)
        )
        self.fusion = nn.Sequential(
            nn.Linear(256 + 128 + 64, output_dim), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(output_dim, output_dim)
        )

    def forward(self, drag_point, drag_vector, joint_origin):
        direction_input = torch.cat([drag_point, drag_vector], dim=1)
        direction_feat = self.direction_encoder(direction_input)
        
        relative_pos = drag_point - joint_origin
        relative_feat = self.relative_position_encoder(relative_pos)
        
        drag_magnitude = torch.norm(drag_vector, dim=1, keepdim=True)
        magnitude_feat = self.magnitude_encoder(drag_magnitude)
        
        combined = torch.cat([direction_feat, relative_feat, magnitude_feat], dim=1)
        drag_feat = self.fusion(combined)
        return drag_feat

class PointCloudEncoder(nn.Module):
    """PointNet-style """
    def __init__(self, input_dim=4, output_dim=1024):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_dim, 128, 1), nn.GroupNorm(16, 128), nn.ReLU(),
            nn.Conv1d(128, 256, 1), nn.GroupNorm(32, 256), nn.ReLU(),
            nn.Conv1d(256, 1024, 1), nn.GroupNorm(64, 1024), nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(1024, 1024), nn.LayerNorm(1024), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(1024, output_dim),
        )
        self.point_feature_dim = 1024

    def forward(self, points):
        # points: [B, N, C]
        x = points.transpose(1, 2) # [B, C, N]
        point_features = self.mlp1(x) # [B, 1024, N]
        
        # max pooling 
        global_features = torch.max(point_features, dim=2)[0] # [B, 1024]
        global_features = self.mlp2(global_features) # [B, output_dim]
        
        return global_features, point_features

class LocalFeatureSampler(nn.Module):
    def __init__(self, k=32, in_features=1024, out_features=512):
        super().__init__()
        self.k = k
        self.joint_mlp = nn.Sequential(
            nn.Linear(in_features, 512), nn.ReLU(), nn.Linear(512, out_features)
        )
        self.drag_mlp = nn.Sequential(
            nn.Linear(in_features, 512), nn.ReLU(), nn.Linear(512, out_features)
        )
        self.fusion = nn.Sequential(
            nn.Linear(out_features * 2, out_features), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(out_features, out_features)
        )

    def forward(self, points_xyz, point_features, joint_origin, drag_point):
        B, C, N = point_features.shape

        joint_origin_expanded = joint_origin.unsqueeze(1) # [B, 1, 3]
        dist_to_joint = torch.cdist(joint_origin_expanded, points_xyz) # [B, 1, N]
        knn_joint = dist_to_joint.topk(self.k, dim=2, largest=False).indices.squeeze(1) # [B, k]
        
        knn_joint_expanded = knn_joint.unsqueeze(1).expand(-1, C, -1) # [B, C, k]
        joint_local_feats = torch.gather(point_features, dim=2, index=knn_joint_expanded)
        joint_local_feat = torch.max(joint_local_feats, dim=2).values # [B, C]
        joint_local_feat = self.joint_mlp(joint_local_feat)

        drag_point_expanded = drag_point.unsqueeze(1) # [B, 1, 3]
        dist_to_drag = torch.cdist(drag_point_expanded, points_xyz) # [B, 1, N]
        knn_drag = dist_to_drag.topk(self.k, dim=2, largest=False).indices.squeeze(1) # [B, k]
        
        knn_drag_expanded = knn_drag.unsqueeze(1).expand(-1, C, -1) # [B, C, k]
        drag_local_feats = torch.gather(point_features, dim=2, index=knn_drag_expanded)
        drag_local_feat = torch.max(drag_local_feats, dim=2).values # [B, C]
        drag_local_feat = self.drag_mlp(drag_local_feat)

        combined = torch.cat([joint_local_feat, drag_local_feat], dim=1)
        local_feat = self.fusion(combined)
        return local_feat

class JointConditionEncoder(nn.Module):
    def __init__(self, joint_type_embed_dim=128, joint_feat_dim=512):
        super().__init__()
        self.joint_type_embedding = nn.Embedding(2, joint_type_embed_dim) # 0: revolute, 1: prismatic
        
        self.axis_encoder = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 256)
        )
        self.origin_encoder = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 256)
        )
        
        fusion_dim = joint_type_embed_dim + 256 + 256
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, joint_feat_dim), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(joint_feat_dim + 256, joint_feat_dim) 
        )

    def forward(self, joint_type, joint_axis, joint_origin):
        type_feat = self.joint_type_embedding(joint_type)
        axis_feat = self.axis_encoder(joint_axis)
        origin_feat = self.origin_encoder(joint_origin)
        
        combined = torch.cat([type_feat, axis_feat, origin_feat], dim=1)
        
        x = self.fusion[0](combined); x = self.fusion[1](x); x = self.fusion[2](x)
        x = torch.cat([x, origin_feat], dim=1)
        return self.fusion[3](x)


class DualQuaternionVAE(nn.Module):
    """
    A dual quaternion variational autoencoder (VAE) for generating articulated motion.
    Architecture:
        1. Multimodal encoder (Mesh, Joint, Motion Intent)
        2. Learnable Feature Fusion Gate
        3. Conditional VAE (CVAE) encoder
        4. Conditional Transformer decoder (FiLM injection)
        5. Physics Correction module
    """
    def __init__(
        self,
        mesh_feat_dim=1024,
        drag_feat_dim=512,
        latent_dim=512,
        num_frames=16,
        joint_type_embed_dim=128,
        use_film=True,
        transformer_dim=512,
        transformer_layers=8,
        transformer_heads=8
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_frames = num_frames
        self.joint_feat_dim = 512
        self.local_feat_dim = 512
        self.drag_feat_dim = drag_feat_dim
        self.decoder_hidden_dim = transformer_dim
        
        self.register_buffer('pos_encoding',
                             get_sinusoidal_encoding(num_frames, self.decoder_hidden_dim))

        self.mesh_encoder = PointCloudEncoder(input_dim=4, output_dim=mesh_feat_dim)
        self.drag_encoder = MultiScaleDragEncoder(output_dim=drag_feat_dim)
        self.joint_encoder = JointConditionEncoder(
            joint_type_embed_dim=joint_type_embed_dim,
            joint_feat_dim=self.joint_feat_dim
        )
        self.local_sampler = LocalFeatureSampler(
            k=32, in_features=1024, out_features=self.local_feat_dim
        )

        self.rotation_dir_encoder = nn.Sequential(
            nn.Linear(3, 256), nn.ReLU(),
            nn.Linear(256, drag_feat_dim)
        )
        self.trajectory_vec_encoder = nn.Sequential(
            nn.Linear(3, 256), nn.ReLU(),
            nn.Linear(256, drag_feat_dim)
        )
        self.amplitude_encoder = nn.Sequential(
            nn.Linear(1, 128), nn.ReLU(),
            nn.Linear(128, drag_feat_dim) 
        )

        self.trajectory_embed_dim = 512
        trajectory_transformer_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=512, batch_first=True)
        self.trajectory_encoder = nn.Sequential(
            nn.Linear(3, 128),  # -> embed dim
            nn.TransformerEncoder(trajectory_transformer_layer, num_layers=2),
            nn.Linear(128, self.trajectory_embed_dim)  
        )


        
        #  (local, joint, motion)
        # motion_feature (512) + trajectory_feature (512) = 1024
        # local_feat (512) + joint_feat (512) + motion_feature (1024) = 2048
        self.fusion_input_dim = self.local_feat_dim + self.joint_feat_dim + self.drag_feat_dim + self.trajectory_embed_dim
        self.feature_fusion_gate = nn.Sequential(
            nn.Linear(self.fusion_input_dim, transformer_dim),
            nn.ReLU(),
            nn.Linear(transformer_dim, transformer_dim),
        )
        
        # VAE : Fused_Motion_Feat (512) + Joint_Feat (512) = 1024
        self.vae_input_dim = transformer_dim + self.joint_feat_dim

        # --- VAE ---
        self.fc_mu = nn.Linear(self.vae_input_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.vae_input_dim, latent_dim)

        # --- FiLM (Feature-wise Linear Modulation) ---
        if use_film:
            #  Local Features
            self.film_scale = nn.Linear(self.joint_feat_dim, self.local_feat_dim)
            self.film_shift = nn.Linear(self.joint_feat_dim, self.local_feat_dim)

        self.decoder_joint_influence_scale = 10.0
        
        # decode  torch.cat is:
        # [z_rep, joint_influence, joint_rep, joint_influence, pos]
        # dim is:
        # latent_dim + joint_feat_dim + joint_feat_dim + joint_feat_dim + decoder_hidden_dim
        self.decoder_input_dim = (latent_dim + 
                                  (self.joint_feat_dim * 3) + 
                                  self.decoder_hidden_dim)
        
        self.decoder_input_projection = nn.Linear(self.decoder_input_dim, transformer_dim)

        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim, nhead=transformer_heads,
            dim_feedforward=transformer_dim * 4, dropout=0.1, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_encoder_layer, num_layers=transformer_layers
        )

        self.decoder_output_mlp = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim // 2), nn.ReLU(),
            nn.Linear(transformer_dim // 2, 8) # (qr + qd)
        )
        
        with torch.no_grad():
            if hasattr(self.decoder_output_mlp[-1], 'bias') and self.decoder_output_mlp[-1].bias is not None:
                self.decoder_output_mlp[-1].bias[:4].fill_(0.1)
                self.decoder_output_mlp[-1].bias[4:].fill_(0.0)

        self.physics_type_embed_dim = 128
        self.physics_type_embedding = nn.Embedding(2, self.physics_type_embed_dim)
        
        self.physics_correction_scale = 2.0
        self.physics_joint_feat_scale = 5.0
        self.physics_joint_axis_scale = 10.0
        

        # decode fun torch.cat is:
        # [output_dq_seq, (joint_feat * scale), (joint_axis * scale), joint_feat, joint_axis, type_feat]
        # dim is:
        # 8 + joint_feat_dim + 3 + joint_feat_dim + 3 + physics_type_embed_dim
        physics_input_dim = (8 + 
                             (self.joint_feat_dim * 2) + 
                             (3 * 2) + 
                             self.physics_type_embed_dim)
        
        self.physics_correction = nn.Sequential(
            nn.Linear(physics_input_dim, transformer_dim),
            nn.ReLU(),
            nn.LayerNorm(transformer_dim),
            nn.Linear(transformer_dim, transformer_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(transformer_dim // 2),
            nn.Linear(transformer_dim // 2, transformer_dim // 4),
            nn.ReLU(),
            nn.Linear(transformer_dim // 4, 8),
            nn.Tanh()
        )
        
        # Transformer with FiLM generation
        self.transformer_layers_num = transformer_layers
        self.film_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.joint_feat_dim, transformer_dim * 2), 
                nn.ReLU()
            ) for _ in range(self.transformer_layers_num)
        ])
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar); eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode_features(self, mesh, part_mask, drag_point, drag_vector,
                        joint_type, joint_axis, joint_origin, rotation_direction=None, 
                        trajectory_vectors=None, drag_trajectory=None):
        
        # 1.  Mesh
        part_mask_expanded = part_mask.unsqueeze(-1)
        mesh_with_mask = torch.cat([mesh, part_mask_expanded], dim=-1)
        global_mesh_feat, point_features = self.mesh_encoder(mesh_with_mask)

        is_revolute = (joint_type == 0).float().unsqueeze(-1)
        is_prismatic = (joint_type == 1).float().unsqueeze(-1)
        
        # 2. Motion Intent
        
        # 2a.String features (for Prismatic)
        drag_feat_chord = self.drag_encoder(drag_point, drag_vector, joint_origin)
        
        # 2b. Rotation direction characteristics
        rotation_feat = torch.zeros_like(drag_feat_chord)
        if rotation_direction is not None:
            if rotation_direction.shape[0] != drag_point.shape[0]:
                 rotation_direction_expanded = rotation_direction.unsqueeze(0).expand(drag_point.shape[0], -1)
            else:
                 rotation_direction_expanded = rotation_direction
            rotation_feat = self.rotation_dir_encoder(rotation_direction_expanded)

        # 2c. Average velocity characteristics
        trajectory_vec_feat = torch.zeros_like(drag_feat_chord)
        if trajectory_vectors is not None:
            if trajectory_vectors.shape[0] != drag_point.shape[0]:
                trajectory_vectors_expanded = trajectory_vectors.unsqueeze(0).expand(drag_point.shape[0], -1, -1) if trajectory_vectors.dim() == 2 else trajectory_vectors.unsqueeze(0).expand(drag_point.shape[0], -1, -1, -1)
            else:
                trajectory_vectors_expanded = trajectory_vectors
            
            if trajectory_vectors_expanded.dim() == 3:
                trajectory_vectors_expanded = trajectory_vectors_expanded.mean(dim=1) # [B, 3]

            trajectory_vec_feat = self.trajectory_vec_encoder(trajectory_vectors_expanded)
            
        # 2d. Drag trajectory characteristics (Transformer)
        trajectory_feat = torch.zeros(drag_point.shape[0], self.trajectory_embed_dim, device=drag_point.device)
        if drag_trajectory is not None:
            # [B, T, 3] -> embed
            traj_proj = self.trajectory_encoder[0](drag_trajectory)  # [B, T, 128]
            traj_encoded = self.trajectory_encoder[1](traj_proj)  # Transformer [B, T, 128]
            traj_pooled = traj_encoded.mean(dim=1)  # [B, 128]
            trajectory_feat = self.trajectory_encoder[2](traj_pooled)  # [B, 512]
            
        # 2e. Amplitude Characteristics (General)
        amplitude = torch.norm(drag_vector, dim=1, keepdim=True)
        amplitude_feat = self.amplitude_encoder(amplitude)
        
        motion_feature = (
            (rotation_feat + trajectory_vec_feat) * is_revolute +
            drag_feat_chord * is_prismatic
        )
        motion_feature += amplitude_feat
        
        #  512  + 512  = 1024
        motion_feature_combined = torch.cat([motion_feature, trajectory_feat], dim=1)

        joint_feat = self.joint_encoder(joint_type, joint_axis, joint_origin)

        local_feat = self.local_sampler(
            mesh[..., :3], point_features, joint_origin, drag_point
        )
        if hasattr(self, 'film_scale'):
            scale = torch.sigmoid(self.film_scale(joint_feat))
            shift = self.film_shift(joint_feat)
            local_feat = local_feat * scale + shift

        # 5. Fusion (using learnable gating)
        fusion_input = torch.cat([local_feat, joint_feat, motion_feature_combined], dim=1)
        fused_motion_feat = self.feature_fusion_gate(fusion_input)
        combined_condition = torch.cat([fused_motion_feat, joint_feat], dim=1) 
        
        return combined_condition, joint_feat

    def forward(self, mesh, drag_point, drag_vector, joint_type, joint_axis,
                joint_origin, part_mask, rotation_direction=None, 
                trajectory_vectors=None, drag_trajectory=None):
        
        combined_condition, joint_feat = self.encode_features(
            mesh, part_mask, drag_point, drag_vector,
            joint_type, joint_axis, joint_origin, rotation_direction, 
            trajectory_vectors, drag_trajectory
        )
        
        encoder_input = combined_condition 

        mu = self.fc_mu(encoder_input); logvar = self.fc_logvar(encoder_input)
        
        if self.training:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu 
            
        pred_qr, pred_qd = self.decode(z, joint_feat, joint_axis, joint_type)
        
        return pred_qr, pred_qd, mu, logvar
    
    def decode(self, z, joint_feat, joint_axis, joint_type):
        batch_size = z.shape[0]
        transformer_dim = self.decoder_hidden_dim

        z_rep = z.unsqueeze(1).expand(-1, self.num_frames, -1)
        joint_rep = joint_feat.unsqueeze(1).expand(-1, self.num_frames, -1)
        pos = self.pos_encoding.expand(batch_size, -1, -1).to(z.device)

        joint_influence = joint_rep * self.decoder_joint_influence_scale
        
        # [z_rep, joint_influence, joint_rep, joint_influence, pos]
        decoder_input = torch.cat([z_rep, joint_influence, joint_rep, joint_influence, pos], dim=-1)

        x = self.decoder_input_projection(decoder_input) # [B, T, D_transformer]
        
        # 2. Transformer (FiLM )
        for i in range(self.transformer_layers_num):
            #  FiLM 
            film_params = self.film_generators[i](joint_feat)
            scale = film_params[:, :transformer_dim]
            shift = film_params[:, transformer_dim:]
            
            scale = scale.unsqueeze(1) + 1.0 # [B, 1, D]
            shift = shift.unsqueeze(1)       # [B, 1, D]
            
            x = self.transformer_encoder.layers[i](x)
            x = x * scale + shift
        
        output_dq_seq = self.decoder_output_mlp(x) # [B, T, 8]

        batch_size, seq_len = output_dq_seq.shape[0], output_dq_seq.shape[1]
        joint_feat_expanded = joint_feat.unsqueeze(1).expand(-1, seq_len, -1)
        joint_axis_expanded = joint_axis.unsqueeze(1).expand(-1, seq_len, -1)
        joint_axis_expanded = F.normalize(joint_axis_expanded, p=2, dim=-1)
        
        joint_type_expanded = joint_type.unsqueeze(1).expand(-1, seq_len)
        type_feat_expanded = self.physics_type_embedding(joint_type_expanded)

        # [output_dq_seq, (joint_feat * scale), (joint_axis * scale), joint_feat, joint_axis, type_feat]
        correction_input = torch.cat([
            output_dq_seq,
            joint_feat_expanded * self.physics_joint_feat_scale,
            joint_axis_expanded * self.physics_joint_axis_scale,
            joint_feat_expanded,
            joint_axis_expanded,
            type_feat_expanded
        ], dim=-1)

        correction_output = self.physics_correction(correction_input.view(-1, correction_input.shape[-1]))
        correction_output = correction_output.view(batch_size, seq_len, 8)

        refined_dq_seq = output_dq_seq + correction_output * self.physics_correction_scale
        
        pred_qr = F.normalize(refined_dq_seq[..., :4], p=2, dim=-1)
        pred_qd = refined_dq_seq[..., 4:]

        return pred_qr, pred_qd
    
    def inference(self, mesh, drag_point, drag_vector, joint_type, joint_axis,
                  joint_origin, part_mask, rotation_direction=None, 
                  trajectory_vectors=None, drag_trajectory=None):
        
        combined_condition, joint_feat = self.encode_features(
            mesh, part_mask, drag_point, drag_vector,
            joint_type, joint_axis, joint_origin, rotation_direction, 
            trajectory_vectors, drag_trajectory
        )
        
        encoder_input = combined_condition
        mu = self.fc_mu(encoder_input)
        z = mu
        
        qr, qd = self.decode(z, joint_feat, joint_axis, joint_type)
        
        return qr, qd