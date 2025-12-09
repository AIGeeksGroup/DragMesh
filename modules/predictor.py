# -------------------------------------------------------------------
#  modules/predictor.py
# -------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- PointAttentionEncoder ---
class PointAttentionEncoder(nn.Module):
    def __init__(self, input_dim=4, output_dim=1024, 
                 d_model=512, nhead=8, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_dim, 256, 1), nn.GroupNorm(16, 256), nn.ReLU(),
            nn.Conv1d(256, d_model, 1), nn.GroupNorm(32, d_model), nn.ReLU(),
        )
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, 
            dropout=0.1, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_layer, num_layers=num_layers
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(d_model, 1024), nn.LayerNorm(1024), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(1024, output_dim),
        )
        self.pos_encoding = nn.Parameter(torch.randn(1, 4096, d_model))

    def forward(self, points):
        x = points.transpose(1, 2) # [B, C, N]
        point_features = self.mlp1(x) # [B, d_model, N]
        x = point_features.transpose(1, 2) # [B, N, d_model]
        
        B, N, D = x.shape
        if N != self.pos_encoding.shape[1]:
            pos = self.pos_encoding[:, :N, :]
        else:
            pos = self.pos_encoding
        x = x + pos
        
        x = self.transformer_encoder(x) # [B, N, d_model]
        global_features = torch.max(x, dim=1)[0] 
        global_features_out = self.mlp2(global_features) # [B, output_dim]
        return global_features_out, x.transpose(1, 2)

class KeypointPredictor(nn.Module):
    def __init__(self, feat_dim=512, use_mask=True, use_drag=True,
                 encoder_type='attention', head_type='decoupled', predict_type=True):
        super().__init__()
        
        self.use_mask = use_mask
        self.use_drag = use_drag
        self.encoder_type = encoder_type
        self.head_type = head_type
        self.predict_type = predict_type
        
        global_input_dim = 3
        if use_mask: global_input_dim += 1
        if use_drag: global_input_dim += 6
        
        if encoder_type == 'attention':
            self.global_encoder = PointAttentionEncoder(
                input_dim=global_input_dim, output_dim=feat_dim,
                d_model=512, nhead=8, num_layers=2
            )
            self.local_encoder = PointAttentionEncoder(
                input_dim=3, output_dim=feat_dim,
                d_model=512, nhead=8, num_layers=2
            )
        elif encoder_type == 'pointnet':
            self.global_encoder = PointNetEncoder(input_dim=global_input_dim, output_dim=feat_dim)
            self.local_encoder = PointNetEncoder(input_dim=3, output_dim=feat_dim)
        else:
            raise ValueError("Invalid encoder_type")
        
        fusion_input_dim_geom = feat_dim * 2  # 1024 for decoupled geom
        
        if head_type == 'decoupled':
            if self.predict_type:
                # Type Head (only global_feat)
                self.type_fusion_mlp = nn.Sequential(
                    nn.Linear(feat_dim, 256), nn.ReLU(), nn.LayerNorm(256),
                    nn.Linear(256, 128), nn.ReLU(), nn.LayerNorm(128),
                    nn.Dropout(0.1)
                )
                self.type_head = nn.Linear(128, 2)
            
            # Axis Head
            self.axis_fusion_mlp = nn.Sequential(
                nn.Linear(fusion_input_dim_geom, 256), nn.ReLU(), nn.LayerNorm(256),
                nn.Linear(256, 128), nn.ReLU(), nn.LayerNorm(128),
                nn.Dropout(0.1)
            )
            self.axis_head = nn.Linear(128, 3)
            
            # Origin Head
            self.origin_fusion_mlp = nn.Sequential(
                nn.Linear(fusion_input_dim_geom, 256), nn.ReLU(), nn.LayerNorm(256),
                nn.Linear(256, 128), nn.ReLU(), nn.LayerNorm(128),
                nn.Dropout(0.1)
            )
            self.origin_head = nn.Linear(128, 3)
        
        elif head_type == 'coupled':
            # Coupled head: single MLP for all outputs
            input_dim = feat_dim * 2  # global + local
            output_dim = 3 + 3  # axis + origin
            if self.predict_type:
                output_dim += 2  # + type logits
            
            self.fusion_mlp = nn.Sequential(
                nn.Linear(input_dim, 512), nn.ReLU(), nn.LayerNorm(512),
                nn.Linear(512, 256), nn.ReLU(), nn.LayerNorm(256),
                nn.Linear(256, 128), nn.ReLU(), nn.LayerNorm(128),
                nn.Dropout(0.1),
                nn.Linear(128, output_dim)
            )
        else:
            raise ValueError("Invalid head_type")

    
    def forward(self, mesh, part_mask=None, drag_point=None, drag_vector=None):
        
        B, N, _ = mesh.size()
        
        if self.use_mask:
            if part_mask is None:
                part_mask = torch.ones(B, N, device=mesh.device)
        else:
            part_mask = torch.zeros(B, N, device=mesh.device)  # or handle accordingly

        input_feats_global = [mesh]
        if self.use_mask:
            input_feats_global.append(part_mask.unsqueeze(-1))
        if self.use_drag:
            if drag_point is None or drag_vector is None:
                raise ValueError("Drag features required")
            drag_point_rep = drag_point.unsqueeze(1).repeat(1, N, 1)
            drag_vector_rep = drag_vector.unsqueeze(1).repeat(1, N, 1)
            input_feats_global.append(drag_point_rep)
            input_feats_global.append(drag_vector_rep)
        input_mesh_global = torch.cat(input_feats_global, dim=2)
        
        mask_float_expanded = part_mask.float().unsqueeze(-1) if self.use_mask else torch.ones_like(mesh)
        input_mesh_local = mesh * mask_float_expanded
        
        global_feat, _ = self.global_encoder(input_mesh_global)
        local_feat, _ = self.local_encoder(input_mesh_local)

        combined_feat = torch.cat([global_feat, local_feat], dim=1)
        
        if self.head_type == 'decoupled':
            pred_type_logits = None
            if self.predict_type:
                fused_feat_type = self.type_fusion_mlp(global_feat)
                pred_type_logits = self.type_head(fused_feat_type)
            
            fused_feat_axis = self.axis_fusion_mlp(combined_feat)
            pred_axis_raw = self.axis_head(fused_feat_axis)
            pred_axis = F.normalize(pred_axis_raw, dim=1)
            
            fused_feat_origin = self.origin_fusion_mlp(combined_feat)
            pred_origin = self.origin_head(fused_feat_origin)
        
        elif self.head_type == 'coupled':
            outputs = self.fusion_mlp(combined_feat)
            idx = 0
            pred_type_logits = None
            if self.predict_type:
                pred_type_logits = outputs[:, idx:idx+2]
                idx += 2
            pred_axis_raw = outputs[:, idx:idx+3]
            pred_axis = F.normalize(pred_axis_raw, dim=1)
            idx += 3
            pred_origin = outputs[:, idx:idx+3]
        
        return pred_type_logits, pred_axis, pred_origin


class PointNetEncoder(nn.Module):
    def __init__(self, input_dim=4, output_dim=1024):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(input_dim, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 512, 1), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Conv1d(512, output_dim, 1), nn.BatchNorm1d(output_dim), nn.ReLU()
        )
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, points):
        x = points.transpose(1, 2)  # [B, C, N]
        x = self.mlp(x)  # [B, output_dim, N]
        global_feat = self.pool(x).squeeze(-1)  # [B, output_dim]
        return global_feat, x  # Return similar to AttentionEncoder