# -------------------------------------------------------------------
# train_predictor.py
# -------------------------------------------------------------------

import sys
import os
import torch
import argparse
from tqdm import tqdm
import json
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import math 

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
chamfer_lib_path = os.path.join(os.path.dirname(__file__), '..', 'ChamferDistancePytorch')
sys.path.insert(0, chamfer_lib_path)

from utils.balanced_dataset_utils import VAE_LMDBDataset, get_motion_type_weights
from utils.logger import create_logger

from modules.predictor import KeypointPredictor 
from modules.predictor_loss import compute_predictor_loss, geodesic_loss_report 

try:
    from modules.model_v2 import count_parameters
except ImportError:
    print("Warning: 'count_parameters' not found in 'modules.model_v2'.")
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

def str2bool(v):
    """
    NOTE: `argparse` with `type=bool` is error-prone (e.g., the string 'False' becomes True).
    This helper supports common CLI boolean strings: True/False/1/0/yes/no.
    """
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")

def _coerce_pointcloud(x: torch.Tensor) -> torch.Tensor:
    """
    Handle common point-cloud layouts from different LMDB formats:
    - [N,3] / [B,N,3]
    - [3,N] / [B,3,N]
    - [N,>=3] keep the first 3 channels
    """
    if x is None:
        return x
    if x.dim() == 2:
        # [N,C] or [C,N]
        if x.shape[0] == 3 and x.shape[1] != 3:
            x = x.transpose(0, 1)  # [N,3]
        if x.shape[-1] > 3:
            x = x[:, :3]
        return x
    if x.dim() == 3:
        # [B,N,C] or [B,C,N]
        if x.shape[1] == 3 and x.shape[2] != 3:
            x = x.transpose(1, 2)  # [B,N,3]
        if x.shape[-1] > 3:
            x = x[:, :, :3]
        return x
    return x

def _get_with_aliases(batch: dict, keys: list[str]):
    for k in keys:
        if k in batch:
            return batch[k]
    return None

def _ensure_batch_fields(batch: dict, device: torch.device):
    """
    Make training robust to dataset / LMDB field-name variations:
    - initial_mesh: required; shape is coerced to [B, N, 3]
    - drag_point/drag_vector/part_mask: default to zeros/ones if missing
    - joint_type/axis/origin: support common aliases with dtype/shape fallbacks
    """
    mesh = _get_with_aliases(batch, ["initial_mesh", "mesh", "points", "pc"])
    if mesh is None:
        raise KeyError("Missing required field: initial_mesh (or mesh/points/pc)")
    mesh = _coerce_pointcloud(mesh).to(device)

    # N
    if mesh.dim() != 3:
        raise ValueError(f"initial_mesh should be [B,N,3], got shape={tuple(mesh.shape)}")
    B, N, _ = mesh.shape

    part_mask = _get_with_aliases(batch, ["part_mask", "mask", "seg", "part_seg"])
    if part_mask is None:
        part_mask = torch.ones((B, N), device=device, dtype=torch.float32)
    else:
        if part_mask.dim() == 1:
            part_mask = part_mask.unsqueeze(0).expand(B, -1)
        part_mask = part_mask.to(device).float()

    drag_point = _get_with_aliases(batch, ["drag_point", "drag_pt", "start_point"])
    if drag_point is None:
        drag_point = torch.zeros((B, 3), device=device, dtype=mesh.dtype)
    else:
        drag_point = drag_point.to(device).float()
        if drag_point.dim() == 1:
            drag_point = drag_point.unsqueeze(0)

    drag_vector = _get_with_aliases(batch, ["drag_vector", "drag_vec", "direction", "drag_dir"])
    if drag_vector is None:
        drag_vector = torch.zeros((B, 3), device=device, dtype=mesh.dtype)
    else:
        drag_vector = drag_vector.to(device).float()
        if drag_vector.dim() == 1:
            drag_vector = drag_vector.unsqueeze(0)

    gt_joint_type = _get_with_aliases(batch, ["joint_type", "gt_joint_type", "type"])
    if gt_joint_type is None:
        # Default to revolute.
        gt_joint_type = torch.zeros((B,), device=device, dtype=torch.long)
    else:
        gt_joint_type = gt_joint_type.to(device)
        if gt_joint_type.dim() == 0:
            gt_joint_type = gt_joint_type.view(1)
        if gt_joint_type.dim() == 2 and gt_joint_type.shape[-1] == 1:
            gt_joint_type = gt_joint_type[:, 0]
        # Accept both float and int.
        gt_joint_type = gt_joint_type.long()

    gt_joint_axis = _get_with_aliases(batch, ["joint_axis", "gt_joint_axis", "axis"])
    if gt_joint_axis is None:
        gt_joint_axis = torch.tensor([0.0, 0.0, 1.0], device=device).view(1, 3).expand(B, 3)
    else:
        gt_joint_axis = gt_joint_axis.to(device).float()
        if gt_joint_axis.dim() == 1:
            gt_joint_axis = gt_joint_axis.unsqueeze(0)
        gt_joint_axis = F.normalize(gt_joint_axis, dim=-1)

    gt_joint_origin = _get_with_aliases(batch, ["joint_origin", "gt_joint_origin", "origin"])
    if gt_joint_origin is None:
        gt_joint_origin = torch.zeros((B, 3), device=device, dtype=torch.float32)
    else:
        gt_joint_origin = gt_joint_origin.to(device).float()
        if gt_joint_origin.dim() == 1:
            gt_joint_origin = gt_joint_origin.unsqueeze(0)

    return mesh, part_mask, drag_point, drag_vector, gt_joint_type, gt_joint_axis, gt_joint_origin


def custom_collate_fn(batch):

    batch = [b for b in batch if b is not None]
    if not batch: 
        return None 
    
    return torch.utils.data.dataloader.default_collate(batch)


def train_epoch(model, dataloader, optimizer, device, 
                type_weight, axis_weight, 
                origin_weight, origin_l1_beta, 
                epoch):
    model.train()
    
    losses = {'total': 0, 'type': 0, 'axis': 0, 'origin': 0} 
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}")
    for batch in pbar:
        if batch is None: 
            continue
        try:
            mesh, part_mask, drag_point, drag_vector, gt_joint_type, gt_joint_axis, gt_joint_origin = _ensure_batch_fields(batch, device=device)
        except Exception as e:
            # Skip malformed samples to avoid interrupting long runs.
            pbar.set_postfix({'skip': type(e).__name__})
            continue

        pred_type_logits, pred_axis, pred_origin = model(
            mesh, part_mask, drag_point, drag_vector
        )

        loss_dict = compute_predictor_loss(
            pred_type_logits, pred_axis, pred_origin,
            gt_type=gt_joint_type, gt_axis=gt_joint_axis, gt_origin=gt_joint_origin,
            type_weight=type_weight,
            axis_weight=axis_weight,
            origin_weight=origin_weight,
            origin_l1_beta=origin_l1_beta 
        )

        optimizer.zero_grad()
        loss_dict['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0) 
        optimizer.step()

        losses['total'] += loss_dict['total_loss'].item()
        losses['type'] += loss_dict['type_loss'].item()
        losses['axis'] += loss_dict['axis_loss'].item() 
        # 'origin_loss' ->  SmoothL1
        losses['origin'] += loss_dict['origin_loss'].item() 
        num_batches += 1

        pbar.set_postfix({
            'loss': f'{loss_dict["total_loss"].item():.4f}',
            'axis': f'{loss_dict["axis_loss"].item():.6f}', 
            'origin(S-L1)': f'{loss_dict["origin_loss"].item():.4f}'
        })

    if num_batches == 0:
        return {k: 0 for k in losses.keys()}
        
    return {k: v / num_batches for k, v in losses.items()}


def validate(model, dataloader, device, 
             type_weight, axis_weight, 
             origin_weight, origin_l1_beta, predict_type): 
    model.eval()
    
    losses = {'total': 0, 'type': 0, 'axis': 0, 'origin': 0}
    num_batches = 0
    
    total_samples = 0
    correct_type = 0
    
    axis_errors_sum = 0.0 # mrad
    origin_errors_sum = 0.0 # mm
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            if batch is None: 
                continue
            try:
                mesh, part_mask, drag_point, drag_vector, gt_joint_type, gt_joint_axis, gt_joint_origin = _ensure_batch_fields(batch, device=device)
            except Exception:
                continue
            
            pred_type_logits, pred_axis, pred_origin = model(
                mesh, part_mask, drag_point, drag_vector
            )
            
            # Loss ( SmoothL1 )
            loss_dict = compute_predictor_loss(
                pred_type_logits, pred_axis, pred_origin,
                gt_type=gt_joint_type, gt_axis=gt_joint_axis, gt_origin=gt_joint_origin,
                type_weight=type_weight,
                axis_weight=axis_weight,
                origin_weight=origin_weight,
                origin_l1_beta=origin_l1_beta 
            )

            losses['total'] += loss_dict['total_loss'].item()
            losses['type'] += loss_dict['type_loss'].item()
            losses['axis'] += loss_dict['axis_loss'].item()
            losses['origin'] += loss_dict['origin_loss'].item()
            
            num_batches += 1
            batch_size = gt_joint_type.size(0)
            total_samples += batch_size
            
            # Type Acc
            if predict_type and pred_type_logits is not None:
                pred_type = pred_type_logits.argmax(dim=1)
                correct_type += (pred_type == gt_joint_type).sum().item()
            # else skip
            
            # Axis Error (mrad)
            angle_rad_per_sample = geodesic_loss_report(pred_axis, gt_joint_axis) 
            angle_mrad_per_sample = angle_rad_per_sample * 1000.0
            axis_errors_sum += angle_mrad_per_sample.sum().item()
            
            # Origin Error (mm) 
            origin_l1_per_sample = F.l1_loss(pred_origin, gt_joint_origin, reduction='none').mean(dim=1) 
            origin_mm_per_sample = (origin_l1_per_sample * 1000)
            origin_errors_sum += origin_mm_per_sample.sum().item()

    if num_batches == 0:
        return {k: 0 for k in losses.keys()}
    
    avg_losses = {k: v / num_batches for k, v in losses.items()}
    
    type_acc = (correct_type / total_samples) * 100 if total_samples > 0 and predict_type else -1  # or 0
    avg_axis_error = axis_errors_sum / total_samples if total_samples > 0 else 0
    avg_origin_error = origin_errors_sum / total_samples if total_samples > 0 else 0
    
    avg_losses['type_acc'] = type_acc
    avg_losses['axis_error_mrad'] = avg_axis_error
    avg_losses['origin_error_mm'] = avg_origin_error
    
    return avg_losses


def main(args):
    print("\n" + "="*70)
    print("Training KPP (Keypoint Predictor) Model - V-Opt-5 (Loss Tuning)")
    print("   V-Opt-3 (Decoupled Heads)")
    print("   Smooth L1 Loss")
    print("   Beta={:.4f} (10mm)".format(args.origin_l1_beta))
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger = create_logger(
        output_dir=args.output_dir, 
        use_tensorboard=args.use_tensorboard, 
        use_wandb=args.use_wandb, 
        wandb_project=args.wandb_project if args.use_wandb else None,
        wandb_config=vars(args) if args.use_wandb else None
    )
    
    with open(os.path.join(args.output_dir, 'config_predictor_kpp.json'), 'w') as f: 
        json.dump(vars(args), f, indent=2)

    print(f"\nLoading datasets...")
    train_dataset = VAE_LMDBDataset(lmdb_path=args.lmdb_train_path, augment=False)
    val_dataset = VAE_LMDBDataset(lmdb_path=args.lmdb_val_path, augment=False)
    
    print(f"  Training: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=custom_collate_fn, 
        drop_last=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.num_workers, pin_memory=True, collate_fn=custom_collate_fn
    )
    
    print("\nCreating model...")
    model = KeypointPredictor(
        use_mask=args.use_mask, 
        use_drag=args.use_drag,
        encoder_type=args.encoder_type,
        head_type=args.head_type,
        predict_type=args.predict_type
    ).to(device)
    
    total_params = count_parameters(model)
    print(f"Parameters: {total_params:,} (~{total_params/1e6:.2f}M)")

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=args.lr_patience, min_lr=1e-7, verbose=True
    )
    
    start_epoch = 0
    best_val_metric = float('inf') 
    
    # --- Resume ---
    if args.resume and os.path.isfile(args.resume):
        print(f"\nLoading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            print(" Model loaded (strict=True)")
        except Exception as e: 
            print(f" Model load error: {e}, attempting strict=False")
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        if 'optimizer_state_dict' in checkpoint:
             optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
             scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        best_val_metric = checkpoint.get('val_origin_error_mm', float('inf'))
        print(f" Resumed from epoch {start_epoch}, Best Origin Error: {best_val_metric:.3f} mm")
    else:
        print(" Starting training from scratch.")
        
    print("\n" + "="*70)
    print("Loss Weights (V-Opt-5):")
    print(f" Type (CE):{args.type_weight:.2f}")
    print(f" Origin (SL1): {args.origin_weight:.2f} (beta={args.origin_l1_beta:.4f})")
    print(f" Axis (Geo): {args.axis_weight:.2f}")
    print("="*70)

    print(f"\nStarting training (epochs {start_epoch+1}-{args.num_epochs})...\n")
    
    for epoch in range(start_epoch, args.num_epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{args.num_epochs} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"{'='*70}")

        train_losses = train_epoch(
            model, train_loader, optimizer, device,
            type_weight=args.type_weight,
            axis_weight=args.axis_weight,
            origin_weight=args.origin_weight,
            origin_l1_beta=args.origin_l1_beta, 
            epoch=epoch
        )

        val_losses = validate(
            model, val_loader, device,
            type_weight=args.type_weight,
            axis_weight=args.axis_weight,
            origin_weight=args.origin_weight,
            origin_l1_beta=args.origin_l1_beta, 
            predict_type=args.predict_type 
        )
        
        scheduler.step(val_losses['origin_error_mm'])

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train - Total: {train_losses['total']:.4f} | Origin(SL1): {train_losses['origin']:.4f}")
        print(f"  Val   - Total: {val_losses['total']:.4f} | Origin(SL1): {val_losses['origin']:.4f}")
        print(f"  Val Metrics - Type Acc: {val_losses['type_acc']:.2f}% | Axis Error: {val_losses['axis_error_mrad']:.3f} mrad | Origin Error: {val_losses['origin_error_mm']:.3f} mm")

        logger.log_metrics({
            'total_loss': train_losses['total'], 'type_loss': train_losses['type'], 
            'axis_loss': train_losses['axis'], 'origin_loss_smoothl1': train_losses['origin'],
            'lr': optimizer.param_groups[0]['lr']
        }, step=epoch + 1, prefix='train/')

        logger.log_metrics({
            'total_loss': val_losses['total'], 'type_loss': val_losses['type'], 
            'axis_loss': val_losses['axis'], 'origin_loss_smoothl1': val_losses['origin'],
            'type_acc': val_losses['type_acc'], 
            'axis_error_mrad': val_losses['axis_error_mrad'], 
            'origin_error_mm': val_losses['origin_error_mm']
        }, step=epoch + 1, prefix='val/')
        
        current_val_metric = val_losses['origin_error_mm']
        
        if current_val_metric < best_val_metric: 
            best_val_metric = current_val_metric
            torch.save({
                'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                'val_origin_error_mm': best_val_metric, 'config': vars(args)
            }, os.path.join(args.output_dir, 'best_model_kpp.pth'))
            print(f"   Saved best model (Origin Error: {best_val_metric:.3f} mm)")

        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                'val_origin_error_mm': current_val_metric, 'config': vars(args)
            }, os.path.join(args.output_dir, f'checkpoint_kpp_epoch_{epoch+1}.pth'))
            print(f"   Saved checkpoint")
    
    logger.close()
    
    print("\n" + "="*70)
    print(" KPP training complete!")
    print(f"Best val origin error: {best_val_metric:.3f} mm")
    print(f"Saved to: {args.output_dir}")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train KPP (Keypoint Predictor)")


    parser.add_argument('--lmdb_train_path', default='/root/autodl-tmp/vae_train.lmdb')
    parser.add_argument('--lmdb_val_path', default='/root/autodl-tmp/vae_val.lmdb')
    parser.add_argument('--data_split_json_path', type=str, default='/root/222/config/category_split_v2.json')
    

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=200) 
    parser.add_argument('--learning_rate', type=float, default=3e-5) 
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lr_patience', type=int, default=15) 


    parser.add_argument('--type_weight', type=float, default=10.0)
    parser.add_argument('--axis_weight', type=float, default=300.0) 
    parser.add_argument('--origin_weight', type=float, default=500.0) 
    

    parser.add_argument('--origin_l1_beta', type=float, default=0.005)  
    

    parser.add_argument('--use_mask', type=str2bool, default=True)
    parser.add_argument('--use_drag', type=str2bool, default=True)
    
    # Ablation parameters
    parser.add_argument('--encoder_type', type=str, default='attention', choices=['pointnet', 'attention'])
    parser.add_argument('--head_type', type=str, default='decoupled', choices=['coupled', 'decoupled'])
    parser.add_argument('--predict_type', type=str2bool, default=True)
    
    parser.add_argument('--use_tensorboard', action='store_true', default=True)
    parser.add_argument('--use_wandb', action='store_true', default=True)
    parser.add_argument('--wandb_project', type=str, default='kpp_ablation') 
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='/root/autodl-tmp/outputs/kpp_ablation') 
    parser.add_argument('--save_every', type=int, default=50)
    parser.add_argument('--resume', type=str, default='')

    args = parser.parse_args()
    main(args)