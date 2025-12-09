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
        
        mesh = batch['initial_mesh'].to(device)
        drag_point = batch['drag_point'].to(device)
        drag_vector = batch['drag_vector'].to(device)
        part_mask = batch['part_mask'].to(device) if 'part_mask' in batch else None
        
        gt_joint_type = batch['joint_type'].to(device)
        gt_joint_axis = batch['joint_axis'].to(device)
        gt_joint_origin = batch['joint_origin'].to(device)

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
            
            mesh = batch['initial_mesh'].to(device)
            drag_point = batch['drag_point'].to(device)
            drag_vector = batch['drag_vector'].to(device)
            part_mask = batch['part_mask'].to(device) if 'part_mask' in batch else None
            
            gt_joint_type = batch['joint_type'].to(device)
            gt_joint_axis = batch['joint_axis'].to(device)
            gt_joint_origin = batch['joint_origin'].to(device)
            
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
    

    parser.add_argument('--use_mask', type=bool, default=True)
    parser.add_argument('--use_drag', type=bool, default=True)
    
    # Ablation parameters
    parser.add_argument('--encoder_type', type=str, default='attention', choices=['pointnet', 'attention'])
    parser.add_argument('--head_type', type=str, default='decoupled', choices=['coupled', 'decoupled'])
    parser.add_argument('--predict_type', type=bool, default=True)
    
    parser.add_argument('--use_tensorboard', action='store_true', default=True)
    parser.add_argument('--use_wandb', action='store_true', default=True)
    parser.add_argument('--wandb_project', type=str, default='kpp_ablation') 
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='/root/autodl-tmp/outputs/kpp_ablation') 
    parser.add_argument('--save_every', type=int, default=50)
    parser.add_argument('--resume', type=str, default='')

    args = parser.parse_args()
    main(args)