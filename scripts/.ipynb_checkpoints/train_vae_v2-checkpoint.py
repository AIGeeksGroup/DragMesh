import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import lmdb
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import time
from tqdm import tqdm
import json

from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from utils.balanced_dataset_utils import VAE_LMDBDataset, get_motion_type_weights
from utils.logger import create_logger
from modules.loss import enhanced_dualquat_vae_loss
from modules.model_v2 import DualQuaternionVAE, count_parameters


def custom_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None 
    return torch.utils.data.dataloader.default_collate(batch)


def train_epoch(model, dataloader, optimizer, device, kl_weight=0.0001):
    """
    Train for one epoch.

    Model inputs:
    - mesh: Point cloud [B, N, 3]
    - drag_point, drag_vector: Initial drag interaction
    - joint_type, joint_axis, joint_origin: Joint constraints
    - part_mask: Movable vertices mask

    Joint type conditioning:
    - 0 = revolute: only rotation loss applies
    - 1 = prismatic: only translation loss applies
    """
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_qr_loss = 0
    total_qd_loss = 0
    total_kl_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        if batch is None:
            continue

        # Move data to device
        mesh = batch['initial_mesh'].to(device)
        drag_point = batch['drag_point'].to(device)
        drag_vector = batch['drag_vector'].to(device)
        qr_gt = batch['qr_gt'].to(device)
        qd_gt = batch['qd_gt'].to(device)
        joint_type = batch['joint_type'].to(device)
        joint_axis = batch['joint_axis'].to(device)
        joint_origin = batch['joint_origin'].to(device)
        part_mask = batch['part_mask'].to(device)

        # Forward pass
        pred_qr, pred_qd, mu, logvar = model(
            mesh, drag_point, drag_vector,
            joint_type, joint_axis, joint_origin, part_mask
        )

        # Compute loss with dual-head: separate qr/qd loss based on joint type
        loss_dict = enhanced_dualquat_vae_loss(
            pred_qr, pred_qd, qr_gt, qd_gt, mu, logvar,
            joint_type=joint_type,  # NEW: pass joint_type for dual-head loss
            kl_weight=kl_weight,
            temporal_weight=0.1,
            translation_weight=2.0,
            ortho_weight=0.0,
            norm_weight=0.1,
            first_step_weight=5.0,
            free_bits=2.0,
            use_geodesic=True,
            reduction='mean'
        )

        loss = loss_dict['total_loss']
        recon_loss = loss_dict['recon_loss']
        qr_loss = loss_dict['qr_loss']
        qd_loss = loss_dict['qd_loss']
        kl_loss = loss_dict['kl_loss_raw']

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate losses
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_qr_loss += qr_loss.item()
        total_qd_loss += qd_loss.item()
        total_kl_loss += kl_loss.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'qr': f'{qr_loss.item():.4f}',
            'qd': f'{qd_loss.item():.4f}',
            'kl': f'{kl_loss.item():.4f}'
        })

    # Average losses
    avg_loss = total_loss / num_batches
    avg_recon_loss = total_recon_loss / num_batches
    avg_qr_loss = total_qr_loss / num_batches
    avg_qd_loss = total_qd_loss / num_batches
    avg_kl_loss = total_kl_loss / num_batches

    return {
        'total_loss': avg_loss,
        'recon_loss': avg_recon_loss,
        'qr_loss': avg_qr_loss,
        'qd_loss': avg_qd_loss,
        'kl_loss': avg_kl_loss
    }


def validate(model, dataloader, device, kl_weight=0.0001):
    """
    Validate the model.

    In validation, model operates in non-autoregressive mode:
    - No ground truth guidance
    - Joint priors still influence all output frames
    - Evaluates generalization capability
    """
    model.eval()

    total_loss = 0
    total_recon_loss = 0
    total_qr_loss = 0
    total_qd_loss = 0
    total_kl_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            if batch is None:
                continue

            # Move data to device
            mesh = batch['initial_mesh'].to(device)
            drag_point = batch['drag_point'].to(device)
            drag_vector = batch['drag_vector'].to(device)
            qr_gt = batch['qr_gt'].to(device)
            qd_gt = batch['qd_gt'].to(device)
            joint_type = batch['joint_type'].to(device)
            joint_axis = batch['joint_axis'].to(device)
            joint_origin = batch['joint_origin'].to(device)
            part_mask = batch['part_mask'].to(device)

            # Forward pass (no ground truth guidance)
            pred_qr, pred_qd, mu, logvar = model(
                mesh, drag_point, drag_vector,
                joint_type, joint_axis, joint_origin, part_mask
            )

            # Compute loss
            loss_dict = enhanced_dualquat_vae_loss(
                pred_qr, pred_qd, qr_gt, qd_gt, mu, logvar,
                joint_type=joint_type,  # NEW: dual-head loss
                kl_weight=kl_weight,
                temporal_weight=0.1,
                translation_weight=2.0,
                ortho_weight=0.0,
                norm_weight=0.1,
                first_step_weight=5.0,
                free_bits=2.0,
                use_geodesic=True,
                reduction='mean'
            )

            loss = loss_dict['total_loss']
            recon_loss = loss_dict['recon_loss']
            qr_loss = loss_dict['qr_loss']
            qd_loss = loss_dict['qd_loss']
            kl_loss = loss_dict['kl_loss_raw']

            # Accumulate losses
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_qr_loss += qr_loss.item()
            total_qd_loss += qd_loss.item()
            total_kl_loss += kl_loss.item()
            num_batches += 1

    # Average losses
    avg_loss = total_loss / num_batches
    avg_recon_loss = total_recon_loss / num_batches
    avg_qr_loss = total_qr_loss / num_batches
    avg_qd_loss = total_qd_loss / num_batches
    avg_kl_loss = total_kl_loss / num_batches

    return {
        'total_loss': avg_loss,
        'recon_loss': avg_recon_loss,
        'qr_loss': avg_qr_loss,
        'qd_loss': avg_qd_loss,
        'kl_loss': avg_kl_loss
    }


def main(args):

    print("\n" + "="*70)
    print("Training DualQuaternionVAE with Joint-Prior Influenced Non-AR Decoder")
    print("="*70)
    print("\nKey features:")
    print("  ✓ Non-Autoregressive (Non-AR) decoder: full-sequence prediction")
    print("  ✓ Joint priors directly influence ALL output frames")
    print("  ✓ Drag vector only affects latent initialization")
    print("  ✓ Dual-head loss: rotation vs translation based on joint type")
    print("="*70)

    # 诊断信息
    json_path = args.data_split_json_path 

    if not os.path.exists(json_path):
        print(f"❌ Error: Config file not found: {json_path}")
        return

    with open(json_path, 'r') as f:
        split_config = json.load(f)

    print(f"\nDataset split config: {split_config.get('description', 'N/A')}")
    print(f"Validation strategy: {split_config.get('validation_strategy', 'N/A')}")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger = create_logger(
        output_dir=args.output_dir,
        use_tensorboard=args.use_tensorboard, 
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project if args.use_wandb else None,
        wandb_config=vars(args) if args.use_wandb else None
    )

    # Save training config
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Load datasets
    print(f"\nLoading TRAINING dataset from {args.lmdb_train_path}...")
    train_dataset = VAE_LMDBDataset(lmdb_path=args.lmdb_train_path, augment=True)

    print(f"Loading VALIDATION dataset from {args.lmdb_val_path}...")
    val_dataset = VAE_LMDBDataset(lmdb_path=args.lmdb_val_path, augment=False)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("LMDB dataset(s) are empty! Check paths.")

    print(f"\nDataset loaded:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")

    # Create balanced sampler for training
    print("\nComputing balanced sampling weights...")
    train_weights = get_motion_type_weights(train_dataset, target_ratio=args.motion_balance_ratio)
    train_sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_weights),
        replacement=True
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn,
        drop_last=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    print("\nCreating DualQuaternionVAE model...")
    model = DualQuaternionVAE(
        mesh_feat_dim=1024,
        drag_feat_dim=512,
        latent_dim=args.latent_dim,
        num_frames=args.num_frames,
        joint_type_embed_dim=128
    ).to(device)

    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,} (~{total_params/1e6:.2f}M)")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=args.learning_rate * 0.01
    )

    start_epoch = 0
    best_val_loss = float('inf')

    # Resume from checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"==> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            
            print(f"==> Loaded checkpoint (epoch {checkpoint['epoch']})")
        else:
            print(f"==> No checkpoint found at '{args.resume}'")
            return 

    # Training loop
    print(f"\nStarting training from epoch {start_epoch + 1} to {args.num_epochs}...")
    print("="*70)
    train_history = []
    val_history = []

    for epoch in range(start_epoch, args.num_epochs):
        # KL weight annealing
        current_kl_weight = args.kl_weight * min(1.0, (epoch + 1) / args.kl_anneal_epochs)

        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{args.num_epochs} | KL Weight: {current_kl_weight:.6f}")
        print(f"{'='*70}")

        # Train
        train_losses = train_epoch(
            model, train_loader, optimizer, device, kl_weight=current_kl_weight
        )

        # Validate
        val_losses = validate(
            model, val_loader, device, kl_weight=current_kl_weight
        )

        # Update learning rate
        scheduler.step()

        # Print statistics
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_losses['total_loss']:.4f}")
        print(f"    - Reconstruction: {train_losses['recon_loss']:.4f}")
        print(f"    - QR (Rotation):  {train_losses['qr_loss']:.4f}")
        print(f"    - QD (Translation): {train_losses['qd_loss']:.4f}")
        print(f"    - KL:             {train_losses['kl_loss']:.4f}")
        print(f"  Val Loss: {val_losses['total_loss']:.4f}")
        print(f"    - Reconstruction: {val_losses['recon_loss']:.4f}")
        print(f"    - QR (Rotation):  {val_losses['qr_loss']:.4f}")
        print(f"    - QD (Translation): {val_losses['qd_loss']:.4f}")
        print(f"    - KL:             {val_losses['kl_loss']:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")


        # Log metrics
        logger.log_metrics({
            'total_loss': train_losses['total_loss'],
            'recon_loss': train_losses['recon_loss'],
            'qr_loss': train_losses['qr_loss'],
            'qd_loss': train_losses['qd_loss'],
            'kl_loss': train_losses['kl_loss'],
            'kl_weight': current_kl_weight,
            'lr': optimizer.param_groups[0]['lr']
        }, step=epoch + 1, prefix='train/')

        logger.log_metrics({
            'total_loss': val_losses['total_loss'],
            'recon_loss': val_losses['recon_loss'],
            'qr_loss': val_losses['qr_loss'],
            'qd_loss': val_losses['qd_loss'],
            'kl_loss': val_losses['kl_loss'],
        }, step=epoch + 1, prefix='val/')

        # Save history
        train_history.append({
            'epoch': epoch + 1,
            'total_loss': train_losses['total_loss'],
            'recon_loss': train_losses['recon_loss'],
            'qr_loss': train_losses['qr_loss'],
            'qd_loss': train_losses['qd_loss'],
            'kl_loss': train_losses['kl_loss']
        })

        val_history.append({
            'epoch': epoch + 1,
            'total_loss': val_losses['total_loss'],
            'recon_loss': val_losses['recon_loss'],
            'qr_loss': val_losses['qr_loss'],
            'qd_loss': val_losses['qd_loss'],
            'kl_loss': val_losses['kl_loss']
        })

        # Save best model
        if val_losses['total_loss'] < best_val_loss:
            best_val_loss = val_losses['total_loss']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': best_val_loss
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"  ✓ Saved best model (val_loss: {best_val_loss:.4f})")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_losses['total_loss']
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
            print(f"  ✓ Saved checkpoint")

    # Save final model
    torch.save({
        'epoch': args.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': best_val_loss
    }, os.path.join(args.output_dir, 'final_model.pth'))
    
    # Save training history
    with open(os.path.join(args.output_dir, 'train_history.json'), 'w') as f:
        json.dump(train_history, f, indent=2)
    with open(os.path.join(args.output_dir, 'val_history.json'), 'w') as f:
        json.dump(val_history, f, indent=2)
    
    logger.close()

    print(f"\n{'='*70}")
    print("✅ Training complete!")
    print(f"Best validation loss (in-category): {best_val_loss:.4f}")
    print(f"Models saved to: {args.output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DragMesh V9 with Teacher Forcing")

    # Dataset parameters
    parser.add_argument('--lmdb_train_path', type=str, 
                        default='/root/autodl-tmp/vae_data_train.lmdb/', 
                        help='Path to training LMDB dataset')
    parser.add_argument('--lmdb_val_path', type=str, 
                        default='/root/autodl-tmp/vae_data_val.lmdb/', 
                        help='Path to validation LMDB dataset')
    parser.add_argument('--data_split_json_path', type=str, 
                        default='/root/222/config/category_split_v2.json', 
                        help='Path to dataset split config')
    
    parser.add_argument('--motion_balance_ratio', type=float, default=3.0,
                        help='Target ratio of revolute:prismatic samples')
    parser.add_argument('--num_frames', type=int, default=16,
                        help='Number of trajectory frames')
    parser.add_argument('--num_points', type=int, default=4096,
                        help='Number of points to sample from mesh')

    # Model parameters
    parser.add_argument('--latent_dim', type=int, default=128,  # ✅ V9: 小 z
                        help='Latent dimension (V9: 128 for geometric variations)')

    # Training parameters
    parser.add_argument('--kl_weight', type=float, default=0.1, 
                        help='Maximum KL divergence weight')
    parser.add_argument('--kl_anneal_epochs', type=int, default=50,
                        help='Epochs to ramp up KL weight')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=500,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    
    # Logging
    parser.add_argument('--use_tensorboard', action='store_true', default=True,
                        help='Use TensorBoard for logging')
    parser.add_argument('--use_wandb', action='store_true', default=True,
                        help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='dragmesh-v9-teacher-forcing',
                        help='W&B project name')

    # System parameters
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')
    parser.add_argument('--output_dir', type=str, default='/root/autodl-tmp/outputs/dragmesh_v9',
                        help='Output directory for checkpoints')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to checkpoint to resume from')

    args = parser.parse_args()
    main(args)