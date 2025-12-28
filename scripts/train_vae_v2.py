# -------------------------------------------------------------------
#  train.py
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


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
chamfer_lib_path = os.path.join(os.path.dirname(__file__), '..', 'ChamferDistancePytorch')
sys.path.insert(0, chamfer_lib_path)


from utils.balanced_dataset_utils import VAE_LMDBDataset, get_motion_type_weights
from utils.logger import create_logger
from modules.loss import enhanced_dualquat_vae_loss
from modules.model_v2 import DualQuaternionVAE, count_parameters 


def custom_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch: 
        return None 
    return torch.utils.data.dataloader.default_collate(batch)


def train_epoch(model, dataloader, optimizer, device, 
                kl_weight, cd_weight, mesh_recon_weight, quat_recon_weight,
                constraint_weight, qd_zero_weight, qr_identity_weight,
                direction_weight, free_bits, epoch):

    model.train()
    
    # Initialize loss meters (including direction loss 'dir').
    losses = {
        'total': 0, 'mesh_recon': 0, 'quat_recon': 0, 'recon': 0,
        'cd': 0, 'constraint': 0, 'qd_zero': 0, 'qr_identity': 0,
        'static': 0, 'hinge': 0, 'kl': 0, 'dir': 0
    }
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}")
    for batch in pbar:
        if batch is None: 
            continue
        
        mesh = batch['initial_mesh'].to(device)
        drag_point = batch['drag_point'].to(device)
        drag_vector = batch['drag_vector'].to(device)
        qr_gt = batch['qr_gt'].to(device)
        qd_gt = batch['qd_gt'].to(device)
        joint_type = batch['joint_type'].to(device)
        joint_axis = batch['joint_axis'].to(device)
        joint_origin = batch['joint_origin'].to(device)
        part_mask = batch['part_mask'].to(device)
        
        # Optional fields (may be absent depending on dataset).
        rotation_direction = batch.get('rotation_direction')
        if rotation_direction is not None:
            rotation_direction = rotation_direction.to(device)
            
        trajectory_vectors = batch.get('trajectory_vectors')
        if trajectory_vectors is not None:
            trajectory_vectors = trajectory_vectors.to(device)
            
        drag_trajectory = batch.get('drag_trajectory')
        if drag_trajectory is not None:
            drag_trajectory = drag_trajectory.to(device)

        # Forward pass (model_v2.forward must accept these optional arguments).
        pred_qr, pred_qd, mu, logvar = model(
            mesh, drag_point, drag_vector, joint_type, joint_axis, 
            joint_origin, part_mask, rotation_direction, trajectory_vectors,
            drag_trajectory 
        )

        # Compute loss (including rotation direction consistency, if provided).
        loss_dict = enhanced_dualquat_vae_loss(
            pred_qr, pred_qd, qr_gt, qd_gt, mu, logvar,
            joint_type=joint_type, joint_axis=joint_axis, joint_origin=joint_origin,
            initial_mesh=mesh, part_mask=part_mask,
            # New arguments
            rotation_direction=rotation_direction,
            direction_weight=direction_weight,
            # Existing arguments
            kl_weight=kl_weight, cd_weight=cd_weight,
            mesh_recon_weight=mesh_recon_weight, quat_recon_weight=quat_recon_weight,
            constraint_weight=constraint_weight, qd_zero_weight=qd_zero_weight,
            qr_identity_weight=qr_identity_weight, free_bits=free_bits, 
        )

        optimizer.zero_grad()
        loss_dict['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        # Accumulate losses
        losses['total'] += loss_dict['total_loss'].item()
        losses['mesh_recon'] += loss_dict['mesh_recon_loss'].item()
        losses['quat_recon'] += (loss_dict['qr_recon_loss'].item() + 
                                 loss_dict['qd_recon_loss'].item())
        losses['recon'] += loss_dict['recon_loss'].item()
        losses['cd'] += loss_dict['cd_loss'].item()
        losses['constraint'] += loss_dict['constraint_loss'].item()
        losses['qd_zero'] += loss_dict['qd_zero_loss'].item()
        losses['qr_identity'] += loss_dict['qr_identity_loss'].item()
        losses['static'] += loss_dict['static_loss'].item() 
        losses['hinge'] += loss_dict['hinge_loss'].item()   
        losses['kl'] += loss_dict['kl_loss'].item() 
        losses['dir'] += loss_dict['direction_loss'].item()  # new
        
        num_batches += 1

        amplitude = torch.norm(batch['drag_vector'], dim=1).mean().item()
        pbar.set_postfix({
            'loss': f'{loss_dict["total_loss"].item():.4f}',
            'mesh': f'{loss_dict["mesh_recon_loss"].item():.4f}',
            'cd': f'{loss_dict["cd_loss"].item():.4f}',
            'dir': f'{loss_dict["direction_loss"].item():.3f}',
            'amp': f'{amplitude:.2f}'
        })

    if num_batches == 0:
        return {k: 0 for k in losses.keys()}
        
    return {k: v / num_batches for k, v in losses.items()}


def validate(model, dataloader, device, 
             kl_weight, cd_weight, mesh_recon_weight, quat_recon_weight,
             constraint_weight, qd_zero_weight, qr_identity_weight,
             direction_weight, free_bits):
    model.eval()
    
    losses = {
        'total': 0, 'mesh_recon': 0, 'quat_recon': 0, 'recon': 0,
        'cd': 0, 'constraint': 0, 'qd_zero': 0, 'qr_identity': 0,
        'static': 0, 'hinge': 0, 'kl': 0, 'dir': 0
    }
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            if batch is None: 
                continue
            
            mesh = batch['initial_mesh'].to(device)
            drag_point = batch['drag_point'].to(device)
            drag_vector = batch['drag_vector'].to(device)
            qr_gt = batch['qr_gt'].to(device)
            qd_gt = batch['qd_gt'].to(device)
            joint_type = batch['joint_type'].to(device)
            joint_axis = batch['joint_axis'].to(device)
            joint_origin = batch['joint_origin'].to(device)
            part_mask = batch['part_mask'].to(device)
            
            rotation_direction = batch.get('rotation_direction')
            if rotation_direction is not None:
                rotation_direction = rotation_direction.to(device)
            trajectory_vectors = batch.get('trajectory_vectors')
            if trajectory_vectors is not None:
                trajectory_vectors = trajectory_vectors.to(device)
            drag_trajectory = batch.get('drag_trajectory')
            if drag_trajectory is not None:
                drag_trajectory = drag_trajectory.to(device)

            pred_qr, pred_qd, mu, logvar = model(
                mesh, drag_point, drag_vector, joint_type, joint_axis,
                joint_origin, part_mask, rotation_direction, trajectory_vectors,
                drag_trajectory 
            )

            loss_dict = enhanced_dualquat_vae_loss(
                pred_qr, pred_qd, qr_gt, qd_gt, mu, logvar,
                joint_type=joint_type, joint_axis=joint_axis, joint_origin=joint_origin,
                initial_mesh=mesh, part_mask=part_mask,
                rotation_direction=rotation_direction,
                direction_weight=direction_weight,
                kl_weight=kl_weight, cd_weight=cd_weight,
                mesh_recon_weight=mesh_recon_weight, quat_recon_weight=quat_recon_weight,
                constraint_weight=constraint_weight, qd_zero_weight=qd_zero_weight,
                qr_identity_weight=qr_identity_weight, free_bits=free_bits,
            )

            losses['total'] += loss_dict['total_loss'].item()
            losses['mesh_recon'] += loss_dict['mesh_recon_loss'].item()
            losses['quat_recon'] += (loss_dict['qr_recon_loss'].item() + 
                                     loss_dict['qd_recon_loss'].item())
            losses['recon'] += loss_dict['recon_loss'].item()
            losses['cd'] += loss_dict['cd_loss'].item()
            losses['constraint'] += loss_dict['constraint_loss'].item()
            losses['qd_zero'] += loss_dict['qd_zero_loss'].item()
            losses['qr_identity'] += loss_dict['qr_identity_loss'].item()
            losses['static'] += loss_dict['static_loss'].item()
            losses['hinge'] += loss_dict['hinge_loss'].item()
            losses['kl'] += loss_dict['kl_loss'].item() 
            losses['dir'] += loss_dict['direction_loss'].item()  # new
            num_batches += 1

    if num_batches == 0:
        return {k: 0 for k in losses.keys()}

    return {k: v / num_batches for k, v in losses.items()}

def main(args):
    print("\n" + "="*70)
    print("Training DualQuaternionVAE")
    print("    (KL, Recon, CD, Constraint, Direction)")
    print("    LR ")
    print("="*70)
    
    if os.path.exists(args.data_split_json_path):
        with open(args.data_split_json_path, 'r') as f: 
            split_config = json.load(f)
        print(f"\nDataset config: {split_config.get('description', 'N/A')}")
    
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
    
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f: 
        json.dump(vars(args), f, indent=2)

    print(f"\nLoading datasets...")
    train_dataset = VAE_LMDBDataset(lmdb_path=args.lmdb_train_path, augment=True)
    val_dataset = VAE_LMDBDataset(lmdb_path=args.lmdb_val_path, augment=False)
    
    if len(train_dataset) == 0 or len(val_dataset) == 0: 
        raise ValueError("Empty dataset!")
    
    print(f"  Training: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")

    if args.motion_type != 'all':
        print(f"\nFiltering for {args.motion_type.upper()} only...")
        target_type = 0 if args.motion_type == 'rotation' else 1
        
        train_indices = [i for i in tqdm(range(len(train_dataset)), desc="Train") 
                         if train_dataset[i] is not None and 
                         train_dataset[i]['joint_type'] == target_type]
        val_indices = [i for i in tqdm(range(len(val_dataset)), desc="Val") 
                       if val_dataset[i] is not None and 
                       val_dataset[i]['joint_type'] == target_type]
        
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
        
        print(f"  Filtered Train: {len(train_dataset)}")
        print(f"  Filtered Val: {len(val_dataset)}")

    if args.motion_type == 'all':
        print("\nComputing balanced sampling weights...")
        train_weights = get_motion_type_weights(train_dataset, 
                                                target_ratio=args.motion_balance_ratio)
        train_sampler = WeightedRandomSampler(
            weights=train_weights, 
            num_samples=len(train_weights), 
            replacement=True
        )
    else: 
        train_sampler = None
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        sampler=train_sampler, 
        shuffle=(train_sampler is None),
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
    
    print("\nCreating model...")
    model = DualQuaternionVAE(
        mesh_feat_dim=1024, 
        drag_feat_dim=512, 
        latent_dim=args.latent_dim,
        num_frames=args.num_frames, 
        joint_type_embed_dim=128, 
        use_film=True,
        transformer_dim=512, 
        transformer_layers=args.transformer_layers, 
        transformer_heads=args.transformer_heads  
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
    best_val_total_loss = float('inf') 
    
    if args.resume and os.path.isfile(args.resume):
        print(f"\nLoading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(" Model loaded")
        except Exception as e: 
            print(f" Model load error: {e}")
        
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except:
                print(" Scheduler state not fully loaded. Skipping.")

        start_epoch = checkpoint.get('epoch', 0)
        best_val_total_loss = checkpoint.get('val_total_loss', checkpoint.get('val_loss', float('inf')))
        print(f" Resumed from epoch {start_epoch}, Best Val Loss: {best_val_total_loss:.4f}")


    print("\n" + "="*70)
    print("Core Loss Weights:")
    print(f"  Mesh Recon:  {args.mesh_recon_weight:.1f}")
    print(f"  CD:          {args.cd_weight:.1f}")
    print(f"  Constraint:  {args.constraint_weight:.1f}")
    print(f"  Direction:   {args.direction_weight:.1f}")
    print(f"  KL:          {args.kl_weight:.4f} (FreeBits: {args.free_bits:.1f})")
    print("="*70)


    print(f"\nStarting training (epochs {start_epoch+1}-{args.num_epochs})...\n")
    
    train_history = []
    val_history = []

    for epoch in range(start_epoch, args.num_epochs):
        if epoch < args.kl_anneal_epochs:
            current_kl = args.kl_weight * (epoch + 1) / args.kl_anneal_epochs
        else:
            current_kl = args.kl_weight

        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{args.num_epochs} | LR: {optimizer.param_groups[0]['lr']:.2e} | KL: {current_kl:.4f}")
        print(f"{'='*70}")


        train_losses = train_epoch(
            model, train_loader, optimizer, device,
            kl_weight=current_kl, cd_weight=args.cd_weight, mesh_recon_weight=args.mesh_recon_weight,
            quat_recon_weight=args.quat_recon_weight, constraint_weight=args.constraint_weight,
            qd_zero_weight=args.qd_zero_weight, qr_identity_weight=args.qr_identity_weight,
            direction_weight=args.direction_weight,  # pass through
            free_bits=args.free_bits, epoch=epoch
        )

        val_losses = validate(
            model, val_loader, device,
            kl_weight=current_kl, cd_weight=args.cd_weight, mesh_recon_weight=args.mesh_recon_weight,
            quat_recon_weight=args.quat_recon_weight, constraint_weight=args.constraint_weight,
            qd_zero_weight=args.qd_zero_weight, qr_identity_weight=args.qr_identity_weight,
            direction_weight=args.direction_weight,  # pass through
            free_bits=args.free_bits
        )
        
        scheduler.step(val_losses['total'])

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train - Total: {train_losses['total']:.4f} | Mesh: {train_losses['mesh_recon']:.4f} | CD: {train_losses['cd']:.4f} | Dir: {train_losses['dir']:.3f} | KL: {train_losses['kl']:.2f}")
        print(f"  Val   - Total: {val_losses['total']:.4f} | Mesh: {val_losses['mesh_recon']:.4f} | CD: {val_losses['cd']:.4f} | Dir: {val_losses['dir']:.3f} | KL: {val_losses['kl']:.2f}")

        logger.log_metrics({
            'total_loss': train_losses['total'], 'mesh_recon_loss': train_losses['mesh_recon'], 
            'cd_loss': train_losses['cd'], 'constraint_loss': train_losses['constraint'], 
            'dir_loss': train_losses['dir'], 'kl_loss': train_losses['kl'],
            'lr': optimizer.param_groups[0]['lr']
        }, step=epoch + 1, prefix='train/')

        logger.log_metrics({
            'total_loss': val_losses['total'], 'mesh_recon_loss': val_losses['mesh_recon'], 
            'cd_loss': val_losses['cd'], 'constraint_loss': val_losses['constraint'], 
            'dir_loss': val_losses['dir'], 'kl_loss': val_losses['kl']
        }, step=epoch + 1, prefix='val/')

        train_history.append({'epoch': epoch + 1, **train_losses})
        val_history.append({'epoch': epoch + 1, **val_losses})

        current_val_total_loss = val_losses['total']
        
        if epoch >= args.kl_anneal_epochs and current_val_total_loss < best_val_total_loss:
            best_val_total_loss = current_val_total_loss
            torch.save({
                'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                'val_total_loss': best_val_total_loss, 'config': vars(args)
            }, os.path.join(args.output_dir, 'best_model.pth')) 
            print(f"  ✓ Saved best model (Total Loss: {best_val_total_loss:.4f})")

        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                'val_total_loss': current_val_total_loss, 'config': vars(args)
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
            print(f"  ✓ Saved checkpoint")
    
    torch.save({
        'epoch': args.num_epochs, 'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_total_loss': best_val_total_loss, 'config': vars(args)
    }, os.path.join(args.output_dir, 'final_model.pth')) 
    
    with open(os.path.join(args.output_dir, 'train_history.json'), 'w') as f:
        json.dump(train_history, f, indent=2)
    with open(os.path.join(args.output_dir, 'val_history.json'), 'w') as f:
        json.dump(val_history, f, indent=2)
    
    logger.close()
    
    print("\n" + "="*70)
    print(" Training complete!")
    print(f"Best val total loss: {best_val_total_loss:.4f}")
    print(f"Saved to: {args.output_dir}")
    print("="*70)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DualQuaternion VAE")

    parser.add_argument('--lmdb_train_path', default='/root/autodl-tmp/vae_train.lmdb')
    parser.add_argument('--lmdb_val_path', default='/root/autodl-tmp/vae_val.lmdb')
    parser.add_argument('--data_split_json_path', type=str, default='/root/222/config/category_split_v2.json')
    parser.add_argument('--motion_balance_ratio', type=float, default=3.0)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--num_points', type=int, default=4096)
    parser.add_argument('--motion_type', type=str, default='all', choices=['all', 'rotation', 'translation'])
    
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--transformer_layers', type=int, default=4, help='Number of layers in the Transformer decoder')
    parser.add_argument('--transformer_heads', type=int, default=8, help='Number of heads in the Transformer decoder')

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lr_patience', type=int, default=15, help='Patience for ReduceLROnPlateau scheduler (epochs)')

    parser.add_argument('--kl_weight', type=float, default=0.001)
    parser.add_argument('--kl_anneal_epochs', type=int, default=80)
    parser.add_argument('--mesh_recon_weight', type=float, default=10.0)
    parser.add_argument('--quat_recon_weight', type=float, default=0.2)
    parser.add_argument('--cd_weight', type=float, default=30.0)
    parser.add_argument('--constraint_weight', type=float, default=100.0)
    parser.add_argument('--qd_zero_weight', type=float, default=50.0)
    parser.add_argument('--qr_identity_weight', type=float, default=10.0)
    parser.add_argument('--free_bits', type=float, default=48.0)
    # Direction weight (rotation direction consistency loss).
    parser.add_argument('--direction_weight', type=float, default=5.0, help='Weight for rotation direction consistency loss')

    parser.add_argument('--use_tensorboard', action='store_true', default=True)
    parser.add_argument('--use_wandb', action='store_true', default=True)
    parser.add_argument('--wandb_project', type=str, default='dragmesh-vae') 
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='/root/autodl-tmp/outputs/dragmesh_vae') 
    parser.add_argument('--save_every', type=int, default=100)
    parser.add_argument('--resume', type=str, default='')

    args = parser.parse_args()
    main(args)