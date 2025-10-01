"""
EfficientNet-B1 Training with Medical Augmentation
Phase 5: Medical-specific data augmentation for super-precision

Goal: Improve from 7.23±3.66px to <6.0px using medical augmentation

Features:
- Breathing simulation (diaphragm movement)
- Patient positioning variation (±2° rotation)
- Elastic deformation (tissue simulation)
- Pathology-aware augmentation
- Anatomical constraint validation

Usage:
    python train_efficientnet_medical.py --checkpoint checkpoints/efficientnet/efficientnet_phase4_best.pt
"""

import os
import sys
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from typing import Dict, Tuple
import time
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.efficientnet_regressor import EfficientNetLandmarkRegressor
from src.models.losses import WingLoss, SymmetryLoss, DistancePreservationLoss
from src.data.dataset import LandmarkDataset, create_data_splits
from src.data.medical_transforms import get_medical_transforms


class CompleteLandmarkLoss(nn.Module):
    """
    Complete loss: Wing + Symmetry + Distance Preservation
    Same as Phase 4 but with medical augmentation
    """

    def __init__(self,
                 wing_omega: float = 10.0,
                 wing_epsilon: float = 2.0,
                 symmetry_weight: float = 0.3,
                 distance_weight: float = 0.2):
        super().__init__()

        self.wing_loss = WingLoss(omega=wing_omega, epsilon=wing_epsilon)
        self.symmetry_loss = SymmetryLoss()
        self.distance_loss = DistancePreservationLoss()

        self.symmetry_weight = symmetry_weight
        self.distance_weight = distance_weight

    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor,
                image_size: int = 224) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute complete loss

        Returns:
            (total_loss, loss_components_dict)
        """
        # Wing loss
        wing = self.wing_loss(predictions, targets)

        # Symmetry loss (takes only landmarks, no image_size)
        symmetry = self.symmetry_loss(predictions)

        # Distance preservation loss (takes predictions and targets, no image_size)
        distance = self.distance_loss(predictions, targets)

        # Total loss
        total_loss = wing + self.symmetry_weight * symmetry + self.distance_weight * distance

        loss_components = {
            'wing_loss': wing.item(),
            'symmetry_loss': symmetry.item(),
            'distance_loss': distance.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, loss_components


class EfficientNetMedicalTrainer:
    """
    Trainer for EfficientNet with medical augmentation
    """

    def __init__(self, config_path: str, checkpoint_path: str):
        """
        Args:
            config_path: Path to YAML config file
            checkpoint_path: Path to Phase 4 checkpoint
        """
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.checkpoint_path = checkpoint_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Using device: {self.device}")
        print(f"Loading Phase 4 checkpoint: {checkpoint_path}")

        # Create directories
        self.checkpoint_dir = Path(self.config['checkpoints']['save_dir'])
        self.log_dir = Path(self.config['logging']['log_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Load model from Phase 4 checkpoint
        self.model, checkpoint = EfficientNetLandmarkRegressor.load_from_checkpoint(
            checkpoint_path, device=self.device
        )

        # Move model to device
        self.model = self.model.to(self.device)

        best_val_error = checkpoint.get('best_val_error', checkpoint.get('metrics', {}).get('best_pixel_error', 'N/A'))
        if isinstance(best_val_error, (int, float)):
            print(f"✓ Loaded Phase 4 model (epoch {checkpoint['epoch']}, "
                  f"best val error: {best_val_error:.4f} px)")
        else:
            print(f"✓ Loaded Phase 4 model (epoch {checkpoint['epoch']})")

        # Create dataloaders with MEDICAL AUGMENTATION
        self._create_dataloaders()

        # Setup training components
        self._setup_training()

    def _create_dataloaders(self):
        """Create dataloaders with medical augmentation enabled"""
        print("\n" + "="*80)
        print("Creating dataloaders with MEDICAL AUGMENTATION")
        print("="*80)

        # Create splits
        train_indices, val_indices, test_indices = create_data_splits(
            annotations_file=self.config['data']['coordenadas_path'],
            train_ratio=self.config['data']['train_split'],
            val_ratio=self.config['data']['val_split'],
            test_ratio=self.config['data']['test_split'],
            random_seed=self.config['data']['random_seed']
        )

        # Create transforms with MEDICAL AUGMENTATION
        train_transform = get_medical_transforms(
            image_size=(224, 224),
            is_training=True,
            enable_medical_aug=True,  # ENABLE MEDICAL AUGMENTATION
            validation_tolerance=0.20,
            verbose=False
        )

        val_transform = get_medical_transforms(
            image_size=(224, 224),
            is_training=False,
            enable_medical_aug=False,
            validation_tolerance=0.20,
            verbose=False
        )

        # Create datasets
        train_dataset = LandmarkDataset(
            annotations_file=self.config['data']['coordenadas_path'],
            images_dir=self.config['data']['dataset_path'],
            transform=train_transform,
            indices=train_indices
        )

        val_dataset = LandmarkDataset(
            annotations_file=self.config['data']['coordenadas_path'],
            images_dir=self.config['data']['dataset_path'],
            transform=val_transform,
            indices=val_indices
        )

        test_dataset = LandmarkDataset(
            annotations_file=self.config['data']['coordenadas_path'],
            images_dir=self.config['data']['dataset_path'],
            transform=val_transform,
            indices=test_indices
        )

        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers'],
            pin_memory=self.config['data']['pin_memory'],
            drop_last=True
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            pin_memory=self.config['data']['pin_memory']
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            pin_memory=self.config['data']['pin_memory']
        )

        print(f"\n✓ Dataloaders created with MEDICAL AUGMENTATION:")
        print(f"  Train: {len(self.train_loader)} batches")
        print(f"  Val: {len(self.val_loader)} batches")
        print(f"  Test: {len(self.test_loader)} batches")

        # Show augmentation statistics after first epoch
        self.train_transform = train_transform

    def _setup_training(self):
        """Setup training components"""
        # Use Phase 4 configuration
        phase_config = self.config['training_phase4']

        # Loss function (complete loss)
        self.criterion = CompleteLandmarkLoss(
            wing_omega=self.config['loss']['wing_omega'],
            wing_epsilon=self.config['loss']['wing_epsilon'],
            symmetry_weight=self.config['loss']['symmetry_weight'],
            distance_weight=self.config['loss']['distance_weight']
        )

        # Optimizer with differentiated learning rates
        backbone_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        self.optimizer = optim.Adam([
            {'params': backbone_params, 'lr': phase_config['backbone_lr']},
            {'params': head_params, 'lr': phase_config['head_lr']}
        ], weight_decay=phase_config['weight_decay'])

        # Learning rate scheduler (Cosine Annealing with Warm Restarts)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=phase_config['scheduler_params']['T_0'],
            T_mult=phase_config['scheduler_params']['T_mult'],
            eta_min=phase_config['scheduler_params']['eta_min']
        )

        # Training state
        self.start_epoch = 0
        self.best_val_error = float('inf')
        self.epochs_without_improvement = 0

        print(f"\n✓ Training setup complete")
        print(f"  Backbone LR: {phase_config['backbone_lr']}")
        print(f"  Head LR: {phase_config['head_lr']}")
        print(f"  Loss: Complete (Wing + Symmetry + Distance)")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()

        total_loss = 0.0
        wing_loss = 0.0
        symmetry_loss = 0.0
        distance_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch_idx, (images, targets, metadata) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            predictions = self.model(images)

            # Compute loss
            loss, loss_components = self.criterion(predictions, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update metrics
            total_loss += loss_components['total_loss']
            wing_loss += loss_components['wing_loss']
            symmetry_loss += loss_components['symmetry_loss']
            distance_loss += loss_components['distance_loss']

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_components['total_loss']:.4f}",
                'wing': f"{loss_components['wing_loss']:.4f}",
                'sym': f"{loss_components['symmetry_loss']:.4f}"
            })

        # Step scheduler
        self.scheduler.step()

        # Average metrics
        num_batches = len(self.train_loader)
        metrics = {
            'train_loss': total_loss / num_batches,
            'train_wing_loss': wing_loss / num_batches,
            'train_symmetry_loss': symmetry_loss / num_batches,
            'train_distance_loss': distance_loss / num_batches
        }

        return metrics

    @torch.no_grad()
    def validate(self) -> Tuple[float, Dict[str, float]]:
        """Validate model"""
        self.model.eval()

        all_errors = []
        total_loss = 0.0

        for images, targets, metadata in tqdm(self.val_loader, desc="Validating"):
            images = images.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            predictions = self.model(images)

            # Compute loss
            loss, _ = self.criterion(predictions, targets)
            total_loss += loss.item()

            # Compute pixel errors
            pred_pixels = predictions * 224
            target_pixels = targets * 224

            errors = torch.sqrt(torch.sum((pred_pixels - target_pixels) ** 2, dim=1))
            all_errors.extend(errors.cpu().numpy())

        # Compute metrics
        all_errors = np.array(all_errors)
        mean_error = np.mean(all_errors)
        std_error = np.std(all_errors)
        median_error = np.median(all_errors)

        metrics = {
            'val_loss': total_loss / len(self.val_loader),
            'val_pixel_error': mean_error,
            'val_pixel_error_std': std_error,
            'val_pixel_error_median': median_error
        }

        return mean_error, metrics

    def train(self, num_epochs: int = 80):
        """
        Train model with medical augmentation

        Args:
            num_epochs: Number of epochs to train (default 80, same as Phase 4)
        """
        print(f"\n" + "="*80)
        print(f"PHASE 5: MEDICAL AUGMENTATION TRAINING")
        print(f"="*80)
        print(f"Target: <6.0px mean error (from 7.23px baseline)")
        print(f"Epochs: {num_epochs}")
        print(f"Early stopping patience: {self.config['training_phase4']['early_stopping']['patience']}")
        print("="*80 + "\n")

        for epoch in range(self.start_epoch, num_epochs):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch(epoch + 1)

            # Validate
            val_error, val_metrics = self.validate()

            epoch_time = time.time() - epoch_start

            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{num_epochs} - {epoch_time:.1f}s")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"  Val Pixel Error: {val_error:.4f} ± {val_metrics['val_pixel_error_std']:.4f} px")

            # Check if best model
            if val_error < self.best_val_error:
                improvement = self.best_val_error - val_error
                self.best_val_error = val_error
                self.epochs_without_improvement = 0

                # Save best model
                checkpoint_path = self.checkpoint_dir / "efficientnet_medical_best.pt"
                self._save_checkpoint(checkpoint_path, epoch + 1, val_error, val_metrics)

                print(f"  ✓ New best model! Improvement: {improvement:.4f} px")
                print(f"  ✓ Saved: {checkpoint_path}")
            else:
                self.epochs_without_improvement += 1
                print(f"  No improvement ({self.epochs_without_improvement}/"
                      f"{self.config['training_phase4']['early_stopping']['patience']})")

            # Early stopping check
            if self.epochs_without_improvement >= self.config['training_phase4']['early_stopping']['patience']:
                print(f"\n✗ Early stopping triggered after {epoch + 1} epochs")
                break

        # Show augmentation statistics
        if hasattr(self.train_transform, 'get_stats'):
            stats = self.train_transform.get_stats()
            print(f"\n" + "="*80)
            print("MEDICAL AUGMENTATION STATISTICS")
            print("="*80)
            print(f"Total augmentations: {stats['total_augmentations']}")
            print(f"Validation failure rate: {stats['validation_failure_rate']:.2%}")
            print(f"Breathing simulation rate: {stats['breathing_rate']:.2%}")
            print(f"Positioning variation rate: {stats['positioning_rate']:.2%}")
            print(f"Elastic deformation rate: {stats['elastic_rate']:.2%}")
            print(f"Intensity augmentation rate: {stats['intensity_rate']:.2%}")
            print("="*80 + "\n")

        print(f"\n✓ Training complete!")
        print(f"  Best validation error: {self.best_val_error:.4f} px")
        print(f"  Target: <6.0px")

        if self.best_val_error < 6.0:
            print(f"  ✓ TARGET ACHIEVED! 🎉")
        elif self.best_val_error < 6.5:
            print(f"  ✓ Excellent result! Close to target")
        elif self.best_val_error < 7.0:
            print(f"  ✓ Good improvement over baseline (7.23px)")
        else:
            print(f"  Consider additional training or hyperparameter tuning")

    def _save_checkpoint(self, path: Path, epoch: int, val_error: float, metrics: Dict):
        """Save checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_error': val_error,
            'metrics': metrics,
            'config': self.config
        }, path)


def main():
    parser = argparse.ArgumentParser(description='Train EfficientNet-B1 with Medical Augmentation')
    parser.add_argument('--config', type=str, default='configs/efficientnet_config.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str,
                       default='checkpoints/efficientnet/efficientnet_phase4_best.pt',
                       help='Path to Phase 4 checkpoint')
    parser.add_argument('--epochs', type=int, default=80,
                       help='Number of epochs (default: 80)')

    args = parser.parse_args()

    # Verify checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"✗ Error: Checkpoint not found: {args.checkpoint}")
        print(f"\nYou need to train Phase 4 first:")
        print(f"  python train_efficientnet_phases.py --phase 4")
        print(f"\nOr use --checkpoint to specify a different checkpoint path")
        sys.exit(1)

    # Create trainer
    trainer = EfficientNetMedicalTrainer(
        config_path=args.config,
        checkpoint_path=args.checkpoint
    )

    # Train
    trainer.train(num_epochs=args.epochs)


if __name__ == '__main__':
    main()
