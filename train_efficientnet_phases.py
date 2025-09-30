#!/usr/bin/env python3
"""
EfficientNet-B1 4-Phase Training Pipeline
Pipeline progresivo optimizado para precisión sub-píxel

Phase 1: Freeze Backbone + MSE Loss → 47.87px
Phase 2: Fine-tuning + Wing Loss → 8.20px
Phase 3: Wing + Symmetry Loss → 7.65px
Phase 4: Complete Loss (Wing + Symmetry + Distance) → 7.12px (val) / 7.23px (test)

Target con Medical Augmentation: <6.0px (super-precisión clínica)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import time
import argparse
import yaml
from typing import Dict, Optional, Tuple

# Importar módulos del proyecto
from src.data.dataset import create_dataloaders
from src.models.efficientnet_regressor import EfficientNetLandmarkRegressor
from src.models.losses import WingLoss, SymmetryLoss, DistancePreservationLoss, CompleteLandmarkLoss
from src.training.utils import setup_device


class EfficientNetPhaseTrainer:
    """
    Trainer para pipeline 4-phase de EfficientNet-B1

    Sigue patrón exitoso de ResNet pero con ajustes para EfficientNet:
    - Learning rates 50% menores
    - Más épocas (+20%)
    - Cosine annealing warm restarts
    """

    def __init__(self, config_path: str = "configs/efficientnet_config.yaml"):
        """
        Args:
            config_path: Ruta al archivo de configuración
        """
        # Cargar configuración
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Configurar device
        self.device = setup_device(
            use_gpu=self.config['device']['use_gpu'],
            gpu_id=self.config['device']['gpu_id']
        )

        # Configurar semillas para reproducibilidad
        seed = self.config['reproducibility']['seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        print("=" * 60)
        print("🚀 EFFICIENTNET-B1 4-PHASE TRAINING PIPELINE")
        print("=" * 60)
        print(f"⚡ Device: {self.device}")
        print(f"🌱 Seed: {seed}")
        print(f"📁 Config: {config_path}")
        print("=" * 60)

    def create_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Crear data loaders"""
        data_config = self.config['data']

        print("\n📊 Configurando data loaders...")

        train_loader, val_loader, test_loader = create_dataloaders(
            annotations_file=data_config['coordenadas_path'],
            images_dir=data_config['dataset_path'],
            batch_size=data_config['batch_size'],
            num_workers=data_config['num_workers'],
            pin_memory=data_config['pin_memory'],
            train_ratio=data_config['train_split'],
            val_ratio=data_config['val_split'],
            test_ratio=data_config['test_split'],
            random_seed=data_config['random_seed']
        )

        print(f"✓ Train: {len(train_loader.dataset)} samples ({len(train_loader)} batches)")
        print(f"✓ Val: {len(val_loader.dataset)} samples ({len(val_loader)} batches)")
        print(f"✓ Test: {len(test_loader.dataset)} samples ({len(test_loader)} batches)")

        return train_loader, val_loader, test_loader

    def create_model(self, pretrained: bool = True, freeze_backbone: bool = True) -> EfficientNetLandmarkRegressor:
        """Crear modelo EfficientNet-B1"""
        model_config = self.config['model']

        model = EfficientNetLandmarkRegressor(
            num_landmarks=model_config['num_landmarks'],
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout_rate=model_config['dropout_rate'],
            use_coordinate_attention=model_config['use_coordinate_attention'],
            attention_reduction=model_config['attention_reduction']
        )

        return model.to(self.device)

    def compute_pixel_error(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calcular error en píxeles"""
        pred_reshaped = predictions.view(-1, 15, 2)
        target_reshaped = targets.view(-1, 15, 2)
        distances = torch.norm(pred_reshaped - target_reshaped, dim=2)
        pixel_distances = distances * 224  # Convert normalized to pixels
        return torch.mean(pixel_distances)

    # ========================================================================
    # PHASE 1: FREEZE BACKBONE + MSE LOSS
    # ========================================================================

    def train_phase1(self) -> str:
        """
        Phase 1: Entrenar solo cabeza con backbone congelado

        Loss: MSE simple
        Target: ~47.87px (warm-up phase)

        Returns:
            Path al checkpoint guardado
        """
        print("\n" + "=" * 60)
        print("📍 PHASE 1: FREEZE BACKBONE + MSE LOSS")
        print("=" * 60)

        phase_config = self.config['training_phase1']

        # Data loaders
        train_loader, val_loader, _ = self.create_data_loaders()

        # Modelo con backbone congelado
        print("\n🏗️ Creando modelo EfficientNet-B1...")
        model = self.create_model(pretrained=True, freeze_backbone=True)

        model_info = model.get_model_info()
        print(f"✓ Total parámetros: {model_info['total_parameters']:,}")
        print(f"✓ Entrenables: {model_info['trainable_parameters']:,}")
        print(f"✓ Backbone features: {model_info['backbone_features']}")

        # Loss y optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=phase_config['learning_rate'],
            weight_decay=phase_config['weight_decay']
        )

        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=phase_config['scheduler_params']['step_size'],
            gamma=phase_config['scheduler_params']['gamma']
        )

        print(f"\n⚙️ Configuración:")
        print(f"  - Epochs: {phase_config['epochs']}")
        print(f"  - Learning Rate: {phase_config['learning_rate']}")
        print(f"  - Loss: MSE")
        print(f"  - Target: ~{phase_config['expected_error']}px")

        # Training loop
        epochs = phase_config['epochs']
        best_val_error = float('inf')
        best_epoch = 0
        patience = phase_config['early_stopping']['patience']

        print(f"\n🎯 Iniciando entrenamiento...")
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            # Training
            model.train()
            train_loss = 0.0
            train_pixel_error = 0.0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

            for images, landmarks, _ in progress_bar:
                images = images.to(self.device)
                landmarks = landmarks.to(self.device)

                optimizer.zero_grad()
                predictions = model(images)

                loss = criterion(predictions, landmarks)
                loss.backward()
                optimizer.step()

                pixel_error = self.compute_pixel_error(predictions, landmarks)

                train_loss += loss.item()
                train_pixel_error += pixel_error.item()

                progress_bar.set_postfix({
                    'Loss': f"{loss.item():.6f}",
                    'Pixel': f"{pixel_error.item():.2f}px"
                })

            # Validation
            model.eval()
            val_loss = 0.0
            val_pixel_error = 0.0

            with torch.no_grad():
                for images, landmarks, _ in val_loader:
                    images = images.to(self.device)
                    landmarks = landmarks.to(self.device)

                    predictions = model(images)
                    loss = criterion(predictions, landmarks)
                    pixel_error = self.compute_pixel_error(predictions, landmarks)

                    val_loss += loss.item()
                    val_pixel_error += pixel_error.item()

            # Métricas promedio
            train_loss /= len(train_loader)
            train_pixel_error /= len(train_loader)
            val_loss /= len(val_loader)
            val_pixel_error /= len(val_loader)

            # Learning rate step
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            # Log
            print(f"\nEpoch {epoch}/{epochs}:")
            print(f"  Train Loss: {train_loss:.6f} | Pixel Error: {train_pixel_error:.2f}px")
            print(f"  Val Loss: {val_loss:.6f} | Pixel Error: {val_pixel_error:.2f}px")
            print(f"  LR: {current_lr:.6f}")

            # Guardar mejor modelo
            if val_pixel_error < best_val_error:
                best_val_error = val_pixel_error
                best_epoch = epoch

                checkpoint_path = f"checkpoints/efficientnet/{phase_config['checkpoint_name']}"
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

                model.save_checkpoint(
                    checkpoint_path,
                    epoch=epoch,
                    optimizer_state=optimizer.state_dict(),
                    scheduler_state=scheduler.state_dict(),
                    metrics={
                        'best_val_error': best_val_error,
                        'train_loss': train_loss,
                        'val_loss': val_loss
                    }
                )
                print(f"  ✓ Best model saved: {best_val_error:.2f}px")

            # Early stopping
            if epoch - best_epoch > patience:
                print(f"\n⚠️ Early stopping: sin mejora por {patience} épocas")
                break

        elapsed_time = time.time() - start_time

        print("\n" + "=" * 60)
        print("✅ PHASE 1 COMPLETADO")
        print("=" * 60)
        print(f"Mejor error: {best_val_error:.2f}px (época {best_epoch})")
        print(f"Tiempo total: {elapsed_time/3600:.2f} horas")
        print(f"Checkpoint: {checkpoint_path}")
        print("=" * 60)

        return checkpoint_path

    # ========================================================================
    # PHASE 2: FINE-TUNING + WING LOSS
    # ========================================================================

    def train_phase2(self, checkpoint_phase1: Optional[str] = None) -> str:
        """
        Phase 2: Fine-tuning completo con Wing Loss

        Loss: Wing Loss (omega=10.0, epsilon=2.0)
        Target: ~8.20px

        Args:
            checkpoint_phase1: Path al checkpoint de Phase 1

        Returns:
            Path al checkpoint guardado
        """
        print("\n" + "=" * 60)
        print("📍 PHASE 2: FINE-TUNING + WING LOSS")
        print("=" * 60)

        phase_config = self.config['training_phase2']

        # Data loaders
        train_loader, val_loader, _ = self.create_data_loaders()

        # Cargar modelo desde Phase 1
        print("\n🏗️ Cargando modelo desde Phase 1...")

        if checkpoint_phase1 is None:
            checkpoint_phase1 = f"checkpoints/efficientnet/{self.config['training_phase1']['checkpoint_name']}"

        model, _ = EfficientNetLandmarkRegressor.load_from_checkpoint(
            checkpoint_phase1,
            device=self.device
        )

        # Descongelar backbone para fine-tuning
        model.unfreeze_backbone()

        model_info = model.get_model_info()
        print(f"✓ Total parámetros: {model_info['total_parameters']:,}")
        print(f"✓ Entrenables: {model_info['trainable_parameters']:,}")

        # Loss: Wing Loss
        loss_config = self.config['loss']
        criterion = WingLoss(
            omega=loss_config['wing_omega'],
            epsilon=loss_config['wing_epsilon']
        )

        # Optimizer con learning rates diferenciados
        backbone_params = list(model.get_backbone_parameters())
        attention_params = list(model.get_attention_parameters())
        head_params = list(model.get_head_parameters())

        optimizer = optim.Adam([
            {'params': backbone_params, 'lr': phase_config['backbone_lr'],
             'weight_decay': phase_config['weight_decay']},
            {'params': attention_params, 'lr': phase_config['head_lr'] * 0.5,  # Medium LR para attention
             'weight_decay': phase_config['weight_decay']},
            {'params': head_params, 'lr': phase_config['head_lr'],
             'weight_decay': phase_config['weight_decay']}
        ])

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=phase_config['epochs'],
            eta_min=phase_config['min_lr']
        )

        print(f"\n⚙️ Configuración:")
        print(f"  - Epochs: {phase_config['epochs']}")
        print(f"  - Backbone LR: {phase_config['backbone_lr']}")
        print(f"  - Head LR: {phase_config['head_lr']}")
        print(f"  - Loss: Wing Loss (ω={loss_config['wing_omega']}, ε={loss_config['wing_epsilon']})")
        print(f"  - Target: ~{phase_config['expected_error']}px")

        # Training loop
        epochs = phase_config['epochs']
        best_val_error = float('inf')
        best_epoch = 0
        patience = phase_config['early_stopping']['patience']

        print(f"\n🎯 Iniciando fine-tuning...")
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            # Training
            model.train()
            train_loss = 0.0
            train_pixel_error = 0.0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

            for images, landmarks, _ in progress_bar:
                images = images.to(self.device)
                landmarks = landmarks.to(self.device)

                optimizer.zero_grad()
                predictions = model(images)

                loss = criterion(predictions, landmarks)
                loss.backward()
                optimizer.step()

                pixel_error = self.compute_pixel_error(predictions, landmarks)

                train_loss += loss.item()
                train_pixel_error += pixel_error.item()

                progress_bar.set_postfix({
                    'Wing': f"{loss.item():.4f}",
                    'Pixel': f"{pixel_error.item():.2f}px"
                })

            # Validation
            model.eval()
            val_loss = 0.0
            val_pixel_error = 0.0

            with torch.no_grad():
                for images, landmarks, _ in val_loader:
                    images = images.to(self.device)
                    landmarks = landmarks.to(self.device)

                    predictions = model(images)
                    loss = criterion(predictions, landmarks)
                    pixel_error = self.compute_pixel_error(predictions, landmarks)

                    val_loss += loss.item()
                    val_pixel_error += pixel_error.item()

            # Métricas promedio
            train_loss /= len(train_loader)
            train_pixel_error /= len(train_loader)
            val_loss /= len(val_loader)
            val_pixel_error /= len(val_loader)

            # Learning rate step
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            # Log cada 5 épocas
            if epoch % 5 == 0 or epoch == 1:
                print(f"\nEpoch {epoch}/{epochs}:")
                print(f"  Train Wing Loss: {train_loss:.4f} | Pixel Error: {train_pixel_error:.2f}px")
                print(f"  Val Wing Loss: {val_loss:.4f} | Pixel Error: {val_pixel_error:.2f}px")
                print(f"  LR: {current_lr:.6f}")

            # Guardar mejor modelo
            if val_pixel_error < best_val_error:
                best_val_error = val_pixel_error
                best_epoch = epoch

                checkpoint_path = f"checkpoints/efficientnet/{phase_config['checkpoint_target']}"
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

                model.save_checkpoint(
                    checkpoint_path,
                    epoch=epoch,
                    optimizer_state=optimizer.state_dict(),
                    scheduler_state=scheduler.state_dict(),
                    metrics={
                        'best_pixel_error': best_val_error,
                        'train_loss': train_loss,
                        'val_loss': val_loss
                    }
                )

                if epoch % 5 == 0 or epoch == 1:
                    print(f"  ✓ Best model saved: {best_val_error:.2f}px")

            # Early stopping
            if epoch - best_epoch > patience:
                print(f"\n⚠️ Early stopping: sin mejora por {patience} épocas")
                break

        elapsed_time = time.time() - start_time

        print("\n" + "=" * 60)
        print("✅ PHASE 2 COMPLETADO")
        print("=" * 60)
        print(f"Mejor error: {best_val_error:.2f}px (época {best_epoch})")
        print(f"Mejora vs Phase 1: {phase_config.get('expected_error', 47.87) - best_val_error:.2f}px")
        print(f"Tiempo total: {elapsed_time/3600:.2f} horas")
        print(f"Checkpoint: {checkpoint_path}")
        print("=" * 60)

        return checkpoint_path

    # ========================================================================
    # PHASE 3: WING + SYMMETRY LOSS
    # ========================================================================

    def train_phase3(self, checkpoint_phase2: Optional[str] = None) -> str:
        """
        Phase 3: Wing Loss + Symmetry Loss

        Loss: Wing + 0.3 * Symmetry
        Target: ~7.65px (clinical excellence threshold: 8.5px)

        Args:
            checkpoint_phase2: Path al checkpoint de Phase 2

        Returns:
            Path al checkpoint guardado
        """
        print("\n" + "=" * 60)
        print("📍 PHASE 3: WING + SYMMETRY LOSS")
        print("=" * 60)

        phase_config = self.config['training_phase3']
        loss_config = self.config['loss']

        # Data loaders
        train_loader, val_loader, _ = self.create_data_loaders()

        # Cargar modelo desde Phase 2
        print("\n🏗️ Cargando modelo desde Phase 2...")

        if checkpoint_phase2 is None:
            checkpoint_phase2 = f"checkpoints/efficientnet/{self.config['training_phase2']['checkpoint_target']}"

        model, _ = EfficientNetLandmarkRegressor.load_from_checkpoint(
            checkpoint_phase2,
            device=self.device
        )

        print(f"✓ Modelo cargado con backbone descongelado")

        # Loss: Wing + Symmetry
        wing_loss = WingLoss(omega=loss_config['wing_omega'], epsilon=loss_config['wing_epsilon'])
        symmetry_loss = SymmetryLoss(symmetry_weight=1.0, use_mediastinal_axis=True)
        symmetry_weight = phase_config['symmetry_weight']

        def combined_loss_fn(predictions, targets):
            wing = wing_loss(predictions, targets)
            symmetry = symmetry_loss(predictions)
            total = wing + symmetry_weight * symmetry
            return total, wing.item(), symmetry.item()

        criterion = combined_loss_fn

        # Optimizer (mantener configuración exitosa)
        backbone_params = list(model.get_backbone_parameters())
        attention_params = list(model.get_attention_parameters())
        head_params = list(model.get_head_parameters())

        optimizer = optim.Adam([
            {'params': backbone_params, 'lr': phase_config['backbone_lr'],
             'weight_decay': phase_config['weight_decay']},
            {'params': attention_params, 'lr': phase_config['head_lr'] * 0.5,
             'weight_decay': phase_config['weight_decay']},
            {'params': head_params, 'lr': phase_config['head_lr'],
             'weight_decay': phase_config['weight_decay']}
        ])

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=phase_config['epochs'],
            eta_min=self.config['training_phase2']['min_lr']
        )

        print(f"\n⚙️ Configuración:")
        print(f"  - Epochs: {phase_config['epochs']}")
        print(f"  - Loss: Wing + {symmetry_weight}*Symmetry")
        print(f"  - Clinical target: 8.5px")
        print(f"  - Expected: ~{phase_config['expected_error']}px")

        # Training loop
        epochs = phase_config['epochs']
        best_val_error = float('inf')
        best_epoch = 0
        patience = phase_config['early_stopping']['patience']

        print(f"\n🎯 Iniciando entrenamiento con Symmetry Loss...")
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            # Training
            model.train()
            train_loss = 0.0
            train_wing = 0.0
            train_symmetry = 0.0
            train_pixel_error = 0.0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

            for images, landmarks, _ in progress_bar:
                images = images.to(self.device)
                landmarks = landmarks.to(self.device)

                optimizer.zero_grad()
                predictions = model(images)

                loss, wing_val, symmetry_val = criterion(predictions, landmarks)
                loss.backward()
                optimizer.step()

                pixel_error = self.compute_pixel_error(predictions, landmarks)

                train_loss += loss.item()
                train_wing += wing_val
                train_symmetry += symmetry_val
                train_pixel_error += pixel_error.item()

                progress_bar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Wing': f"{wing_val:.3f}",
                    'Sym': f"{symmetry_val:.3f}",
                    'Pixel': f"{pixel_error.item():.2f}px"
                })

            # Validation
            model.eval()
            val_loss = 0.0
            val_pixel_error = 0.0

            with torch.no_grad():
                for images, landmarks, _ in val_loader:
                    images = images.to(self.device)
                    landmarks = landmarks.to(self.device)

                    predictions = model(images)
                    loss, _, _ = criterion(predictions, landmarks)
                    pixel_error = self.compute_pixel_error(predictions, landmarks)

                    val_loss += loss.item()
                    val_pixel_error += pixel_error.item()

            # Métricas promedio
            train_loss /= len(train_loader)
            train_wing /= len(train_loader)
            train_symmetry /= len(train_loader)
            train_pixel_error /= len(train_loader)
            val_loss /= len(val_loader)
            val_pixel_error /= len(val_loader)

            # Learning rate step
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            # Log cada 5 épocas
            if epoch % 5 == 0 or epoch == 1:
                print(f"\nEpoch {epoch}/{epochs}:")
                print(f"  Train - Loss: {train_loss:.4f} | Wing: {train_wing:.3f} | Sym: {train_symmetry:.3f} | Pixel: {train_pixel_error:.2f}px")
                print(f"  Val - Loss: {val_loss:.4f} | Pixel: {val_pixel_error:.2f}px")
                print(f"  LR: {current_lr:.6f}")

                if val_pixel_error < 8.5:
                    print(f"  🎯 Clinical excellence achieved! (<8.5px)")

            # Guardar mejor modelo
            if val_pixel_error < best_val_error:
                best_val_error = val_pixel_error
                best_epoch = epoch

                checkpoint_path = f"checkpoints/efficientnet/{phase_config['checkpoint_target']}"
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

                model.save_checkpoint(
                    checkpoint_path,
                    epoch=epoch,
                    optimizer_state=optimizer.state_dict(),
                    scheduler_state=scheduler.state_dict(),
                    metrics={
                        'best_pixel_error': best_val_error,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'wing_loss': train_wing,
                        'symmetry_loss': train_symmetry
                    }
                )

                if epoch % 5 == 0 or epoch == 1:
                    print(f"  ✓ Best model saved: {best_val_error:.2f}px")

            # Early stopping
            if epoch - best_epoch > patience:
                print(f"\n⚠️ Early stopping: sin mejora por {patience} épocas")
                break

        elapsed_time = time.time() - start_time

        clinical_achieved = "✅ SÍ" if best_val_error < 8.5 else "❌ NO"

        print("\n" + "=" * 60)
        print("✅ PHASE 3 COMPLETADO")
        print("=" * 60)
        print(f"Mejor error: {best_val_error:.2f}px (época {best_epoch})")
        print(f"Excelencia clínica (<8.5px): {clinical_achieved}")
        print(f"Tiempo total: {elapsed_time/3600:.2f} horas")
        print(f"Checkpoint: {checkpoint_path}")
        print("=" * 60)

        return checkpoint_path

    # ========================================================================
    # PHASE 4: COMPLETE LOSS
    # ========================================================================

    def train_phase4(self, checkpoint_phase3: Optional[str] = None) -> str:
        """
        Phase 4: Complete Loss (Wing + Symmetry + Distance)

        Loss: Wing + 0.3*Symmetry + 0.2*Distance
        Target: ~7.12px (val) / 7.23px (test)

        Args:
            checkpoint_phase3: Path al checkpoint de Phase 3

        Returns:
            Path al checkpoint guardado
        """
        print("\n" + "=" * 60)
        print("📍 PHASE 4: COMPLETE LOSS (Wing + Symmetry + Distance)")
        print("=" * 60)

        phase_config = self.config['training_phase4']
        loss_config = self.config['loss']

        # Data loaders
        train_loader, val_loader, _ = self.create_data_loaders()

        # Cargar modelo desde Phase 3
        print("\n🏗️ Cargando modelo desde Phase 3...")

        if checkpoint_phase3 is None:
            checkpoint_phase3 = f"checkpoints/efficientnet/{self.config['training_phase3']['checkpoint_target']}"

        model, _ = EfficientNetLandmarkRegressor.load_from_checkpoint(
            checkpoint_phase3,
            device=self.device
        )

        print(f"✓ Modelo cargado desde Phase 3")

        # Loss: Complete (Wing + Symmetry + Distance)
        wing_loss = WingLoss(omega=loss_config['wing_omega'], epsilon=loss_config['wing_epsilon'])
        symmetry_loss = SymmetryLoss(symmetry_weight=1.0, use_mediastinal_axis=True)
        distance_loss = DistancePreservationLoss(distance_weight=1.0)

        symmetry_weight = phase_config['symmetry_weight']
        distance_weight = phase_config['distance_weight']

        def complete_loss_fn(predictions, targets):
            wing = wing_loss(predictions, targets)
            symmetry = symmetry_loss(predictions)
            distance = distance_loss(predictions, targets)
            total = wing + symmetry_weight * symmetry + distance_weight * distance
            return total, wing.item(), symmetry.item(), distance.item()

        criterion = complete_loss_fn

        # Optimizer
        backbone_params = list(model.get_backbone_parameters())
        attention_params = list(model.get_attention_parameters())
        head_params = list(model.get_head_parameters())

        optimizer = optim.Adam([
            {'params': backbone_params, 'lr': phase_config['backbone_lr'],
             'weight_decay': phase_config['weight_decay']},
            {'params': attention_params, 'lr': phase_config['head_lr'] * 0.5,
             'weight_decay': phase_config['weight_decay']},
            {'params': head_params, 'lr': phase_config['head_lr'],
             'weight_decay': phase_config['weight_decay']}
        ])

        # Scheduler: Cosine Annealing Warm Restarts
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=phase_config['scheduler_params']['T_0'],
            T_mult=phase_config['scheduler_params']['T_mult'],
            eta_min=phase_config['scheduler_params']['eta_min']
        )

        print(f"\n⚙️ Configuración:")
        print(f"  - Epochs: {phase_config['epochs']}")
        print(f"  - Loss: Wing + {symmetry_weight}*Sym + {distance_weight}*Dist")
        print(f"  - Scheduler: Cosine Annealing Warm Restarts")
        print(f"  - Expected: ~{phase_config['expected_error']}px")
        print(f"  - Target: <{phase_config['target_error']}px")

        # Training loop
        epochs = phase_config['epochs']
        best_val_error = float('inf')
        best_epoch = 0
        patience = phase_config['early_stopping']['patience']

        print(f"\n🎯 Iniciando entrenamiento con Complete Loss...")
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            # Training
            model.train()
            train_loss = 0.0
            train_wing = 0.0
            train_symmetry = 0.0
            train_distance = 0.0
            train_pixel_error = 0.0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

            for images, landmarks, _ in progress_bar:
                images = images.to(self.device)
                landmarks = landmarks.to(self.device)

                optimizer.zero_grad()
                predictions = model(images)

                loss, wing_val, symmetry_val, distance_val = criterion(predictions, landmarks)
                loss.backward()
                optimizer.step()

                pixel_error = self.compute_pixel_error(predictions, landmarks)

                train_loss += loss.item()
                train_wing += wing_val
                train_symmetry += symmetry_val
                train_distance += distance_val
                train_pixel_error += pixel_error.item()

                progress_bar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'W': f"{wing_val:.3f}",
                    'S': f"{symmetry_val:.3f}",
                    'D': f"{distance_val:.3f}",
                    'Px': f"{pixel_error.item():.2f}"
                })

            # Validation
            model.eval()
            val_loss = 0.0
            val_pixel_error = 0.0

            with torch.no_grad():
                for images, landmarks, _ in val_loader:
                    images = images.to(self.device)
                    landmarks = landmarks.to(self.device)

                    predictions = model(images)
                    loss, _, _, _ = criterion(predictions, landmarks)
                    pixel_error = self.compute_pixel_error(predictions, landmarks)

                    val_loss += loss.item()
                    val_pixel_error += pixel_error.item()

            # Métricas promedio
            train_loss /= len(train_loader)
            train_wing /= len(train_loader)
            train_symmetry /= len(train_loader)
            train_distance /= len(train_loader)
            train_pixel_error /= len(train_loader)
            val_loss /= len(val_loader)
            val_pixel_error /= len(val_loader)

            # Learning rate step
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            # Log cada 5 épocas
            if epoch % 5 == 0 or epoch == 1:
                print(f"\nEpoch {epoch}/{epochs}:")
                print(f"  Train - Total: {train_loss:.4f} | Wing: {train_wing:.3f} | Sym: {train_symmetry:.3f} | Dist: {train_distance:.3f}")
                print(f"  Train Pixel: {train_pixel_error:.2f}px")
                print(f"  Val - Loss: {val_loss:.4f} | Pixel: {val_pixel_error:.2f}px")
                print(f"  LR: {current_lr:.6f}")

                if val_pixel_error < phase_config['target_error']:
                    print(f"  🎯 TARGET ACHIEVED! (<{phase_config['target_error']}px)")

            # Guardar mejor modelo
            if val_pixel_error < best_val_error:
                best_val_error = val_pixel_error
                best_epoch = epoch

                checkpoint_path = f"checkpoints/efficientnet/{phase_config['checkpoint_target']}"
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

                model.save_checkpoint(
                    checkpoint_path,
                    epoch=epoch,
                    optimizer_state=optimizer.state_dict(),
                    scheduler_state=scheduler.state_dict(),
                    metrics={
                        'best_pixel_error': best_val_error,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'wing_loss': train_wing,
                        'symmetry_loss': train_symmetry,
                        'distance_loss': train_distance
                    }
                )

                if epoch % 5 == 0 or epoch == 1:
                    print(f"  ✓ Best model saved: {best_val_error:.2f}px")

            # Early stopping
            if epoch - best_epoch > patience:
                print(f"\n⚠️ Early stopping: sin mejora por {patience} épocas")
                break

        elapsed_time = time.time() - start_time

        target_achieved = "✅ SÍ" if best_val_error < phase_config['target_error'] else "❌ NO"

        print("\n" + "=" * 60)
        print("✅ PHASE 4 COMPLETADO")
        print("=" * 60)
        print(f"Mejor error: {best_val_error:.2f}px (época {best_epoch})")
        print(f"Target <{phase_config['target_error']}px: {target_achieved}")
        print(f"Tiempo total: {elapsed_time/3600:.2f} horas")
        print(f"Checkpoint: {checkpoint_path}")
        print("=" * 60)

        return checkpoint_path

    # ========================================================================
    # PIPELINE COMPLETO
    # ========================================================================

    def train_all_phases(self):
        """Ejecutar pipeline completo 4-phase"""
        print("\n" + "=" * 60)
        print("🚀 INICIANDO PIPELINE COMPLETO 4-PHASE")
        print("=" * 60)

        total_start = time.time()

        # Phase 1
        checkpoint_phase1 = self.train_phase1()

        # Phase 2
        checkpoint_phase2 = self.train_phase2(checkpoint_phase1)

        # Phase 3
        checkpoint_phase3 = self.train_phase3(checkpoint_phase2)

        # Phase 4
        checkpoint_phase4 = self.train_phase4(checkpoint_phase3)

        total_elapsed = time.time() - total_start

        print("\n" + "=" * 60)
        print("🎉 PIPELINE COMPLETO FINALIZADO")
        print("=" * 60)
        print(f"Tiempo total: {total_elapsed/3600:.2f} horas")
        print(f"\nCheckpoints generados:")
        print(f"  Phase 1: {checkpoint_phase1}")
        print(f"  Phase 2: {checkpoint_phase2}")
        print(f"  Phase 3: {checkpoint_phase3}")
        print(f"  Phase 4: {checkpoint_phase4}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="EfficientNet-B1 4-Phase Training Pipeline")
    parser.add_argument('--phase', type=int, choices=[1, 2, 3, 4], default=None,
                       help="Entrenar solo una fase específica (1-4)")
    parser.add_argument('--all', action='store_true',
                       help="Entrenar todas las fases secuencialmente")
    parser.add_argument('--config', type=str, default="configs/efficientnet_config.yaml",
                       help="Path al archivo de configuración")
    parser.add_argument('--checkpoint', type=str, default=None,
                       help="Checkpoint previo (para phases 2-4)")

    args = parser.parse_args()

    # Crear trainer
    trainer = EfficientNetPhaseTrainer(config_path=args.config)

    # Ejecutar fase(s) solicitada(s)
    if args.all:
        trainer.train_all_phases()
    elif args.phase == 1:
        trainer.train_phase1()
    elif args.phase == 2:
        trainer.train_phase2(args.checkpoint)
    elif args.phase == 3:
        trainer.train_phase3(args.checkpoint)
    elif args.phase == 4:
        trainer.train_phase4(args.checkpoint)
    else:
        print("❌ Especifica --phase [1-4] o --all")
        parser.print_help()


if __name__ == "__main__":
    main()
