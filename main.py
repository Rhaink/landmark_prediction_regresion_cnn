#!/usr/bin/env python3
"""
Script principal para el proyecto de regresión de landmarks con ResNet-18
Permite ejecutar todas las fases del proyecto de manera organizada
"""

import os
import sys
import argparse
from pathlib import Path


def print_banner():
    """Mostrar banner del proyecto"""
    print("="*70)
    print("  LANDMARK REGRESSION WITH RESNET-18 TRANSFER LEARNING")
    print("  Predicción de landmarks en imágenes médicas")
    print("="*70)


def check_environment():
    """Verificar que el entorno esté configurado correctamente"""
    print("\n🔍 Verificando entorno...")

    # Verificar directorios necesarios
    required_dirs = ['data', 'configs', 'src']
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            print(f"❌ Directorio faltante: {dir_name}")
            return False

    # Verificar archivo de configuración
    if not os.path.exists('configs/config.yaml'):
        print("❌ Archivo de configuración faltante: configs/config.yaml")
        return False

    # Verificar datos
    if not os.path.exists('data/coordenadas/coordenadas_maestro.csv'):
        print("❌ Archivo de anotaciones faltante: data/coordenadas/coordenadas_maestro.csv")
        return False

    if not os.path.exists('data/dataset'):
        print("❌ Directorio de imágenes faltante: data/dataset")
        return False

    print("✓ Entorno configurado correctamente")
    return True


def explore_data():
    """Ejecutar exploración de datos"""
    print("\n📊 Ejecutando exploración de datos...")
    os.system("python explore_data.py")


def test_setup():
    """Probar configuración del sistema"""
    print("\n🧪 Probando configuración del sistema...")

    # Test GPU
    print("- Probando GPU...")
    result = os.system("python test_gpu.py")
    if result != 0:
        print("⚠ Problema con configuración de GPU")

    # Test Dataset
    print("- Probando dataset...")
    result = os.system("python test_dataset.py")
    if result != 0:
        print("❌ Problema con dataset")
        return False

    print("✓ Configuración del sistema OK")
    return True


def train_phase1():
    """Ejecutar entrenamiento Fase 1"""
    print("\n🚀 Ejecutando entrenamiento Fase 1 (Solo cabeza)...")
    os.system("python src/training/train_phase1.py")


def train_phase2():
    """Ejecutar entrenamiento Fase 2"""
    print("\n🚀 Ejecutando entrenamiento Fase 2 (Fine-tuning completo)...")

    # Verificar que existe checkpoint de Fase 1
    if not os.path.exists("checkpoints/phase1_best.pt"):
        print("❌ Checkpoint de Fase 1 no encontrado. Ejecuta primero Fase 1.")
        return False

    os.system("python src/training/train_phase2.py")
    return True


def evaluate_model(checkpoint_path: str):
    """Ejecutar evaluación del modelo"""
    print(f"\n📈 Evaluando modelo: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint no encontrado: {checkpoint_path}")
        return False

    os.system(f"python src/evaluation/evaluate.py --checkpoint {checkpoint_path}")
    return True


def visualize_predictions(checkpoint_path: str, image_path: str = None):
    """Visualizar predicciones"""
    print(f"\n🖼️ Visualizando predicciones...")

    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint no encontrado: {checkpoint_path}")
        return False

    if image_path:
        os.system(f"python src/evaluation/visualize.py --checkpoint {checkpoint_path} --image {image_path}")
    else:
        # Usar algunas imágenes del dataset
        os.system(f"python src/evaluation/visualize.py --checkpoint {checkpoint_path} --image_dir data/dataset --max_images 5")

    return True


def train_ensemble():
    """Ejecutar entrenamiento de ensemble"""
    print("\n🚀 Ejecutando entrenamiento de ensemble...")

    # Verificar configuración de ensemble
    import yaml
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    if 'ensemble' not in config:
        print("❌ Configuración de ensemble no encontrada en config.yaml")
        return False

    # El ensemble ahora entrena automáticamente Fase 1 + Fase 2 para cada seed
    print("✓ Ensemble entrenará automáticamente todos los modelos necesarios")

    os.system("python src/training/train_ensemble.py")
    return True


def evaluate_ensemble():
    """Ejecutar evaluación de ensemble"""
    print("\n📈 Evaluando ensemble...")

    # Verificar directorio de ensemble
    ensemble_dir = "checkpoints/ensemble"
    if not os.path.exists(ensemble_dir):
        print(f"❌ Directorio de ensemble no encontrado: {ensemble_dir}")
        print("💡 Ejecuta primero: python main.py train_ensemble")
        return False

    # Verificar metadata del ensemble
    metadata_path = f"{ensemble_dir}/ensemble_metadata.json"
    if not os.path.exists(metadata_path):
        print(f"❌ Metadata de ensemble no encontrada: {metadata_path}")
        print("💡 El ensemble no está completo o corrupto")
        return False

    os.system(f"python src/evaluation/evaluate_ensemble.py --ensemble_dir {ensemble_dir}")
    return True


def visualize_test_complete(checkpoint_path: str):
    """Ejecutar visualización completa del conjunto de test"""
    print(f"\n🖼️ Generando visualizaciones completas del test...")

    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint no encontrado: {checkpoint_path}")
        return False

    # Usar script directo sin problemas de imports
    try:
        # Determinar si es modelo de symmetry o estándar
        if "symmetry" in checkpoint_path:
            print("🔍 Detectado modelo Symmetry - usando visualización especializada")
            # Crear script temporal para symmetry
            script_content = '''
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from src.data.dataset import create_dataloaders
from src.models.resnet_regressor import ResNetLandmarkRegressor
from src.training.utils import setup_device

def create_prediction_image(image, gt_landmarks, pred_landmarks, error, filename, save_path):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    if isinstance(image, torch.Tensor):
        img = image.cpu().numpy().transpose(1, 2, 0)
        img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        img = np.clip(img, 0, 1)
    else:
        img = image

    ax.imshow(img)

    if gt_landmarks.max() <= 1.0:
        gt_points = gt_landmarks.reshape(15, 2) * 224
        pred_points = pred_landmarks.reshape(15, 2) * 224
    else:
        gt_points = gt_landmarks.reshape(15, 2)
        pred_points = pred_landmarks.reshape(15, 2)

    ax.scatter(gt_points[:, 0], gt_points[:, 1], c='lime', s=80, marker='o', alpha=0.9, label='Ground Truth', edgecolors='darkgreen', linewidth=2)
    ax.scatter(pred_points[:, 0], pred_points[:, 1], c='red', s=60, marker='x', alpha=0.9, label='Prediction', linewidth=3)

    for i in range(15):
        ax.plot([gt_points[i, 0], pred_points[i, 0]], [gt_points[i, 1], pred_points[i, 1]], 'yellow', alpha=0.6, linewidth=2)

    color = '🟢' if error <= 9.3 else '🟡' if error <= 12 else '🔴'
    ax.set_title(f'{color} {filename}\\nError: {error:.2f}px | Symmetry Model | Target: ≤9.3px', fontsize=12, pad=20)

    ax.set_xlim(0, 224)
    ax.set_ylim(224, 0)
    ax.legend(loc='upper right', fontsize=10)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

device = setup_device(use_gpu=True, gpu_id=0)
_, _, test_loader = create_dataloaders(
    annotations_file="data/coordenadas/coordenadas_maestro.csv",
    images_dir="data/dataset",
    batch_size=1,
    num_workers=2,
    pin_memory=True,
    train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42
)

model = ResNetLandmarkRegressor(num_landmarks=15, pretrained=False, freeze_backbone=False)
checkpoint = torch.load("''' + checkpoint_path + '''", map_location=str(device))
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

output_dir = Path("evaluation_results/test_predictions_symmetry")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"📁 Generando {len(test_loader)} visualizaciones en: {output_dir}")

all_errors = []
with torch.no_grad():
    for idx, (images, landmarks, _) in enumerate(tqdm(test_loader, desc="Generando")):
        images, landmarks = images.to(device), landmarks.to(device)
        predictions = model(images)

        pred_reshaped = predictions.view(-1, 15, 2)
        target_reshaped = landmarks.view(-1, 15, 2)
        distances = torch.norm(pred_reshaped - target_reshaped, dim=2)
        error = torch.mean(distances * 224).item()
        all_errors.append(error)

        filename = f"symmetry_test_{idx+1:03d}_error_{error:.2f}px"
        save_path = output_dir / f"{filename}.png"

        create_prediction_image(
            images[0], landmarks[0].cpu().numpy(), predictions[0].cpu().numpy(),
            error, filename, save_path
        )

print(f"✅ Generadas {len(all_errors)} visualizaciones")
print(f"📊 Error promedio: {np.mean(all_errors):.2f}px")
print(f"📁 Ubicación: {output_dir}")
'''
            with open("temp_visualize_symmetry.py", "w") as f:
                f.write(script_content)

            result = os.system("python temp_visualize_symmetry.py")
            os.remove("temp_visualize_symmetry.py")

        else:
            print("🔍 Detectado modelo estándar - usando evaluación estándar")
            # Para modelos estándar, usar evaluate script
            result = os.system(f"python -c \"import sys; sys.path.append('.'); "
                             f"exec(open('evaluate_symmetry.py').read().replace('geometric_symmetry.pt', '{checkpoint_path}'))\"")

        if result == 0:
            print("✅ Visualizaciones completas generadas!")
            if "symmetry" in checkpoint_path:
                print("📁 Revisa: evaluation_results/test_predictions_symmetry/")
            else:
                print("📁 Revisa: evaluation_results/test_predictions/")
            return True
        else:
            print("❌ Error durante la generación")
            return False

    except Exception as e:
        print(f"❌ Error durante la visualización: {e}")
        return False


def visualize_test_complete_loss():
    """Ejecutar visualización completa del conjunto de test para modelo Complete Loss"""
    print("\n🖼️ Generando visualizaciones completas del test - Complete Loss Model...")

    # Verificar que existe el modelo Complete Loss
    checkpoint_path = "checkpoints/geometric_complete.pt"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint Complete Loss no encontrado: {checkpoint_path}")
        print("💡 Ejecuta primero: python main.py train_geometric_complete")
        return False

    print(f"✓ Checkpoint encontrado: {checkpoint_path}")

    # Ejecutar script de visualización
    try:
        result = os.system("python visualize_complete_test.py")

        if result == 0:
            print("✅ Visualizaciones Complete Loss generadas exitosamente!")
            print("📁 Revisa: evaluation_results/test_predictions_complete_loss/")
            return True
        else:
            print("❌ Error durante la generación de visualizaciones")
            return False

    except Exception as e:
        print(f"❌ Error ejecutando visualización Complete Loss: {e}")
        return False


def run_full_pipeline():
    """Ejecutar pipeline completo"""
    print("\n🔄 Ejecutando pipeline completo...")

    steps = [
        ("Verificar entorno", check_environment),
        ("Probar configuración", test_setup),
        ("Explorar datos", explore_data),
        ("Entrenar Fase 1", train_phase1),
        ("Entrenar Fase 2", train_phase2),
        ("Evaluar modelo final", lambda: evaluate_model("checkpoints/phase2_best.pt")),
        ("Visualizar predicciones", lambda: visualize_predictions("checkpoints/phase2_best.pt"))
    ]

    for i, (step_name, step_func) in enumerate(steps, 1):
        print(f"\n--- Paso {i}: {step_name} ---")

        try:
            result = step_func()
            if result is False:
                print(f"❌ Error en paso {i}: {step_name}")
                print("Pipeline interrumpido")
                return False

        except KeyboardInterrupt:
            print(f"\n⚠ Pipeline interrumpido por el usuario en paso {i}")
            return False

        except Exception as e:
            print(f"❌ Error inesperado en paso {i}: {e}")
            return False

    print("\n🎉 Pipeline completo ejecutado exitosamente!")
    return True


# ============================================================================
# FUNCIONES GEOMÉTRICAS (NUEVA IMPLEMENTACIÓN)
# ============================================================================

def train_geometric_phase1():
    """Entrenar Fase 1 con Wing Loss y análisis geométrico"""
    print("\n🚀 Iniciando entrenamiento geométrico Fase 1: Wing Loss")

    try:
        import subprocess
        result = subprocess.run([
            sys.executable, 'src/training/train_geometric_phase1.py',
            '--config', 'configs/config_geometric.yaml'
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Entrenamiento Fase 1 geométrico completado exitosamente!")
            return True
        else:
            print(f"❌ Error en entrenamiento: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Error ejecutando entrenamiento geométrico: {e}")
        return False


def train_geometric_phase2():
    """Entrenar Fase 2 con fine-tuning completo + Wing Loss"""
    print("\n🚀 Iniciando entrenamiento geométrico Fase 2: Fine-tuning + Wing Loss")

    try:
        import subprocess
        result = subprocess.run([
            sys.executable, 'src/training/train_geometric_phase2.py',
            '--config', 'configs/config_geometric.yaml'
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Entrenamiento Fase 2 geométrico completado exitosamente!")
            return True
        else:
            print(f"❌ Error en entrenamiento: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Error ejecutando entrenamiento geométrico: {e}")
        return False


def train_geometric_attention():
    """Entrenar Fase 2-Attention con Coordinate Attention"""
    print("\n🚀 Iniciando entrenamiento geométrico Fase 2-Attention: Coordinate Attention")

    try:
        import subprocess
        result = subprocess.run([
            sys.executable, 'src/training/train_geometric_attention.py',
            '--config', 'configs/config_geometric.yaml'
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Entrenamiento Fase 2-Attention geométrico completado exitosamente!")
            return True
        else:
            print(f"❌ Error en entrenamiento: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Error ejecutando entrenamiento geométrico: {e}")
        return False


def train_geometric_phase3():
    """Entrenar Fase 3 con Symmetry Loss"""
    print("\n🚀 Iniciando entrenamiento geométrico Fase 3: Wing Loss + Symmetry Loss")

    try:
        import subprocess
        result = subprocess.run([
            sys.executable, 'src/training/train_geometric_symmetry.py',
            '--config', 'configs/config_geometric.yaml'
        ], check=True, capture_output=True, text=True)

        print("✅ Entrenamiento Fase 3 completado exitosamente")
        print(result.stdout)
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Error en entrenamiento Fase 3:")
        print(e.stderr)
        return False
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False


def train_geometric_final():
    """Entrenar Fase 4 con loss completo (Wing + Symmetry + Distance)"""
    print("\n🚀 Iniciando entrenamiento geométrico final: Complete Loss")
    print("📊 Base: Phase 3 Symmetry model (8.91px)")
    print("🎯 Target: <8.5px para excelencia clínica")

    # Verificar que existe el modelo base Phase 3
    base_checkpoint = "checkpoints/geometric_symmetry.pt"
    if not os.path.exists(base_checkpoint):
        print(f"❌ Modelo base Phase 3 no encontrado: {base_checkpoint}")
        print("💡 Ejecuta primero: python main.py train_geometric_symmetry")
        return False

    print(f"✓ Modelo base encontrado: {base_checkpoint}")

    # Ejecutar entrenamiento Phase 4
    result = os.system("python train_complete_simple.py")

    if result == 0:
        print("✅ Phase 4 completado exitosamente")
        if os.path.exists("checkpoints/geometric_complete.pt"):
            print("💾 Modelo guardado: checkpoints/geometric_complete.pt")
        return True
    else:
        print("❌ Error en entrenamiento Phase 4")
        return False


def evaluate_geometric(checkpoint_path):
    """Evaluar modelo con métricas geométricas"""
    print(f"\n🔍 Evaluando modelo geométrico: {checkpoint_path}")

    try:
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent))

        from src.evaluation.evaluate import LandmarkEvaluator
        from src.training.utils import validate_geometric_predictions, GeometricLandmarkMetrics
        from src.data.dataset import create_dataloaders
        from src.models.resnet_regressor import ResNetLandmarkRegressor
        import torch

        # Cargar configuración
        config_path = 'configs/config_geometric.yaml'
        if not os.path.exists(config_path):
            print(f"❌ Configuración no encontrada: {config_path}")
            return False

        # Usar evaluador existente pero con métricas geométricas
        evaluator = LandmarkEvaluator(checkpoint_path, config_path)

        print("📊 Ejecutando evaluación con métricas geométricas...")
        results = evaluator.evaluate()

        if results:
            print("✅ Evaluación geométrica completada!")
            print(f"📏 Error promedio: {results.get('pixel_error_mean', 'N/A')} píxeles")

            # Mostrar métricas geométricas si están disponibles
            if 'symmetry_consistency' in results:
                print(f"🔄 Consistencia de simetría: {results['symmetry_consistency']:.3f}")
            if 'anatomical_validity' in results:
                print(f"🏥 Validez anatómica: {results['anatomical_validity']:.3f}")

            return True
        else:
            return False

    except Exception as e:
        print(f"❌ Error durante evaluación geométrica: {e}")
        return False


def analyze_geometric_improvements():
    """Analizar mejoras geométricas comparando con baseline"""
    print("\n📊 Analizando mejoras geométricas...")

    try:
        baseline_error = 11.34  # Error del modelo actual

        # Buscar resultados de fases geométricas
        results_dir = Path("logs")
        geometric_experiments = []

        for exp_dir in results_dir.glob("geometric_*"):
            if exp_dir.is_dir():
                results_file = exp_dir / "phase1_results.yaml"
                if results_file.exists():
                    import yaml
                    with open(results_file, 'r') as f:
                        results = yaml.safe_load(f)

                    geometric_experiments.append({
                        'experiment': exp_dir.name,
                        'error': results.get('best_pixel_error', baseline_error),
                        'improvement': results.get('improvement', 0),
                        'target_achieved': results.get('target_achieved', False)
                    })

        if geometric_experiments:
            print("\n📈 RESULTADOS DE EXPERIMENTOS GEOMÉTRICOS:")
            print(f"{'Experimento':<30} {'Error (px)':<12} {'Mejora (px)':<12} {'Objetivo':<10}")
            print("-" * 70)
            print(f"{'Baseline Original':<30} {baseline_error:<12.2f} {0:<12.2f} {'N/A':<10}")

            for exp in geometric_experiments:
                status = "✅" if exp['target_achieved'] else "⚠️"
                print(f"{exp['experiment']:<30} {exp['error']:<12.2f} "
                      f"{exp['improvement']:<12.2f} {status:<10}")

            # Mejor resultado
            best_exp = min(geometric_experiments, key=lambda x: x['error'])
            total_improvement = baseline_error - best_exp['error']

            print(f"\n🏆 MEJOR RESULTADO:")
            print(f"  Experimento: {best_exp['experiment']}")
            print(f"  Error final: {best_exp['error']:.2f} píxeles")
            print(f"  Mejora total: {total_improvement:.2f} píxeles")
            print(f"  Porcentaje de mejora: {(total_improvement/baseline_error)*100:.1f}%")

            if best_exp['error'] < 10.0:
                print("🎉 ¡OBJETIVO <10px ALCANZADO!")
            else:
                remaining = best_exp['error'] - 10.0
                print(f"🎯 Faltan {remaining:.2f}px para alcanzar objetivo <10px")
        else:
            print("⚠️ No se encontraron experimentos geométricos. Ejecuta train_geometric_phase1 primero.")

        return True

    except Exception as e:
        print(f"❌ Error analizando mejoras: {e}")
        return False


def validate_geometric_comprehensive(checkpoint_path):
    """Validación geométrica exhaustiva"""
    print(f"\n🔍 Validación geométrica exhaustiva: {checkpoint_path}")

    try:
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent))

        from src.training.utils import validate_geometric_predictions, load_config
        from src.data.dataset import create_dataloaders
        from src.models.resnet_regressor import ResNetLandmarkRegressor
        import torch

        # Verificar checkpoint
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint no encontrado: {checkpoint_path}")
            return False

        # Cargar configuración
        config = load_config('configs/config_geometric.yaml')

        # Crear data loaders
        _, val_loader, test_loader = create_dataloaders(
            annotations_file=config['data']['coordenadas_path'],
            images_dir=config['data']['dataset_path'],
            batch_size=config['data']['batch_size'],
            num_workers=config['data']['num_workers'],
            pin_memory=config['data']['pin_memory'],
            train_ratio=config['data']['train_split'],
            val_ratio=config['data']['val_split'],
            test_ratio=config['data']['test_split'],
            random_seed=config['data']['random_seed']
        )

        # Cargar modelo
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, _ = ResNetLandmarkRegressor.load_from_checkpoint(checkpoint_path, map_location=str(device))
        model = model.to(device)

        # Validación en conjunto de validación
        print("📊 Validando en conjunto de validación...")
        val_results = validate_geometric_predictions(
            model=model,
            dataloader=val_loader,
            device=device,
            validity_threshold=0.7
        )

        # Validación en conjunto de test
        print("📊 Validando en conjunto de test...")
        test_results = validate_geometric_predictions(
            model=model,
            dataloader=test_loader,
            device=device,
            validity_threshold=0.7
        )

        # Mostrar resultados
        print("\n📊 RESULTADOS DE VALIDACIÓN GEOMÉTRICA:")
        print("\nConjunto de Validación:")
        print(f"  Error promedio: {val_results['pixel_error_mean']:.2f} píxeles")
        print(f"  Tasa de validez anatómica: {val_results['geometric_validity_rate']:.1%}")
        print(f"  Consistencia de simetría: {val_results['symmetry_consistency']:.3f}")
        print(f"  Validez anatómica: {val_results['anatomical_validity']:.3f}")

        print("\nConjunto de Test:")
        print(f"  Error promedio: {test_results['pixel_error_mean']:.2f} píxeles")
        print(f"  Tasa de validez anatómica: {test_results['geometric_validity_rate']:.1%}")
        print(f"  Consistencia de simetría: {test_results['symmetry_consistency']:.3f}")
        print(f"  Validez anatómica: {test_results['anatomical_validity']:.3f}")

        # Guardar resultados
        results_dir = Path("evaluation_results/geometric_validation")
        results_dir.mkdir(parents=True, exist_ok=True)

        import yaml
        with open(results_dir / "validation_results.yaml", 'w') as f:
            yaml.dump({
                'validation': val_results,
                'test': test_results
            }, f, default_flow_style=False)

        print(f"✅ Resultados guardados en: {results_dir}")
        return True

    except Exception as e:
        print(f"❌ Error en validación geométrica: {e}")
        return False


# ============================================================================
# EFFICIENTNET COMMANDS
# ============================================================================

def train_efficientnet(phase: int = None):
    """
    Entrenar EfficientNet-B1 con pipeline 4-phase

    Args:
        phase: Fase específica (1-4) o None para todas

    Returns:
        True si exitoso
    """
    print("\n🚀 Entrenando EfficientNet-B1...")

    if phase is None:
        # Entrenar todas las fases
        print("📍 Ejecutando pipeline completo 4-phase")
        result = os.system("python train_efficientnet_phases.py --all")
    else:
        # Entrenar fase específica
        print(f"📍 Ejecutando Phase {phase}")
        result = os.system(f"python train_efficientnet_phases.py --phase {phase}")

    return result == 0


def evaluate_efficientnet():
    """
    Comparar EfficientNet-B1 vs ResNet-18

    Returns:
        True si exitoso
    """
    print("\n📊 Comparando EfficientNet-B1 vs ResNet-18...")

    # Verificar que existan checkpoints
    efficientnet_checkpoint = "checkpoints/efficientnet/efficientnet_phase4_best.pt"
    resnet_checkpoint = "checkpoints/geometric_complete.pt"

    if not os.path.exists(efficientnet_checkpoint):
        print(f"❌ Checkpoint de EfficientNet no encontrado: {efficientnet_checkpoint}")
        print("   Ejecuta: python main.py train_efficientnet")
        return False

    if not os.path.exists(resnet_checkpoint):
        print(f"⚠️  Checkpoint de ResNet no encontrado: {resnet_checkpoint}")
        print("   Usando checkpoint por defecto si existe...")

    # Ejecutar comparación
    result = os.system(f"python compare_efficientnet_vs_resnet.py --efficientnet {efficientnet_checkpoint} --resnet {resnet_checkpoint}")

    return result == 0


def visualize_efficientnet(checkpoint_path: str = None):
    """
    Visualizar predicciones de EfficientNet

    Args:
        checkpoint_path: Path al checkpoint (opcional)

    Returns:
        True si exitoso
    """
    if checkpoint_path is None:
        checkpoint_path = "checkpoints/efficientnet/efficientnet_phase4_best.pt"

    print(f"\n🖼️ Visualizando predicciones de EfficientNet...")
    print(f"   Checkpoint: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint no encontrado: {checkpoint_path}")
        return False

    # Usar script de visualización de test complete (compatible con EfficientNet)
    result = os.system(f"python visualize_complete_test.py --checkpoint {checkpoint_path}")

    return result == 0


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Proyecto de regresión de landmarks')

    parser.add_argument('command', choices=[
        'check',           # Verificar entorno
        'explore',         # Explorar datos
        'test',            # Probar configuración
        'train1',          # Entrenar Fase 1
        'train2',          # Entrenar Fase 2
        'evaluate',        # Evaluar modelo
        'visualize',       # Visualizar predicciones
        'visualize_test',  # Visualizar todo el conjunto de test
        'visualize_test_complete_loss',  # Visualizar test set con modelo Complete Loss
        'train_ensemble',  # Entrenar ensemble
        'evaluate_ensemble', # Evaluar ensemble
        'pipeline',        # Ejecutar pipeline completo
        # Comandos geométricos ResNet
        'train_geometric_phase1',  # Entrenar con Wing Loss (Fase 1)
        'train_geometric_phase2',  # Entrenar con fine-tuning + Wing Loss (Fase 2)
        'train_geometric_attention',  # Entrenar con Coordinate Attention (Fase 2b)
        'train_geometric_phase3',  # Entrenar con Symmetry Loss (Fase 3)
        'train_geometric_symmetry',  # Alias para train_geometric_phase3
        'train_geometric_final',   # Entrenar con loss completo (Fase 4)
        'train_geometric_complete', # Alias para train_geometric_final
        'evaluate_geometric',      # Evaluar con métricas geométricas
        'analyze_geometric',       # Análizar mejoras geométricas
        'validate_geometric',      # Validación geométrica exhaustiva
        # Comandos EfficientNet (NUEVOS)
        'train_efficientnet',      # Entrenar EfficientNet-B1 (pipeline 4-phase completo)
        'evaluate_efficientnet',   # Comparar EfficientNet vs ResNet estadísticamente
        'visualize_efficientnet'   # Visualizar predicciones de EfficientNet
    ], help='Comando a ejecutar')

    parser.add_argument('--checkpoint', type=str,
                       help='Ruta al checkpoint del modelo (para evaluate/visualize)')
    parser.add_argument('--image', type=str,
                       help='Ruta a imagen específica (para visualize)')

    args = parser.parse_args()

    print_banner()

    if args.command == 'check':
        if not check_environment():
            sys.exit(1)

    elif args.command == 'explore':
        explore_data()

    elif args.command == 'test':
        if not test_setup():
            sys.exit(1)

    elif args.command == 'train1':
        train_phase1()

    elif args.command == 'train2':
        if not train_phase2():
            sys.exit(1)

    elif args.command == 'evaluate':
        checkpoint = args.checkpoint or 'checkpoints/phase2_best.pt'
        if not evaluate_model(checkpoint):
            sys.exit(1)

    elif args.command == 'visualize':
        checkpoint = args.checkpoint or 'checkpoints/phase2_best.pt'
        if not visualize_predictions(checkpoint, args.image):
            sys.exit(1)

    elif args.command == 'visualize_test':
        checkpoint = args.checkpoint or 'checkpoints/phase2_best.pt'
        if not visualize_test_complete(checkpoint):
            sys.exit(1)

    elif args.command == 'visualize_test_complete_loss':
        if not visualize_test_complete_loss():
            sys.exit(1)

    elif args.command == 'train_ensemble':
        if not train_ensemble():
            sys.exit(1)

    elif args.command == 'evaluate_ensemble':
        if not evaluate_ensemble():
            sys.exit(1)

    elif args.command == 'pipeline':
        if not run_full_pipeline():
            sys.exit(1)

    # Comandos geométricos nuevos
    elif args.command == 'train_geometric_phase1':
        if not train_geometric_phase1():
            sys.exit(1)

    elif args.command == 'train_geometric_phase2':
        if not train_geometric_phase2():
            sys.exit(1)

    elif args.command == 'train_geometric_attention':
        if not train_geometric_attention():
            sys.exit(1)

    elif args.command == 'train_geometric_phase3' or args.command == 'train_geometric_symmetry':
        if not train_geometric_phase3():
            sys.exit(1)

    elif args.command == 'train_geometric_final' or args.command == 'train_geometric_complete':
        if not train_geometric_final():
            sys.exit(1)

    elif args.command == 'evaluate_geometric':
        checkpoint = args.checkpoint or 'checkpoints/geometric_final.pt'
        if not evaluate_geometric(checkpoint):
            sys.exit(1)

    elif args.command == 'analyze_geometric':
        if not analyze_geometric_improvements():
            sys.exit(1)

    elif args.command == 'validate_geometric':
        checkpoint = args.checkpoint or 'checkpoints/geometric_final.pt'
        if not validate_geometric_comprehensive(checkpoint):
            sys.exit(1)

    # Comandos EfficientNet
    elif args.command == 'train_efficientnet':
        if not train_efficientnet():
            sys.exit(1)

    elif args.command == 'evaluate_efficientnet':
        if not evaluate_efficientnet():
            sys.exit(1)

    elif args.command == 'visualize_efficientnet':
        checkpoint = args.checkpoint or 'checkpoints/efficientnet/efficientnet_phase4_best.pt'
        if not visualize_efficientnet(checkpoint):
            sys.exit(1)

    print("\n✅ Comando ejecutado exitosamente!")


if __name__ == "__main__":
    main()