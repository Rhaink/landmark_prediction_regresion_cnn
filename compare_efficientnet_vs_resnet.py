#!/usr/bin/env python3
"""
Comparación Estadística: EfficientNet-B1 vs ResNet-18
Análisis riguroso de mejora en precisión de landmarks

Métricas:
- Mean pixel error y desviación estándar
- Mejora porcentual y absoluta
- Significancia estadística (paired t-test)
- Análisis por categoría médica (COVID, Normal, Viral)
- Análisis por landmark individual
- Clinical thresholds achievement
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Dict, List, Tuple

# Importar módulos del proyecto
from src.data.dataset import create_dataloaders
from src.models.efficientnet_regressor import EfficientNetLandmarkRegressor
from src.models.resnet_regressor import ResNetLandmarkRegressor
from src.training.utils import setup_device


class ModelComparator:
    """
    Comparador estadístico de modelos para landmark regression

    Implementa análisis riguroso de mejora con:
    - Statistical significance testing
    - Per-category analysis
    - Per-landmark analysis
    - Clinical threshold evaluation
    """

    def __init__(self):
        self.device = setup_device(use_gpu=True, gpu_id=0)
        print("=" * 70)
        print("📊 EFFICIENTNET-B1 vs RESNET-18 STATISTICAL COMPARISON")
        print("=" * 70)
        print(f"Device: {self.device}\n")

    def load_models(
        self,
        efficientnet_path: str,
        resnet_path: str
    ) -> Tuple[torch.nn.Module, torch.nn.Module]:
        """
        Cargar ambos modelos desde checkpoints

        Args:
            efficientnet_path: Path al checkpoint de EfficientNet
            resnet_path: Path al checkpoint de ResNet

        Returns:
            Tuple de (efficientnet_model, resnet_model)
        """
        print("🏗️ Cargando modelos...")

        # Cargar EfficientNet
        print(f"\n  EfficientNet: {efficientnet_path}")
        efficientnet, _ = EfficientNetLandmarkRegressor.load_from_checkpoint(
            efficientnet_path,
            device=self.device
        )
        efficientnet.eval()

        # Cargar ResNet
        print(f"  ResNet: {resnet_path}")
        resnet = ResNetLandmarkRegressor(num_landmarks=15, pretrained=False, dropout_rate=0.5)
        checkpoint = torch.load(resnet_path, map_location=str(self.device))
        resnet.load_state_dict(checkpoint["model_state_dict"])
        resnet = resnet.to(self.device)
        resnet.eval()

        print("\n✓ Ambos modelos cargados exitosamente")

        return efficientnet, resnet

    def compute_pixel_errors(
        self,
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Calcular errores de píxeles para todas las muestras

        Args:
            model: Modelo a evaluar
            test_loader: DataLoader de test

        Returns:
            Tuple de (per_image_errors, per_landmark_errors, categories)
        """
        all_errors_per_image = []  # Error promedio por imagen
        all_errors_per_landmark = []  # Errores por landmark
        all_categories = []

        with torch.no_grad():
            for images, landmarks, metadata in tqdm(test_loader, desc="Evaluating"):
                images = images.to(self.device)
                landmarks = landmarks.to(self.device)

                predictions = model(images)

                # Reshape para calcular errores
                pred_reshaped = predictions.view(-1, 15, 2)
                target_reshaped = landmarks.view(-1, 15, 2)

                # Distancias euclidianas por landmark
                distances = torch.norm(pred_reshaped - target_reshaped, dim=2)  # (batch, 15)
                pixel_distances = distances * 224  # Convert to pixels

                # Error promedio por imagen
                errors_per_image = torch.mean(pixel_distances, dim=1)  # (batch,)
                all_errors_per_image.extend(errors_per_image.cpu().numpy())

                # Errores por landmark (para análisis detallado)
                all_errors_per_landmark.append(pixel_distances.cpu().numpy())

                # Categorías
                for meta in metadata:
                    all_categories.append(meta['category'])

        # Concatenar errores por landmark
        all_errors_per_landmark = np.concatenate(all_errors_per_landmark, axis=0)  # (n_samples, 15)

        return (
            np.array(all_errors_per_image),
            all_errors_per_landmark,
            all_categories
        )

    def statistical_comparison(
        self,
        errors_model1: np.ndarray,
        errors_model2: np.ndarray,
        model1_name: str = "EfficientNet-B1",
        model2_name: str = "ResNet-18"
    ) -> Dict:
        """
        Comparación estadística entre dos modelos

        Args:
            errors_model1: Errores del modelo 1
            errors_model2: Errores del modelo 2
            model1_name: Nombre del modelo 1
            model2_name: Nombre del modelo 2

        Returns:
            Diccionario con resultados estadísticos
        """
        # Métricas básicas
        mean1 = np.mean(errors_model1)
        std1 = np.std(errors_model1)
        median1 = np.median(errors_model1)

        mean2 = np.mean(errors_model2)
        std2 = np.std(errors_model2)
        median2 = np.median(errors_model2)

        # Mejora
        absolute_improvement = mean2 - mean1
        percent_improvement = (absolute_improvement / mean2) * 100

        # Paired t-test (samples are paired: same test images)
        t_statistic, p_value = stats.ttest_rel(errors_model1, errors_model2)

        # Wilcoxon signed-rank test (non-parametric alternative)
        wilcoxon_statistic, wilcoxon_p = stats.wilcoxon(errors_model1, errors_model2)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((std1**2 + std2**2) / 2)
        cohens_d = absolute_improvement / pooled_std

        results = {
            'model1_name': model1_name,
            'model2_name': model2_name,
            'model1_mean': mean1,
            'model1_std': std1,
            'model1_median': median1,
            'model2_mean': mean2,
            'model2_std': std2,
            'model2_median': median2,
            'absolute_improvement': absolute_improvement,
            'percent_improvement': percent_improvement,
            't_statistic': t_statistic,
            'p_value': p_value,
            'wilcoxon_statistic': wilcoxon_statistic,
            'wilcoxon_p': wilcoxon_p,
            'cohens_d': cohens_d,
            'is_significant': p_value < 0.05
        }

        return results

    def category_analysis(
        self,
        errors_model1: np.ndarray,
        errors_model2: np.ndarray,
        categories: List[str],
        model1_name: str = "EfficientNet-B1",
        model2_name: str = "ResNet-18"
    ) -> pd.DataFrame:
        """
        Análisis por categoría médica

        Args:
            errors_model1: Errores del modelo 1
            errors_model2: Errores del modelo 2
            categories: Lista de categorías
            model1_name: Nombre del modelo 1
            model2_name: Nombre del modelo 2

        Returns:
            DataFrame con resultados por categoría
        """
        # Agrupar por categoría
        categories_array = np.array(categories)
        unique_categories = np.unique(categories_array)

        results = []

        for category in unique_categories:
            mask = categories_array == category

            errors1_cat = errors_model1[mask]
            errors2_cat = errors_model2[mask]

            mean1 = np.mean(errors1_cat)
            std1 = np.std(errors1_cat)

            mean2 = np.mean(errors2_cat)
            std2 = np.std(errors2_cat)

            improvement = mean2 - mean1
            percent_improvement = (improvement / mean2) * 100

            # T-test por categoría
            t_stat, p_val = stats.ttest_rel(errors1_cat, errors2_cat)

            results.append({
                'Category': category,
                'N_samples': len(errors1_cat),
                f'{model1_name}_mean': mean1,
                f'{model1_name}_std': std1,
                f'{model2_name}_mean': mean2,
                f'{model2_name}_std': std2,
                'Improvement_px': improvement,
                'Improvement_%': percent_improvement,
                'p_value': p_val,
                'Significant': '✓' if p_val < 0.05 else '✗'
            })

        return pd.DataFrame(results)

    def landmark_analysis(
        self,
        errors_model1: np.ndarray,
        errors_model2: np.ndarray,
        model1_name: str = "EfficientNet-B1",
        model2_name: str = "ResNet-18"
    ) -> pd.DataFrame:
        """
        Análisis por landmark individual

        Args:
            errors_model1: Errores por landmark del modelo 1 (n_samples, 15)
            errors_model2: Errores por landmark del modelo 2 (n_samples, 15)
            model1_name: Nombre del modelo 1
            model2_name: Nombre del modelo 2

        Returns:
            DataFrame con resultados por landmark
        """
        landmark_names = [
            "Mediastino superior", "Mediastino inferior",
            "Ápice izq", "Ápice der",
            "Hilio izq", "Hilio der",
            "Base izq", "Base der",
            "Centro medio", "Centro inferior", "Centro superior",
            "Borde izq", "Borde der",
            "Landmark 13", "Landmark 14"
        ]

        results = []

        for i in range(15):
            errors1_lm = errors_model1[:, i]
            errors2_lm = errors_model2[:, i]

            mean1 = np.mean(errors1_lm)
            std1 = np.std(errors1_lm)

            mean2 = np.mean(errors2_lm)
            std2 = np.std(errors2_lm)

            improvement = mean2 - mean1
            percent_improvement = (improvement / mean2) * 100

            # T-test por landmark
            t_stat, p_val = stats.ttest_rel(errors1_lm, errors2_lm)

            results.append({
                'Landmark_ID': i,
                'Landmark_Name': landmark_names[i],
                f'{model1_name}_mean': mean1,
                f'{model1_name}_std': std1,
                f'{model2_name}_mean': mean2,
                f'{model2_name}_std': std2,
                'Improvement_px': improvement,
                'Improvement_%': percent_improvement,
                'p_value': p_val,
                'Significant': '✓' if p_val < 0.05 else '✗'
            })

        return pd.DataFrame(results)

    def clinical_thresholds_analysis(
        self,
        errors_model1: np.ndarray,
        errors_model2: np.ndarray,
        model1_name: str = "EfficientNet-B1",
        model2_name: str = "ResNet-18"
    ) -> pd.DataFrame:
        """
        Análisis de umbrales clínicos

        Args:
            errors_model1: Errores del modelo 1
            errors_model2: Errores del modelo 2

        Returns:
            DataFrame con análisis de thresholds
        """
        thresholds = {
            'Sub-pixel (research)': 5.0,
            'Super-precision': 6.0,
            'Clinical excellence': 8.5,
            'Clinically useful': 15.0,
            'General analysis': 20.0
        }

        results = []

        for threshold_name, threshold_value in thresholds.items():
            percent1 = (errors_model1 < threshold_value).mean() * 100
            percent2 = (errors_model2 < threshold_value).mean() * 100

            mean1 = np.mean(errors_model1)
            mean2 = np.mean(errors_model2)

            achieved1 = mean1 < threshold_value
            achieved2 = mean2 < threshold_value

            results.append({
                'Threshold': threshold_name,
                'Value_px': threshold_value,
                f'{model1_name}_%_below': percent1,
                f'{model1_name}_achieved': '✅' if achieved1 else '❌',
                f'{model2_name}_%_below': percent2,
                f'{model2_name}_achieved': '✅' if achieved2 else '❌',
                'Improvement': percent1 - percent2
            })

        return pd.DataFrame(results)

    def generate_visualizations(
        self,
        errors_model1: np.ndarray,
        errors_model2: np.ndarray,
        errors_per_landmark_model1: np.ndarray,
        errors_per_landmark_model2: np.ndarray,
        categories: List[str],
        model1_name: str = "EfficientNet-B1",
        model2_name: str = "ResNet-18",
        save_dir: str = "evaluation_results/efficientnet_comparison"
    ):
        """
        Generar visualizaciones de comparación

        Args:
            errors_model1: Errores del modelo 1
            errors_model2: Errores del modelo 2
            errors_per_landmark_model1: Errores por landmark del modelo 1
            errors_per_landmark_model2: Errores por landmark del modelo 2
            categories: Lista de categorías
            model1_name: Nombre del modelo 1
            model2_name: Nombre del modelo 2
            save_dir: Directorio donde guardar visualizaciones
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # 1. Distribución de errores
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.hist(errors_model1, bins=30, alpha=0.7, label=model1_name, color='green')
        plt.hist(errors_model2, bins=30, alpha=0.7, label=model2_name, color='blue')
        plt.axvline(np.mean(errors_model1), color='green', linestyle='--', linewidth=2, label=f'{model1_name} mean')
        plt.axvline(np.mean(errors_model2), color='blue', linestyle='--', linewidth=2, label=f'{model2_name} mean')
        plt.xlabel('Pixel Error')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')
        plt.legend()
        plt.grid(alpha=0.3)

        plt.subplot(1, 2, 2)
        data_box = [errors_model1, errors_model2]
        plt.boxplot(data_box, labels=[model1_name, model2_name])
        plt.ylabel('Pixel Error')
        plt.title('Error Distribution Boxplot')
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{save_dir}/error_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved: {save_dir}/error_distributions.png")

        # 2. Comparación por landmark
        landmark_means1 = np.mean(errors_per_landmark_model1, axis=0)
        landmark_means2 = np.mean(errors_per_landmark_model2, axis=0)

        plt.figure(figsize=(14, 6))
        x = np.arange(15)
        width = 0.35

        plt.bar(x - width/2, landmark_means1, width, label=model1_name, color='green', alpha=0.8)
        plt.bar(x + width/2, landmark_means2, width, label=model2_name, color='blue', alpha=0.8)

        plt.xlabel('Landmark ID')
        plt.ylabel('Mean Pixel Error')
        plt.title('Per-Landmark Error Comparison')
        plt.xticks(x)
        plt.legend()
        plt.grid(alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(f"{save_dir}/per_landmark_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved: {save_dir}/per_landmark_comparison.png")

        # 3. Comparación por categoría
        categories_array = np.array(categories)
        unique_categories = np.unique(categories_array)

        cat_means1 = []
        cat_means2 = []

        for cat in unique_categories:
            mask = categories_array == cat
            cat_means1.append(np.mean(errors_model1[mask]))
            cat_means2.append(np.mean(errors_model2[mask]))

        plt.figure(figsize=(10, 6))
        x = np.arange(len(unique_categories))
        width = 0.35

        plt.bar(x - width/2, cat_means1, width, label=model1_name, color='green', alpha=0.8)
        plt.bar(x + width/2, cat_means2, width, label=model2_name, color='blue', alpha=0.8)

        plt.xlabel('Medical Category')
        plt.ylabel('Mean Pixel Error')
        plt.title('Per-Category Error Comparison')
        plt.xticks(x, unique_categories, rotation=15)
        plt.legend()
        plt.grid(alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(f"{save_dir}/per_category_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved: {save_dir}/per_category_comparison.png")

    def compare(
        self,
        efficientnet_path: str = "checkpoints/efficientnet/efficientnet_phase4_best.pt",
        resnet_path: str = "checkpoints/geometric_complete.pt",
        save_results: bool = True,
        generate_plots: bool = True
    ):
        """
        Ejecutar comparación completa

        Args:
            efficientnet_path: Path al checkpoint de EfficientNet
            resnet_path: Path al checkpoint de ResNet
            save_results: Si guardar resultados en archivos
            generate_plots: Si generar visualizaciones
        """
        # Cargar modelos
        efficientnet, resnet = self.load_models(efficientnet_path, resnet_path)

        # Crear test loader
        print("\n📊 Cargando test set...")
        _, _, test_loader = create_dataloaders(
            annotations_file="data/coordenadas/coordenadas_maestro.csv",
            images_dir="data/dataset",
            batch_size=16,
            num_workers=4,
            pin_memory=True,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42
        )

        print(f"✓ Test set: {len(test_loader.dataset)} samples\n")

        # Evaluar EfficientNet
        print("🔍 Evaluando EfficientNet-B1...")
        efficientnet_errors, efficientnet_per_landmark, categories = self.compute_pixel_errors(
            efficientnet, test_loader
        )

        # Evaluar ResNet
        print("🔍 Evaluando ResNet-18...")
        resnet_errors, resnet_per_landmark, _ = self.compute_pixel_errors(
            resnet, test_loader
        )

        # Comparación estadística
        print("\n" + "=" * 70)
        print("📈 STATISTICAL COMPARISON RESULTS")
        print("=" * 70)

        stats_results = self.statistical_comparison(
            efficientnet_errors,
            resnet_errors,
            "EfficientNet-B1",
            "ResNet-18"
        )

        print(f"\n{stats_results['model1_name']}:")
        print(f"  Mean ± Std: {stats_results['model1_mean']:.2f} ± {stats_results['model1_std']:.2f} px")
        print(f"  Median: {stats_results['model1_median']:.2f} px")

        print(f"\n{stats_results['model2_name']}:")
        print(f"  Mean ± Std: {stats_results['model2_mean']:.2f} ± {stats_results['model2_std']:.2f} px")
        print(f"  Median: {stats_results['model2_median']:.2f} px")

        print(f"\nImprovement:")
        print(f"  Absolute: {stats_results['absolute_improvement']:.2f} px")
        print(f"  Percentage: {stats_results['percent_improvement']:.2f}%")

        print(f"\nStatistical Significance:")
        print(f"  t-statistic: {stats_results['t_statistic']:.4f}")
        print(f"  p-value: {stats_results['p_value']:.6f}")
        print(f"  Significant (p<0.05): {'✅ YES' if stats_results['is_significant'] else '❌ NO'}")
        print(f"  Wilcoxon p-value: {stats_results['wilcoxon_p']:.6f}")
        print(f"  Cohen's d: {stats_results['cohens_d']:.3f}")

        # Análisis por categoría
        print("\n" + "=" * 70)
        print("📊 PER-CATEGORY ANALYSIS")
        print("=" * 70)

        category_results = self.category_analysis(
            efficientnet_errors,
            resnet_errors,
            categories,
            "EfficientNet-B1",
            "ResNet-18"
        )

        print(category_results.to_string(index=False))

        # Análisis por landmark
        print("\n" + "=" * 70)
        print("📍 PER-LANDMARK ANALYSIS")
        print("=" * 70)

        landmark_results = self.landmark_analysis(
            efficientnet_per_landmark,
            resnet_per_landmark,
            "EfficientNet-B1",
            "ResNet-18"
        )

        print(landmark_results.to_string(index=False))

        # Clinical thresholds
        print("\n" + "=" * 70)
        print("🏥 CLINICAL THRESHOLDS ANALYSIS")
        print("=" * 70)

        threshold_results = self.clinical_thresholds_analysis(
            efficientnet_errors,
            resnet_errors,
            "EfficientNet-B1",
            "ResNet-18"
        )

        print(threshold_results.to_string(index=False))

        # Guardar resultados
        if save_results:
            save_dir = "evaluation_results/efficientnet_comparison"
            Path(save_dir).mkdir(parents=True, exist_ok=True)

            print(f"\n💾 Guardando resultados en {save_dir}...")

            # Guardar estadísticas generales
            stats_df = pd.DataFrame([stats_results])
            stats_df.to_csv(f"{save_dir}/statistical_comparison.csv", index=False)

            # Guardar análisis por categoría
            category_results.to_csv(f"{save_dir}/per_category_analysis.csv", index=False)

            # Guardar análisis por landmark
            landmark_results.to_csv(f"{save_dir}/per_landmark_analysis.csv", index=False)

            # Guardar thresholds
            threshold_results.to_csv(f"{save_dir}/clinical_thresholds.csv", index=False)

            print("✓ Todos los resultados guardados")

        # Generar visualizaciones
        if generate_plots:
            print("\n🎨 Generando visualizaciones...")
            self.generate_visualizations(
                efficientnet_errors,
                resnet_errors,
                efficientnet_per_landmark,
                resnet_per_landmark,
                categories,
                "EfficientNet-B1",
                "ResNet-18"
            )

        print("\n" + "=" * 70)
        print("✅ COMPARISON COMPLETED")
        print("=" * 70)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compare EfficientNet-B1 vs ResNet-18")
    parser.add_argument('--efficientnet', type=str,
                       default="checkpoints/efficientnet/efficientnet_phase4_best.pt",
                       help="Path to EfficientNet checkpoint")
    parser.add_argument('--resnet', type=str,
                       default="checkpoints/geometric_complete.pt",
                       help="Path to ResNet checkpoint")
    parser.add_argument('--no-save', action='store_true',
                       help="Don't save results to files")
    parser.add_argument('--no-plots', action='store_true',
                       help="Don't generate plots")

    args = parser.parse_args()

    comparator = ModelComparator()
    comparator.compare(
        efficientnet_path=args.efficientnet,
        resnet_path=args.resnet,
        save_results=not args.no_save,
        generate_plots=not args.no_plots
    )


if __name__ == "__main__":
    main()
