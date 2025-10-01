"""
Dataset personalizado para cargar imágenes médicas y sus landmarks correspondientes
Compatible con PyTorch DataLoader y ResNet-18
"""

import os
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import train_test_split

from .transforms import get_transforms, LandmarkTransforms


class LandmarkDataset(Dataset):
    """
    Dataset personalizado para imágenes médicas con landmarks

    Este dataset:
    1. Carga imágenes de diferentes categorías (COVID, Normal, Viral Pneumonia)
    2. Aplica transformaciones compatibles con ImageNet (ResNet-18)
    3. Maneja landmarks de forma correcta con data augmentation
    4. Normaliza coordenadas para regresión
    """

    def __init__(self,
                 annotations_file: str,
                 images_dir: str,
                 transform: Optional[LandmarkTransforms] = None,
                 indices: Optional[List[int]] = None):
        """
        Args:
            annotations_file: Ruta al archivo CSV con coordenadas
            images_dir: Directorio raíz con subdirectorios de categorías
            transform: Objeto de transformaciones
            indices: Lista de índices para subset (train/val/test)
        """
        self.images_dir = Path(images_dir)
        self.transform = transform

        # Cargar anotaciones
        self.annotations = pd.read_csv(annotations_file, header=None)
        print(f"Cargadas {len(self.annotations)} anotaciones del archivo CSV")

        # Filtrar por índices si se proporciona (para train/val/test split)
        if indices is not None:
            self.annotations = self.annotations.iloc[indices].reset_index(drop=True)
            print(f"Usando subset de {len(self.annotations)} muestras")

        # Verificar y cargar datos válidos
        self.valid_samples = self._load_valid_samples()
        print(f"Muestras válidas encontradas: {len(self.valid_samples)}")

        if len(self.valid_samples) == 0:
            raise ValueError("No se encontraron muestras válidas en el dataset")

    def _load_valid_samples(self) -> List[Dict]:
        """
        Cargar solo las muestras que tienen tanto imagen como anotaciones válidas

        Returns:
            Lista de diccionarios con información de muestras válidas
        """
        valid_samples = []

        for idx, row in self.annotations.iterrows():
            try:
                # Extraer información de la fila
                # Formato: [ID, coord1_x, coord1_y, ..., coord15_x, coord15_y, filename]
                coords = row.iloc[1:-1].values.astype(np.float32)  # Coordenadas
                filename = row.iloc[-1]  # Nombre del archivo

                # Buscar imagen en subdirectorios
                image_path = self._find_image_path(filename)

                if image_path is not None and image_path.exists():
                    # Verificar que las coordenadas sean válidas
                    if len(coords) == 30:  # 15 landmarks * 2 coordenadas
                        valid_samples.append({
                            'image_path': image_path,
                            'landmarks': coords,
                            'filename': filename,
                            'category': self._get_category_from_filename(filename)
                        })
                    else:
                        print(f"Advertencia: Número incorrecto de coordenadas para {filename}: {len(coords)}")
                else:
                    print(f"Advertencia: Imagen no encontrada para {filename}")

            except Exception as e:
                print(f"Error procesando fila {idx}: {e}")
                continue

        return valid_samples

    def _find_image_path(self, filename: str) -> Optional[Path]:
        """
        Buscar imagen en subdirectorios de categorías

        Args:
            filename: Nombre del archivo (sin extensión)

        Returns:
            Path del archivo si existe, None si no se encuentra
        """
        # Buscar en todos los subdirectorios
        for category_dir in self.images_dir.iterdir():
            if category_dir.is_dir():
                # Probar diferentes extensiones
                for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
                    image_path = category_dir / f"{filename}{ext}"
                    if image_path.exists():
                        return image_path

        return None

    def _get_category_from_filename(self, filename: str) -> str:
        """
        Extraer categoría del nombre de archivo

        Args:
            filename: Nombre del archivo

        Returns:
            Categoría de la imagen
        """
        if 'COVID' in filename:
            return 'COVID'
        elif 'Normal' in filename:
            return 'Normal'
        elif 'Viral' in filename:
            return 'Viral_Pneumonia'
        else:
            return 'Unknown'

    def __len__(self) -> int:
        """Retorna el número de muestras en el dataset"""
        return len(self.valid_samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Obtener una muestra del dataset

        Args:
            idx: Índice de la muestra

        Returns:
            Tupla de (imagen, landmarks, metadata)
            - imagen: Tensor de forma (3, 224, 224) normalizado para ImageNet
            - landmarks: Tensor de forma (30,) con coordenadas normalizadas [0,1]
            - metadata: Diccionario con información adicional
        """
        try:
            sample = self.valid_samples[idx]

            # Cargar imagen
            image = cv2.imread(str(sample['image_path']))
            if image is None:
                raise ValueError(f"No se pudo cargar la imagen: {sample['image_path']}")

            # Obtener landmarks
            landmarks = sample['landmarks'].copy()

            # Aplicar transformaciones si están disponibles
            if self.transform is not None:
                # Check if transform supports category parameter (medical transforms)
                if hasattr(self.transform, '__call__') and 'category' in self.transform.__call__.__code__.co_varnames:
                    image_tensor, landmarks_tensor = self.transform(image, landmarks, category=sample['category'])
                else:
                    image_tensor, landmarks_tensor = self.transform(image, landmarks)
            else:
                # Transformación básica sin augmentation
                basic_transform = get_transforms(is_training=False)
                image_tensor, landmarks_tensor = basic_transform(image, landmarks)

            # Metadata
            metadata = {
                'filename': sample['filename'],
                'category': sample['category'],
                'image_path': str(sample['image_path']),
                'original_landmarks': torch.from_numpy(sample['landmarks'])
            }

            return image_tensor, landmarks_tensor, metadata

        except Exception as e:
            print(f"Error cargando muestra {idx}: {e}")
            # Retornar muestra alternativa en caso de error
            if idx < len(self.valid_samples) - 1:
                return self.__getitem__(idx + 1)
            else:
                return self.__getitem__(0)

    def get_category_distribution(self) -> Dict[str, int]:
        """
        Obtener distribución de categorías en el dataset

        Returns:
            Diccionario con conteo por categoría
        """
        categories = [sample['category'] for sample in self.valid_samples]
        return pd.Series(categories).value_counts().to_dict()

    def get_landmarks_statistics(self) -> Dict[str, np.ndarray]:
        """
        Calcular estadísticas de landmarks para análisis

        Returns:
            Diccionario con estadísticas (mean, std, min, max)
        """
        all_landmarks = np.array([sample['landmarks'] for sample in self.valid_samples])

        return {
            'mean': np.mean(all_landmarks, axis=0),
            'std': np.std(all_landmarks, axis=0),
            'min': np.min(all_landmarks, axis=0),
            'max': np.max(all_landmarks, axis=0)
        }


def create_data_splits(annotations_file: str,
                      train_ratio: float = 0.7,
                      val_ratio: float = 0.15,
                      test_ratio: float = 0.15,
                      random_seed: int = 42) -> Tuple[List[int], List[int], List[int]]:
    """
    Crear splits de datos para entrenamiento, validación y prueba

    Args:
        annotations_file: Archivo de anotaciones
        train_ratio: Proporción para entrenamiento
        val_ratio: Proporción para validación
        test_ratio: Proporción para prueba
        random_seed: Semilla aleatoria para reproducibilidad

    Returns:
        Tupla de (índices_train, índices_val, índices_test)
    """
    # Verificar que las proporciones sumen 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Las proporciones deben sumar 1.0"

    # Cargar anotaciones para obtener el número total de muestras
    annotations = pd.read_csv(annotations_file, header=None)
    total_samples = len(annotations)

    # Crear índices
    indices = list(range(total_samples))

    # Primera división: train vs (val + test)
    train_indices, temp_indices = train_test_split(
        indices,
        test_size=(val_ratio + test_ratio),
        random_state=random_seed,
        shuffle=True
    )

    # Segunda división: val vs test
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=random_seed,
        shuffle=True
    )

    print(f"División de datos:")
    print(f"  Total: {total_samples}")
    print(f"  Entrenamiento: {len(train_indices)} ({len(train_indices)/total_samples:.1%})")
    print(f"  Validación: {len(val_indices)} ({len(val_indices)/total_samples:.1%})")
    print(f"  Prueba: {len(test_indices)} ({len(test_indices)/total_samples:.1%})")

    return train_indices, val_indices, test_indices


def create_dataloaders(annotations_file: str,
                      images_dir: str,
                      batch_size: int = 32,
                      num_workers: int = 4,
                      pin_memory: bool = True,
                      use_medical_transforms: bool = False,
                      **split_kwargs) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Crear DataLoaders para entrenamiento, validación y prueba

    Args:
        annotations_file: Archivo de anotaciones
        images_dir: Directorio de imágenes
        batch_size: Tamaño del batch
        num_workers: Número de workers para carga paralela
        pin_memory: Si usar pin_memory para GPU
        use_medical_transforms: Si usar transformaciones médicas avanzadas
        **split_kwargs: Argumentos para create_data_splits

    Returns:
        Tupla de (train_loader, val_loader, test_loader)
    """
    # Crear splits de datos
    train_indices, val_indices, test_indices = create_data_splits(annotations_file, **split_kwargs)

    # Crear transformaciones
    if use_medical_transforms:
        from .medical_transforms import get_medical_transforms
        train_transform = get_medical_transforms(image_size=(224, 224), is_training=True, enable_medical_aug=True)
        val_transform = get_medical_transforms(image_size=(224, 224), is_training=False, enable_medical_aug=False)
    else:
        train_transform = get_transforms(image_size=(224, 224), is_training=True)
        val_transform = get_transforms(image_size=(224, 224), is_training=False)

    # Crear datasets
    train_dataset = LandmarkDataset(
        annotations_file=annotations_file,
        images_dir=images_dir,
        transform=train_transform,
        indices=train_indices
    )

    val_dataset = LandmarkDataset(
        annotations_file=annotations_file,
        images_dir=images_dir,
        transform=val_transform,
        indices=val_indices
    )

    test_dataset = LandmarkDataset(
        annotations_file=annotations_file,
        images_dir=images_dir,
        transform=val_transform,
        indices=test_indices
    )

    # Crear DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Para consistencia en batch size
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    print(f"\nDataLoaders creados:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val: {len(val_loader)} batches")
    print(f"  Test: {len(test_loader)} batches")

    return train_loader, val_loader, test_loader