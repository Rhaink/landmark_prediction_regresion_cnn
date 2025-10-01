"""
Transformaciones para imágenes y landmarks compatibles con ImageNet y data augmentation
"""

import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from typing import Tuple, Dict, Any
import random


class LandmarkTransforms:
    """
    Clase para manejar transformaciones de imágenes y sus landmarks correspondientes
    """

    def __init__(self, image_size: Tuple[int, int] = (224, 224), is_training: bool = True):
        """
        Args:
            image_size: Tamaño objetivo de la imagen (altura, ancho)
            is_training: Si es True, aplica data augmentation
        """
        self.image_size = image_size
        self.is_training = is_training

        # Estadísticas de ImageNet para normalización
        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]

        # Configurar transformaciones base
        self._setup_transforms()

    def _setup_transforms(self):
        """Configurar transformaciones de imagen sin landmarks"""
        base_transforms = [
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.imagenet_mean, std=self.imagenet_std)
        ]

        self.image_transforms = transforms.Compose(base_transforms)

    def __call__(self, image: np.ndarray, landmarks: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aplicar transformaciones a imagen y landmarks

        Args:
            image: Imagen numpy array (H, W, C)
            landmarks: Array de landmarks (num_landmarks * 2,) formato [x1, y1, x2, y2, ...]

        Returns:
            Tupla de (imagen_transformada, landmarks_transformados)
        """
        # Obtener dimensiones originales
        original_height, original_width = image.shape[:2]

        # Asegurar que la imagen sea RGB (3 canales)
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # Aplicar data augmentation si está en modo entrenamiento
        if self.is_training:
            image, landmarks = self._apply_augmentation(image, landmarks, original_width, original_height)

        # Redimensionar imagen y landmarks
        image_resized = cv2.resize(image, self.image_size)
        landmarks_resized = self._resize_landmarks(landmarks, original_width, original_height,
                                                   self.image_size[0], self.image_size[1])

        # Convertir imagen a tensor y normalizar
        image_tensor = self._numpy_to_tensor(image_resized)

        # Normalizar landmarks entre 0 y 1
        landmarks_normalized = self._normalize_landmarks(landmarks_resized, self.image_size[0], self.image_size[1])

        return image_tensor, landmarks_normalized

    def _apply_augmentation(self, image: np.ndarray, landmarks: np.ndarray,
                           width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aplicar data augmentation compatible con landmarks

        IMPORTANTE: Para regresión de landmarks, debemos ser cuidadosos con las transformaciones
        que alteran las posiciones espaciales de los puntos
        """
        # Flip horizontal (con probabilidad 0.5)
        if random.random() < 0.5:
            image = cv2.flip(image, 1)  # Flip horizontal
            landmarks = self._flip_landmarks_horizontal(landmarks, width)

        # Rotación ligera (±2 grados - más apropiado para imágenes médicas)
        if random.random() < 0.3:
            angle = random.uniform(-2, 2)
            image, landmarks = self._rotate_image_and_landmarks(image, landmarks, angle, width, height)

        # Ajustes de brillo y contraste (no afectan landmarks)
        if random.random() < 0.5:
            # Brillo
            brightness_factor = random.uniform(0.8, 1.2)
            image = np.clip(image * brightness_factor, 0, 255).astype(np.uint8)

        if random.random() < 0.5:
            # Contraste
            contrast_factor = random.uniform(0.8, 1.2)
            mean = np.mean(image)
            image = np.clip((image - mean) * contrast_factor + mean, 0, 255).astype(np.uint8)

        return image, landmarks

    def _flip_landmarks_horizontal(self, landmarks: np.ndarray, width: int) -> np.ndarray:
        """
        Aplicar flip horizontal a landmarks

        Para flip horizontal: nuevo_x = width - x_original
        Las coordenadas Y no cambian
        """
        landmarks_flipped = landmarks.copy()

        # Procesar coordenadas x (índices pares) e y (índices impares)
        for i in range(0, len(landmarks), 2):
            landmarks_flipped[i] = width - landmarks[i]  # Flip coordenada x
            # landmarks_flipped[i+1] permanece igual (coordenada y)

        return landmarks_flipped

    def _rotate_image_and_landmarks(self, image: np.ndarray, landmarks: np.ndarray,
                                   angle: float, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rotar imagen y landmarks

        NOTA: La rotación es más compleja para landmarks, requiere transformación matricial
        """
        # Centro de rotación
        center = (width // 2, height // 2)

        # Matriz de rotación
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Rotar imagen
        image_rotated = cv2.warpAffine(image, rotation_matrix, (width, height))

        # Rotar landmarks
        landmarks_rotated = landmarks.copy()

        # Convertir landmarks a formato de puntos para transformación
        points = landmarks.reshape(-1, 2)  # Reshape to (num_landmarks, 2)

        # Aplicar transformación a cada punto
        ones = np.ones(shape=(len(points), 1))
        points_homogeneous = np.hstack([points, ones])  # Coordenadas homogéneas

        # Aplicar matriz de rotación
        transformed_points = rotation_matrix.dot(points_homogeneous.T).T

        # Convertir de vuelta al formato original
        landmarks_rotated = transformed_points.flatten()

        # Asegurar que los landmarks estén dentro de los límites de la imagen
        landmarks_rotated = np.clip(landmarks_rotated, 0, [width, height] * (len(landmarks_rotated) // 2))

        return image_rotated, landmarks_rotated

    def _resize_landmarks(self, landmarks: np.ndarray, orig_width: int, orig_height: int,
                         new_width: int, new_height: int) -> np.ndarray:
        """
        Redimensionar landmarks según el nuevo tamaño de imagen

        Factor de escala = nuevo_tamaño / tamaño_original
        """
        scale_x = new_width / orig_width
        scale_y = new_height / orig_height

        landmarks_resized = landmarks.copy()

        # Escalar coordenadas x e y
        for i in range(0, len(landmarks), 2):
            landmarks_resized[i] *= scale_x      # Coordenada x
            landmarks_resized[i + 1] *= scale_y  # Coordenada y

        return landmarks_resized

    def _normalize_landmarks(self, landmarks: np.ndarray, width: int, height: int) -> torch.Tensor:
        """
        Normalizar landmarks entre 0 y 1

        IMPORTANTE: Esta normalización es crucial para el entrenamiento de regresión
        Ayuda a que el modelo aprenda mejor y converja más rápido
        """
        landmarks_normalized = landmarks.copy().astype(np.float32)

        # Normalizar coordenadas x e y
        for i in range(0, len(landmarks), 2):
            landmarks_normalized[i] /= width       # Normalizar x
            landmarks_normalized[i + 1] /= height  # Normalizar y

        return torch.from_numpy(landmarks_normalized)

    def _numpy_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """
        Convertir imagen numpy a tensor y aplicar normalización ImageNet
        """
        # Convertir de BGR a RGB si es necesario
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convertir a tensor (C, H, W) y normalizar a [0, 1]
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # Aplicar normalización ImageNet
        for channel in range(3):
            image_tensor[channel] = (image_tensor[channel] - self.imagenet_mean[channel]) / self.imagenet_std[channel]

        return image_tensor

    def denormalize_landmarks(self, normalized_landmarks: torch.Tensor,
                             width: int, height: int) -> torch.Tensor:
        """
        Desnormalizar landmarks de [0,1] a coordenadas de píxeles

        Útil para visualización y evaluación
        """
        denormalized = normalized_landmarks.clone()

        # Desnormalizar coordenadas
        for i in range(0, len(denormalized), 2):
            denormalized[i] *= width      # Desnormalizar x
            denormalized[i + 1] *= height # Desnormalizar y

        return denormalized


def get_transforms(image_size: Tuple[int, int] = (224, 224), is_training: bool = True) -> LandmarkTransforms:
    """
    Factory function para obtener transformaciones

    Args:
        image_size: Tamaño objetivo de imagen
        is_training: Si aplicar data augmentation

    Returns:
        Objeto LandmarkTransforms configurado
    """
    return LandmarkTransforms(image_size=image_size, is_training=is_training)


def get_medical_transforms(image_size: Tuple[int, int] = (224, 224),
                          is_training: bool = True,
                          enable_medical_aug: bool = True,
                          validation_tolerance: float = 0.20,
                          verbose: bool = False):
    """
    Factory function para obtener transformaciones médicas avanzadas

    Para usar medical transforms, importar desde medical_transforms.py
    Esta función es un wrapper para compatibilidad

    Args:
        image_size: Tamaño objetivo de imagen
        is_training: Si aplicar data augmentation
        enable_medical_aug: Habilitar augmentación médica
        validation_tolerance: Tolerancia para validación anatómica
        verbose: Imprimir warnings de validación

    Returns:
        Objeto MedicalLandmarkTransforms configurado
    """
    from .medical_transforms import get_medical_transforms as _get_medical_transforms
    return _get_medical_transforms(
        image_size=image_size,
        is_training=is_training,
        enable_medical_aug=enable_medical_aug,
        validation_tolerance=validation_tolerance,
        verbose=verbose
    )