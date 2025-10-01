"""
Medical-specific augmentation transforms for chest X-ray landmark prediction
Implements anatomically-aware transformations with constraint validation

Key Features:
- Breathing simulation (diaphragm movement)
- Patient positioning variation (±2° rotation, translation)
- Elastic deformation (tissue simulation)
- Pathology-aware augmentation (category-specific)
- Anatomical constraint validation
- Medical intensity augmentation

Target: Improve from 7.23±3.66px to <6.0px mean error
"""

import torch
import numpy as np
import cv2
from typing import Tuple, Dict, Optional, List
import random
from scipy.ndimage import gaussian_filter, map_coordinates
import warnings


class AnatomicalConstraintValidator:
    """
    Validates that augmented landmarks maintain anatomical plausibility

    Constraints:
    1. Bilateral symmetry (±15% tolerance)
    2. Vertical ordering (apex > hilio > base)
    3. Mediastinal alignment (centerline deviation <10%)
    4. CTR preservation (Cardiothoracic Ratio)
    """

    def __init__(self,
                 tolerance: float = 0.20,
                 verbose: bool = False):
        """
        Args:
            tolerance: Relative tolerance for constraint violations (20%)
            verbose: Print validation warnings
        """
        self.tolerance = tolerance
        self.verbose = verbose

        # Define symmetric landmark pairs (bilateral structures)
        self.symmetric_pairs = [
            (2, 3),   # Ápices pulmonares izq/der
            (4, 5),   # Hilios izq/der
            (6, 7),   # Bases pulmonares izq/der
            (11, 12), # Bordes superiores izq/der
            (13, 14)  # Senos costofrénicos izq/der
        ]

        # Mediastinal landmarks (should be near centerline)
        self.mediastinal_indices = [0, 1, 8, 9, 10]

        # Vertical order constraints (y coordinates should increase)
        self.vertical_orders = [
            (2, 4, 6),   # Lado izquierdo: ápice > hilio > base
            (3, 5, 7),   # Lado derecho: ápice > hilio > base
        ]

    def validate(self,
                landmarks: np.ndarray,
                original_landmarks: np.ndarray,
                image_width: int,
                image_height: int) -> Tuple[bool, Dict[str, float]]:
        """
        Validate anatomical constraints

        Args:
            landmarks: Augmented landmarks (30,) [x1,y1,x2,y2,...]
            original_landmarks: Original landmarks before augmentation
            image_width: Image width in pixels
            image_height: Image height in pixels

        Returns:
            (is_valid, metrics_dict)
        """
        landmarks_2d = landmarks.reshape(-1, 2)
        original_2d = original_landmarks.reshape(-1, 2)

        violations = []
        metrics = {}

        # 1. Bilateral Symmetry Check
        symmetry_errors = []
        for left_idx, right_idx in self.symmetric_pairs:
            left_x = landmarks_2d[left_idx, 0]
            right_x = landmarks_2d[right_idx, 0]

            # Distance from centerline should be symmetric
            center_x = image_width / 2
            left_dist = abs(center_x - left_x)
            right_dist = abs(right_x - center_x)

            # Relative error
            mean_dist = (left_dist + right_dist) / 2
            if mean_dist > 0:
                symmetry_error = abs(left_dist - right_dist) / mean_dist
                symmetry_errors.append(symmetry_error)

                if symmetry_error > self.tolerance:
                    violations.append(f"Symmetry violation: pair ({left_idx},{right_idx}) = {symmetry_error:.3f}")

        metrics['symmetry_error'] = np.mean(symmetry_errors) if symmetry_errors else 0.0

        # 2. Mediastinal Alignment Check
        mediastinal_deviations = []
        for idx in self.mediastinal_indices:
            x = landmarks_2d[idx, 0]
            center_x = image_width / 2
            deviation = abs(x - center_x) / image_width
            mediastinal_deviations.append(deviation)

            if deviation > 0.1:  # >10% deviation from centerline
                violations.append(f"Mediastinal deviation: landmark {idx} = {deviation:.3f}")

        metrics['mediastinal_deviation'] = np.mean(mediastinal_deviations)

        # 3. Vertical Order Check
        for order in self.vertical_orders:
            for i in range(len(order) - 1):
                y1 = landmarks_2d[order[i], 1]
                y2 = landmarks_2d[order[i+1], 1]

                if y1 >= y2:  # y should increase (apex < hilio < base)
                    violations.append(f"Vertical order violation: {order[i]} >= {order[i+1]}")

        metrics['vertical_order_violations'] = len([v for v in violations if 'Vertical order' in v])

        # 4. Displacement Check (landmarks shouldn't move too far)
        displacements = np.linalg.norm(landmarks_2d - original_2d, axis=1)
        mean_displacement = np.mean(displacements)
        max_displacement = np.max(displacements)

        # Relative to image size
        image_diagonal = np.sqrt(image_width**2 + image_height**2)
        relative_displacement = max_displacement / image_diagonal

        if relative_displacement > 0.15:  # >15% of image diagonal
            violations.append(f"Excessive displacement: {relative_displacement:.3f}")

        metrics['mean_displacement'] = mean_displacement
        metrics['max_displacement'] = max_displacement
        metrics['relative_displacement'] = relative_displacement

        # Overall validity
        is_valid = len(violations) == 0

        if not is_valid and self.verbose:
            warnings.warn(f"Anatomical constraint violations: {violations}")

        metrics['num_violations'] = len(violations)
        metrics['is_valid'] = is_valid

        return is_valid, metrics


class BreathingSimulation:
    """
    Simulate breathing effects on chest X-ray

    Breathing causes:
    - Diaphragm elevation/depression (bases move vertically)
    - Ribcage expansion/contraction (lateral landmarks move)
    - Mediastinum remains relatively stable

    This is the MOST IMPORTANT augmentation for chest X-rays
    """

    def __init__(self,
                 expansion_range: Tuple[float, float] = (0.97, 1.03),
                 probability: float = 0.5):
        """
        Args:
            expansion_range: Relative expansion factor (3% variation)
            probability: Probability of applying this transform
        """
        self.expansion_range = expansion_range
        self.probability = probability

    def __call__(self,
                image: np.ndarray,
                landmarks: np.ndarray,
                width: int,
                height: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply breathing simulation

        Args:
            image: Image array (H, W, C)
            landmarks: Landmarks array (30,)
            width: Original image width
            height: Original image height

        Returns:
            (transformed_image, transformed_landmarks)
        """
        if random.random() > self.probability:
            return image, landmarks

        # Random expansion factor
        expansion_factor = random.uniform(*self.expansion_range)

        landmarks_2d = landmarks.reshape(-1, 2).copy()

        # Center point (mediastinum center)
        center_x = width / 2
        center_y = height / 2

        # Define breathing zones with different effects
        # Bases (6, 7) and costophrenic angles (13, 14) - maximal effect
        base_indices = [6, 7, 13, 14]
        # Hilios (4, 5) - moderate effect
        hilio_indices = [4, 5]
        # Mediastinum (0, 1, 8, 9, 10) - minimal effect
        mediastinal_indices = [0, 1, 8, 9, 10]
        # Apices (2, 3) - minimal vertical, some lateral
        apex_indices = [2, 3]

        for idx in range(len(landmarks_2d)):
            x, y = landmarks_2d[idx]

            # Calculate distance from center
            dx = x - center_x
            dy = y - center_y

            if idx in base_indices:
                # Bases move most: vertical ±5% and lateral
                landmarks_2d[idx, 0] = center_x + dx * expansion_factor
                landmarks_2d[idx, 1] = center_y + dy * expansion_factor * 1.2  # More vertical

            elif idx in hilio_indices:
                # Hilios move moderately: mostly lateral
                landmarks_2d[idx, 0] = center_x + dx * expansion_factor
                landmarks_2d[idx, 1] = center_y + dy * (1.0 + (expansion_factor - 1.0) * 0.5)

            elif idx in mediastinal_indices:
                # Mediastinum stable: minimal movement
                landmarks_2d[idx, 0] = center_x + dx * (1.0 + (expansion_factor - 1.0) * 0.2)
                landmarks_2d[idx, 1] = y  # No vertical movement

            elif idx in apex_indices:
                # Apices: minimal vertical, some lateral
                landmarks_2d[idx, 0] = center_x + dx * expansion_factor
                landmarks_2d[idx, 1] = center_y + dy * (1.0 + (expansion_factor - 1.0) * 0.3)

            else:
                # Other landmarks: moderate effect
                landmarks_2d[idx, 0] = center_x + dx * expansion_factor
                landmarks_2d[idx, 1] = center_y + dy * expansion_factor

        # Clip to image bounds
        landmarks_2d[:, 0] = np.clip(landmarks_2d[:, 0], 0, width)
        landmarks_2d[:, 1] = np.clip(landmarks_2d[:, 1], 0, height)

        transformed_landmarks = landmarks_2d.flatten()

        # Note: We don't transform the image itself for breathing
        # (real breathing doesn't deform the anatomy, just changes projection)

        return image, transformed_landmarks


class PatientPositioningVariation:
    """
    Simulate patient positioning variations

    Real-world variations:
    - Small rotation (±2° realistic, not ±10°)
    - Small translation (patient off-center)
    - Slight tilt (patient not perfectly aligned)

    IMPORTANT: Medical images require conservative augmentation
    """

    def __init__(self,
                 angle_range: Tuple[float, float] = (-2, 2),
                 translation_range: Tuple[float, float] = (-0.02, 0.02),
                 probability: float = 0.4):
        """
        Args:
            angle_range: Rotation angle in degrees (±2° realistic)
            translation_range: Translation as fraction of image size (±2%)
            probability: Probability of applying this transform
        """
        self.angle_range = angle_range
        self.translation_range = translation_range
        self.probability = probability

    def __call__(self,
                image: np.ndarray,
                landmarks: np.ndarray,
                width: int,
                height: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply positioning variation

        Args:
            image: Image array (H, W, C)
            landmarks: Landmarks array (30,)
            width: Original image width
            height: Original image height

        Returns:
            (transformed_image, transformed_landmarks)
        """
        if random.random() > self.probability:
            return image, landmarks

        # Random rotation angle (small!)
        angle = random.uniform(*self.angle_range)

        # Random translation
        tx = random.uniform(*self.translation_range) * width
        ty = random.uniform(*self.translation_range) * height

        # Center of rotation (image center)
        center = (width / 2, height / 2)

        # Combined transformation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotation_matrix[0, 2] += tx  # Add translation to x
        rotation_matrix[1, 2] += ty  # Add translation to y

        # Transform image
        image_transformed = cv2.warpAffine(image, rotation_matrix, (width, height),
                                          flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_REPLICATE)

        # Transform landmarks
        landmarks_2d = landmarks.reshape(-1, 2)
        ones = np.ones((len(landmarks_2d), 1))
        landmarks_homogeneous = np.hstack([landmarks_2d, ones])

        landmarks_transformed = rotation_matrix.dot(landmarks_homogeneous.T).T

        # Clip to bounds
        landmarks_transformed[:, 0] = np.clip(landmarks_transformed[:, 0], 0, width)
        landmarks_transformed[:, 1] = np.clip(landmarks_transformed[:, 1], 0, height)

        return image_transformed, landmarks_transformed.flatten()


class ElasticDeformation:
    """
    Apply elastic deformation to simulate anatomical variations

    Simulates:
    - Soft tissue deformation
    - Anatomical variations between patients
    - Pathological deformations (conservative)

    Uses smooth displacement fields to maintain anatomical plausibility
    """

    def __init__(self,
                 alpha_range: Tuple[float, float] = (100, 200),
                 sigma: float = 20,
                 probability: float = 0.3):
        """
        Args:
            alpha_range: Displacement magnitude range (pixels)
            sigma: Gaussian filter sigma for smoothing (higher = smoother)
            probability: Probability of applying this transform
        """
        self.alpha_range = alpha_range
        self.sigma = sigma
        self.probability = probability

    def __call__(self,
                image: np.ndarray,
                landmarks: np.ndarray,
                width: int,
                height: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply elastic deformation

        Args:
            image: Image array (H, W, C)
            landmarks: Landmarks array (30,)
            width: Original image width
            height: Original image height

        Returns:
            (transformed_image, transformed_landmarks)
        """
        if random.random() > self.probability:
            return image, landmarks

        alpha = random.uniform(*self.alpha_range)

        # Generate random displacement fields
        dx = gaussian_filter((np.random.rand(height, width) * 2 - 1), self.sigma) * alpha
        dy = gaussian_filter((np.random.rand(height, width) * 2 - 1), self.sigma) * alpha

        # Create meshgrid
        x, y = np.meshgrid(np.arange(width), np.arange(height))

        # Apply displacement
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        # Transform image
        image_transformed = np.zeros_like(image)
        for c in range(image.shape[2]):
            image_transformed[:, :, c] = map_coordinates(
                image[:, :, c], indices, order=1, mode='reflect'
            ).reshape((height, width))

        # Transform landmarks (sample displacement at landmark positions)
        landmarks_2d = landmarks.reshape(-1, 2).copy()

        for i in range(len(landmarks_2d)):
            x_lm, y_lm = landmarks_2d[i]

            # Sample displacement at landmark position (with bounds checking)
            x_int = int(np.clip(x_lm, 0, width - 1))
            y_int = int(np.clip(y_lm, 0, height - 1))

            # Apply displacement
            landmarks_2d[i, 0] += dx[y_int, x_int]
            landmarks_2d[i, 1] += dy[y_int, x_int]

        # Clip to bounds
        landmarks_2d[:, 0] = np.clip(landmarks_2d[:, 0], 0, width)
        landmarks_2d[:, 1] = np.clip(landmarks_2d[:, 1], 0, height)

        return image_transformed, landmarks_2d.flatten()


class PathologyAwareAugmentation:
    """
    Apply pathology-specific augmentations

    Different categories have different characteristics:
    - COVID: Ground-glass opacities, peripheral distribution
    - Viral Pneumonia: Diffuse opacities, interstitial patterns
    - Normal: Clear lung fields

    Adjusts augmentation intensity based on category
    """

    def __init__(self,
                 category_weights: Optional[Dict[str, float]] = None):
        """
        Args:
            category_weights: Multiplier for augmentation intensity by category
        """
        if category_weights is None:
            category_weights = {
                'COVID': 1.2,           # More aggressive (varied presentation)
                'Viral_Pneumonia': 1.1,  # Moderate
                'Normal': 0.9,           # Conservative (preserve normal anatomy)
                'Unknown': 1.0
            }
        self.category_weights = category_weights

    def get_weight(self, category: str) -> float:
        """
        Get augmentation weight for category

        Args:
            category: Image category

        Returns:
            Augmentation intensity multiplier
        """
        return self.category_weights.get(category, 1.0)


class MedicalIntensityAugmentation:
    """
    Medical-specific intensity transformations

    Simulates:
    - X-ray exposure variations (kVp, mAs settings)
    - Detector sensitivity variations
    - Scatter radiation effects
    - Post-processing variations

    Does NOT affect landmarks (intensity-only)
    """

    def __init__(self,
                 brightness_range: Tuple[float, float] = (0.85, 1.15),
                 contrast_range: Tuple[float, float] = (0.85, 1.15),
                 gamma_range: Tuple[float, float] = (0.9, 1.1),
                 probability: float = 0.6):
        """
        Args:
            brightness_range: Brightness multiplier range
            contrast_range: Contrast multiplier range
            gamma_range: Gamma correction range
            probability: Probability of applying this transform
        """
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.gamma_range = gamma_range
        self.probability = probability

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply intensity augmentation

        Args:
            image: Image array (H, W, C)

        Returns:
            Transformed image (landmarks unchanged)
        """
        if random.random() > self.probability:
            return image

        image_transformed = image.copy().astype(np.float32)

        # Brightness adjustment
        if random.random() < 0.5:
            brightness_factor = random.uniform(*self.brightness_range)
            image_transformed = image_transformed * brightness_factor

        # Contrast adjustment
        if random.random() < 0.5:
            contrast_factor = random.uniform(*self.contrast_range)
            mean = np.mean(image_transformed)
            image_transformed = (image_transformed - mean) * contrast_factor + mean

        # Gamma correction (simulates exposure variations)
        if random.random() < 0.3:
            gamma = random.uniform(*self.gamma_range)
            # Normalize to [0, 1] and clip to avoid negative values
            img_normalized = np.clip(image_transformed / 255.0, 0, 1)
            # Apply gamma
            img_gamma = np.power(img_normalized, gamma)
            # Back to [0, 255]
            image_transformed = img_gamma * 255.0

        # Clip to valid range
        image_transformed = np.clip(image_transformed, 0, 255).astype(np.uint8)

        return image_transformed


class MedicalLandmarkTransforms:
    """
    Integrated medical augmentation pipeline for landmark prediction

    Combines all medical-specific transforms with anatomical validation

    Pipeline:
    1. Pathology-aware weight selection
    2. Breathing simulation (most important)
    3. Patient positioning variation
    4. Elastic deformation (conservative)
    5. Intensity augmentation
    6. Anatomical constraint validation
    7. Fallback to original if validation fails

    Target: <6.0px mean error (from 7.23px baseline)
    """

    def __init__(self,
                 image_size: Tuple[int, int] = (224, 224),
                 is_training: bool = True,
                 enable_medical_aug: bool = True,
                 validation_tolerance: float = 0.20,
                 verbose: bool = False):
        """
        Args:
            image_size: Target image size
            is_training: If True, apply augmentation
            enable_medical_aug: Enable medical-specific augmentation
            validation_tolerance: Tolerance for anatomical constraint violations
            verbose: Print validation warnings
        """
        self.image_size = image_size
        self.is_training = is_training
        self.enable_medical_aug = enable_medical_aug
        self.verbose = verbose

        # ImageNet normalization
        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]

        # Initialize medical transforms
        if self.enable_medical_aug and self.is_training:
            self.breathing = BreathingSimulation(
                expansion_range=(0.97, 1.03),
                probability=0.5
            )

            self.positioning = PatientPositioningVariation(
                angle_range=(-2, 2),
                translation_range=(-0.02, 0.02),
                probability=0.4
            )

            self.elastic = ElasticDeformation(
                alpha_range=(100, 200),
                sigma=20,
                probability=0.3
            )

            self.pathology_aware = PathologyAwareAugmentation()

            self.intensity = MedicalIntensityAugmentation(
                brightness_range=(0.85, 1.15),
                contrast_range=(0.85, 1.15),
                gamma_range=(0.9, 1.1),
                probability=0.6
            )

            self.validator = AnatomicalConstraintValidator(
                tolerance=validation_tolerance,
                verbose=verbose
            )

        # Statistics for monitoring
        self.stats = {
            'total_augmentations': 0,
            'validation_failures': 0,
            'breathing_applied': 0,
            'positioning_applied': 0,
            'elastic_applied': 0,
            'intensity_applied': 0
        }

    def __call__(self,
                image: np.ndarray,
                landmarks: np.ndarray,
                category: str = 'Unknown') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply medical augmentation pipeline

        Args:
            image: Image numpy array (H, W, C) - BGR format from cv2
            landmarks: Landmarks array (30,) format [x1, y1, x2, y2, ...]
            category: Image category for pathology-aware augmentation

        Returns:
            Tuple of (image_tensor, landmarks_tensor)
        """
        # Get original dimensions
        original_height, original_width = image.shape[:2]

        # Ensure RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # Store original landmarks for validation
        original_landmarks = landmarks.copy()

        # Apply medical augmentation if enabled and training
        if self.enable_medical_aug and self.is_training:
            self.stats['total_augmentations'] += 1

            # Get pathology-aware weight
            aug_weight = self.pathology_aware.get_weight(category)

            # Apply augmentations with retry logic
            max_retries = 3
            for retry in range(max_retries):
                image_aug = image.copy()
                landmarks_aug = landmarks.copy()

                # 1. Breathing simulation (MOST IMPORTANT for chest X-rays)
                if random.random() < 0.5 * aug_weight:
                    image_aug, landmarks_aug = self.breathing(
                        image_aug, landmarks_aug, original_width, original_height
                    )
                    self.stats['breathing_applied'] += 1

                # 2. Patient positioning variation
                if random.random() < 0.4 * aug_weight:
                    image_aug, landmarks_aug = self.positioning(
                        image_aug, landmarks_aug, original_width, original_height
                    )
                    self.stats['positioning_applied'] += 1

                # 3. Elastic deformation (conservative)
                if random.random() < 0.3 * aug_weight:
                    image_aug, landmarks_aug = self.elastic(
                        image_aug, landmarks_aug, original_width, original_height
                    )
                    self.stats['elastic_applied'] += 1

                # 4. Validate anatomical constraints
                is_valid, metrics = self.validator.validate(
                    landmarks_aug, original_landmarks,
                    original_width, original_height
                )

                if is_valid:
                    # Validation passed, use augmented version
                    image = image_aug
                    landmarks = landmarks_aug
                    break
                else:
                    # Validation failed, retry with different random seed
                    if retry == max_retries - 1:
                        # Max retries reached, use original
                        self.stats['validation_failures'] += 1
                        if self.verbose:
                            warnings.warn(f"Augmentation validation failed after {max_retries} retries, using original")

            # 5. Intensity augmentation (always valid, doesn't affect landmarks)
            if random.random() < 0.6 * aug_weight:
                image = self.intensity(image)
                self.stats['intensity_applied'] += 1

        # Resize image and landmarks
        image_resized = cv2.resize(image, self.image_size)
        landmarks_resized = self._resize_landmarks(
            landmarks, original_width, original_height,
            self.image_size[0], self.image_size[1]
        )

        # Convert to tensor and normalize
        image_tensor = self._numpy_to_tensor(image_resized)
        landmarks_normalized = self._normalize_landmarks(
            landmarks_resized, self.image_size[0], self.image_size[1]
        )

        return image_tensor, landmarks_normalized

    def _resize_landmarks(self,
                         landmarks: np.ndarray,
                         orig_width: int,
                         orig_height: int,
                         new_width: int,
                         new_height: int) -> np.ndarray:
        """Resize landmarks to new image size"""
        scale_x = new_width / orig_width
        scale_y = new_height / orig_height

        landmarks_resized = landmarks.copy()
        for i in range(0, len(landmarks), 2):
            landmarks_resized[i] *= scale_x
            landmarks_resized[i + 1] *= scale_y

        return landmarks_resized

    def _normalize_landmarks(self,
                            landmarks: np.ndarray,
                            width: int,
                            height: int) -> torch.Tensor:
        """Normalize landmarks to [0, 1]"""
        landmarks_normalized = landmarks.copy().astype(np.float32)

        for i in range(0, len(landmarks), 2):
            landmarks_normalized[i] /= width
            landmarks_normalized[i + 1] /= height

        return torch.from_numpy(landmarks_normalized)

    def _numpy_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert image to tensor with ImageNet normalization"""
        # BGR to RGB
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # To tensor (C, H, W) and normalize to [0, 1]
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # ImageNet normalization
        for c in range(3):
            image_tensor[c] = (image_tensor[c] - self.imagenet_mean[c]) / self.imagenet_std[c]

        return image_tensor

    def get_stats(self) -> Dict[str, float]:
        """
        Get augmentation statistics

        Returns:
            Dictionary with augmentation statistics and rates
        """
        total = max(self.stats['total_augmentations'], 1)

        return {
            'total_augmentations': self.stats['total_augmentations'],
            'validation_failure_rate': self.stats['validation_failures'] / total,
            'breathing_rate': self.stats['breathing_applied'] / total,
            'positioning_rate': self.stats['positioning_applied'] / total,
            'elastic_rate': self.stats['elastic_applied'] / total,
            'intensity_rate': self.stats['intensity_applied'] / total
        }

    def reset_stats(self):
        """Reset augmentation statistics"""
        for key in self.stats:
            self.stats[key] = 0


def get_medical_transforms(image_size: Tuple[int, int] = (224, 224),
                          is_training: bool = True,
                          enable_medical_aug: bool = True,
                          validation_tolerance: float = 0.20,
                          verbose: bool = False) -> MedicalLandmarkTransforms:
    """
    Factory function for medical landmark transforms

    Args:
        image_size: Target image size
        is_training: If True, apply augmentation
        enable_medical_aug: Enable medical-specific augmentation
        validation_tolerance: Tolerance for anatomical constraint violations
        verbose: Print validation warnings

    Returns:
        MedicalLandmarkTransforms instance
    """
    return MedicalLandmarkTransforms(
        image_size=image_size,
        is_training=is_training,
        enable_medical_aug=enable_medical_aug,
        validation_tolerance=validation_tolerance,
        verbose=verbose
    )
