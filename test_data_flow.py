"""
Test script to trace data flow from dataset to transforms
Investigates why medical augmentation statistics show 0 augmentations applied
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import cv2
import torch
from src.data.medical_transforms import get_medical_transforms
from src.data.transforms import get_transforms

def test_transform_signature_detection():
    """Test the signature detection logic used in LandmarkDataset.__getitem__"""
    print("="*80)
    print("TEST 1: Transform Signature Detection")
    print("="*80)

    # Create both types of transforms
    medical_transform = get_medical_transforms(
        image_size=(224, 224),
        is_training=True,
        enable_medical_aug=True
    )

    basic_transform = get_transforms(
        image_size=(224, 224),
        is_training=True
    )

    # Test the detection logic (same as in dataset.py line 171)
    for name, transform in [('Medical', medical_transform), ('Basic', basic_transform)]:
        print(f"\n{name} Transform:")
        print(f"  Type: {type(transform).__name__}")
        print(f"  hasattr(__call__): {hasattr(transform, '__call__')}")

        if hasattr(transform, '__call__'):
            has_category = 'category' in transform.__call__.__code__.co_varnames
            print(f"  'category' in co_varnames: {has_category}")
            print(f"  co_varnames[:5]: {transform.__call__.__code__.co_varnames[:5]}")

            # This is the exact check from dataset.py line 171
            condition = hasattr(transform, '__call__') and 'category' in transform.__call__.__code__.co_varnames
            print(f"  Would pass dataset check: {condition}")

    print("\n✓ Signature detection test complete\n")


def test_direct_transform_call():
    """Test calling transforms directly with mock data"""
    print("="*80)
    print("TEST 2: Direct Transform Call")
    print("="*80)

    # Create mock data
    image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    landmarks = np.random.rand(30).astype(np.float32) * 512  # Random coordinates
    category = 'COVID'

    print(f"\nInput:")
    print(f"  Image shape: {image.shape}")
    print(f"  Landmarks shape: {landmarks.shape}")
    print(f"  Category: {category}")

    # Test medical transform with category
    medical_transform = get_medical_transforms(
        image_size=(224, 224),
        is_training=True,
        enable_medical_aug=True,
        verbose=True
    )

    print(f"\nCalling medical transform WITH category parameter...")
    try:
        image_tensor, landmarks_tensor = medical_transform(image, landmarks, category=category)
        print(f"✓ Success!")
        print(f"  Output image shape: {image_tensor.shape}")
        print(f"  Output landmarks shape: {landmarks_tensor.shape}")

        # Check statistics
        if hasattr(medical_transform, 'stats'):
            stats = medical_transform.stats
            print(f"\nMedical Transform Stats:")
            print(f"  Total augmentations: {stats['total_augmentations']}")
            print(f"  Breathing applied: {stats['breathing_applied']}")
            print(f"  Positioning applied: {stats['positioning_applied']}")
            print(f"  Elastic applied: {stats['elastic_applied']}")
            print(f"  Intensity applied: {stats['intensity_applied']}")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

    # Test medical transform WITHOUT category (should still work)
    print(f"\nCalling medical transform WITHOUT category parameter...")
    try:
        image_tensor, landmarks_tensor = medical_transform(image, landmarks)
        print(f"✓ Success (uses default category='Unknown')")
        print(f"  Output image shape: {image_tensor.shape}")
        print(f"  Output landmarks shape: {landmarks_tensor.shape}")
    except Exception as e:
        print(f"✗ Error: {e}")

    print("\n✓ Direct transform call test complete\n")


def test_dataset_loading():
    """Test actual dataset loading with medical transforms"""
    print("="*80)
    print("TEST 3: Dataset Loading Flow")
    print("="*80)

    from src.data.dataset import LandmarkDataset, create_data_splits
    import yaml

    # Load config to get data paths
    try:
        with open('configs/efficientnet_config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        annotations_file = config['data']['coordenadas_path']
        images_dir = config['data']['dataset_path']

        print(f"\nDataset paths:")
        print(f"  Annotations: {annotations_file}")
        print(f"  Images: {images_dir}")

        # Create splits
        train_indices, val_indices, test_indices = create_data_splits(
            annotations_file=annotations_file,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42
        )

        print(f"\n✓ Splits created: {len(train_indices)} train samples")

        # Create medical transform
        medical_transform = get_medical_transforms(
            image_size=(224, 224),
            is_training=True,
            enable_medical_aug=True,
            verbose=True
        )

        # Create dataset with medical transform
        print(f"\nCreating dataset with medical transform...")
        dataset = LandmarkDataset(
            annotations_file=annotations_file,
            images_dir=images_dir,
            transform=medical_transform,
            indices=train_indices[:10]  # Just first 10 samples for testing
        )

        print(f"✓ Dataset created: {len(dataset)} samples")

        # Test loading a few samples
        print(f"\nLoading samples to trigger augmentation...")
        for i in range(min(5, len(dataset))):
            try:
                image_tensor, landmarks_tensor, metadata = dataset[i]
                print(f"  Sample {i}: {metadata['filename']} ({metadata['category']})")
            except Exception as e:
                print(f"  ✗ Error loading sample {i}: {e}")

        # Check augmentation stats
        if hasattr(medical_transform, 'get_stats'):
            stats = medical_transform.get_stats()
            print(f"\nAugmentation Statistics after loading 5 samples:")
            print(f"  Total augmentations: {stats['total_augmentations']}")
            print(f"  Validation failure rate: {stats['validation_failure_rate']:.2%}")
            print(f"  Breathing rate: {stats['breathing_rate']:.2%}")
            print(f"  Positioning rate: {stats['positioning_rate']:.2%}")
            print(f"  Elastic rate: {stats['elastic_rate']:.2%}")
            print(f"  Intensity rate: {stats['intensity_rate']:.2%}")

            if stats['total_augmentations'] == 0:
                print("\n✗ CRITICAL: No augmentations were applied!")
                print("   This indicates the medical transform is not being called correctly")
            else:
                print(f"\n✓ SUCCESS: Medical augmentation pipeline is working!")

    except FileNotFoundError as e:
        print(f"✗ Error: Config or data files not found: {e}")
        print("   Skipping dataset loading test")
    except Exception as e:
        print(f"✗ Error in dataset loading: {e}")
        import traceback
        traceback.print_exc()

    print("\n✓ Dataset loading test complete\n")


def test_dataset_getitem_logic():
    """Test the specific __getitem__ logic with mock transform"""
    print("="*80)
    print("TEST 4: Dataset __getitem__ Logic Simulation")
    print("="*80)

    # Create mock sample data
    sample = {
        'image_path': 'mock/path.png',
        'landmarks': np.random.rand(30).astype(np.float32) * 512,
        'filename': 'COVID_001',
        'category': 'COVID'
    }

    # Create mock image
    image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    landmarks = sample['landmarks'].copy()

    # Create medical transform
    transform = get_medical_transforms(
        image_size=(224, 224),
        is_training=True,
        enable_medical_aug=True,
        verbose=False
    )

    print(f"Sample category: {sample['category']}")
    print(f"Transform type: {type(transform).__name__}")

    # Simulate the dataset __getitem__ logic (lines 170-174)
    print(f"\nSimulating dataset.__getitem__ logic:")

    if transform is not None:
        print(f"  transform is not None: True")

        # Check if transform supports category parameter (line 171)
        has_call = hasattr(transform, '__call__')
        print(f"  hasattr(transform, '__call__'): {has_call}")

        if has_call:
            has_category = 'category' in transform.__call__.__code__.co_varnames
            print(f"  'category' in co_varnames: {has_category}")

            if has_category:
                print(f"  → Calling with category parameter")
                image_tensor, landmarks_tensor = transform(image, landmarks, category=sample['category'])
                print(f"  ✓ Called successfully!")
            else:
                print(f"  → Calling without category parameter")
                image_tensor, landmarks_tensor = transform(image, landmarks)
                print(f"  ✓ Called successfully!")

        # Check stats
        if hasattr(transform, 'get_stats'):
            stats = transform.get_stats()
            print(f"\nTransform stats after 1 call:")
            print(f"  Total augmentations: {stats['total_augmentations']}")
            print(f"  Breathing applied: {stats['breathing_applied']}")
            print(f"  Positioning applied: {stats['positioning_applied']}")
            print(f"  Elastic applied: {stats['elastic_applied']}")
            print(f"  Intensity applied: {stats['intensity_applied']}")

    print("\n✓ __getitem__ logic simulation complete\n")


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("DATA FLOW INVESTIGATION: Dataset → Transforms → Medical Augmentation")
    print("="*80 + "\n")

    test_transform_signature_detection()
    test_direct_transform_call()
    test_dataset_getitem_logic()
    test_dataset_loading()

    print("="*80)
    print("ALL TESTS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
