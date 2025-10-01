"""
Test script to verify validation error calculation bug

The issue: In train_efficientnet_medical.py, the validation computes:
1. pred_pixels = predictions * 224
2. target_pixels = targets * 224
3. Computes Euclidean distance

But the TRAINING uses normalized coordinates (0-1) in the loss!

This creates a mismatch where:
- Training loss optimizes normalized coordinates (0-1 range)
- Validation error measures pixel coordinates (0-224 range)

The model might be learning to minimize normalized error, not pixel error!
"""

import torch
import numpy as np

print("="*80)
print("VALIDATION BUG INVESTIGATION")
print("="*80)

# Simulate predictions and targets
batch_size = 16
num_coords = 30

# Normalized coordinates (0-1) - what model actually outputs
pred_normalized = torch.rand(batch_size, num_coords) * 0.8 + 0.1  # Range [0.1, 0.9]
target_normalized = torch.rand(batch_size, num_coords) * 0.8 + 0.1

print("\n1. TRAINING LOSS CALCULATION:")
print("-" * 80)

# Wing Loss operates on normalized coordinates
wing_loss_input = torch.abs(pred_normalized - target_normalized)
print(f"Wing loss input range: {wing_loss_input.min():.6f} - {wing_loss_input.max():.6f}")
print(f"Wing loss input mean: {wing_loss_input.mean():.6f}")
print(f"This is NORMALIZED (0-1 range)")

print("\n2. VALIDATION ERROR CALCULATION (CURRENT - train_efficientnet_medical.py):")
print("-" * 80)

# Current validation: multiply by 224 THEN compute distance
pred_pixels = pred_normalized * 224
target_pixels = target_normalized * 224

pred_reshaped = pred_pixels.view(-1, 15, 2)
target_reshaped = target_pixels.view(-1, 15, 2)

errors_per_landmark = torch.sqrt(torch.sum((pred_reshaped - target_reshaped) ** 2, dim=2))
current_val_error = torch.mean(errors_per_landmark)

print(f"Pixel coordinate range: {pred_pixels.min():.2f} - {pred_pixels.max():.2f}")
print(f"Validation error (current): {current_val_error:.4f} px")

print("\n3. TRAINING ERROR CALCULATION (train_efficientnet_phases.py - compute_pixel_error):")
print("-" * 80)

# Phase 4 method: compute distance FIRST on normalized, then multiply by 224
pred_reshaped_norm = pred_normalized.view(-1, 15, 2)
target_reshaped_norm = target_normalized.view(-1, 15, 2)

distances_normalized = torch.norm(pred_reshaped_norm - target_reshaped_norm, dim=2)
pixel_distances = distances_normalized * 224
phase4_error = torch.mean(pixel_distances)

print(f"Normalized distances: {distances_normalized.min():.6f} - {distances_normalized.max():.6f}")
print(f"Pixel error (Phase 4 method): {phase4_error:.4f} px")

print("\n4. COMPARISON:")
print("-" * 80)
print(f"Current validation method: {current_val_error:.4f} px")
print(f"Phase 4 method:            {phase4_error:.4f} px")
print(f"Difference:                {abs(current_val_error - phase4_error):.4f} px")

# Test with small normalized error
print("\n5. EDGE CASE - Small normalized error (0.01 in normalized space):")
print("-" * 80)

pred_small = torch.ones(batch_size, num_coords) * 0.5
target_small = torch.ones(batch_size, num_coords) * 0.51  # 0.01 difference

# Method 1: Current validation (multiply first, then distance)
pred_pixels_small = pred_small * 224
target_pixels_small = target_small * 224
pred_pix_reshaped = pred_pixels_small.view(-1, 15, 2)
target_pix_reshaped = target_pixels_small.view(-1, 15, 2)
errors_small_current = torch.sqrt(torch.sum((pred_pix_reshaped - target_pix_reshaped) ** 2, dim=2))
current_small_error = torch.mean(errors_small_current)

# Method 2: Phase 4 method (distance first, then multiply)
pred_norm_reshaped = pred_small.view(-1, 15, 2)
target_norm_reshaped = target_small.view(-1, 15, 2)
distances_small_norm = torch.norm(pred_norm_reshaped - target_norm_reshaped, dim=2)
pixel_distances_small = distances_small_norm * 224
phase4_small_error = torch.mean(pixel_distances_small)

print(f"Normalized error per coordinate: 0.01")
print(f"Current validation method: {current_small_error:.4f} px")
print(f"Phase 4 method:            {phase4_small_error:.4f} px")
print(f"Difference:                {abs(current_small_error - phase4_small_error):.4f} px")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)
print("Both methods should mathematically produce the SAME result:")
print("  sqrt((x1-x2)² + (y1-y2)²) * 224 = sqrt((x1*224-x2*224)² + (y1*224-y2*224)²)")
print("\nLet me verify this is actually true...")

# Mathematical verification
x1, y1 = 0.5, 0.6
x2, y2 = 0.51, 0.61

method1 = np.sqrt((x1*224 - x2*224)**2 + (y1*224 - y2*224)**2)
method2 = np.sqrt((x1 - x2)**2 + (y1 - y2)**2) * 224

print(f"\nMethod 1 (multiply first):  {method1:.6f}")
print(f"Method 2 (distance first):  {method2:.6f}")
print(f"Are they equal? {np.isclose(method1, method2)}")

print("\n" + "="*80)
print("REAL BUG INVESTIGATION:")
print("="*80)

print("\nThe validation calculation looks correct mathematically.")
print("\nBUT - let's check if the loss is being computed correctly...")
print("\nPotential issues to investigate:")
print("1. Is the loss function receiving normalized or pixel coordinates?")
print("2. Are we averaging loss incorrectly?")
print("3. Is there a bug in the loss implementation itself?")
print("4. Could the model weights be frozen somehow?")
print("5. Is the learning rate too low?")
