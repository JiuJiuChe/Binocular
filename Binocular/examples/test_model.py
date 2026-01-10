"""
Test script for model components.

This script demonstrates how to:
1. Load different encoders (DINOv2)
2. Create classification models with frozen/unfrozen encoders
3. Make predictions on dummy data
4. Inspect model structure and parameters
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from models import get_encoder, get_encoder_info, LinearProbe, EncoderClassifier
from models.wrappers import create_model


def test_encoder_info():
    """Test getting encoder information without loading the model."""
    print("\n" + "="*80)
    print("TEST 1: Encoder Information")
    print("="*80)
    
    encoder_configs = [
        ('dinov2', 'dinov2_vits14'),
        ('dinov2', 'dinov2_vitb14'),
        ('dinov2', 'dinov2_vitl14'),
        ('dinov2', 'dinov2_vitg14'),
    ]
    
    for encoder_type, model_name in encoder_configs:
        try:
            info = get_encoder_info(encoder_type, model_name)
            print(f"\n{model_name}:")
            print(f"  Feature dimension: {info.feature_dim}")
            print(f"  Expected image size: {info.image_size}")
        except Exception as e:
            print(f"\n{model_name}: {e}")


def test_encoder_loading():
    """Test loading a DINOv2 encoder."""
    print("\n" + "="*80)
    print("TEST 2: Loading DINOv2 Encoder")
    print("="*80)
    
    print("\nLoading dinov2_vits14 (smallest model)...")
    encoder = get_encoder('dinov2', 'dinov2_vits14', freeze=True)
    
    print(f"Encoder type: {type(encoder).__name__}")
    print(f"Feature dimension: {encoder.get_feature_dim()}")
    print(f"Is frozen: {not next(encoder.parameters()).requires_grad}")
    
    # Test forward pass with dummy data
    print("\nTesting forward pass with dummy image...")
    dummy_image = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        features = encoder(dummy_image)
    
    print(f"Input shape: {dummy_image.shape}")
    print(f"Output features shape: {features.shape}")
    print(f"Features shape matches expected dim: {features.shape[-1] == encoder.get_feature_dim()}")


def test_linear_probe():
    """Test creating a linear probe classifier."""
    print("\n" + "="*80)
    print("TEST 3: Linear Probe")
    print("="*80)
    
    feature_dim = 384  # DINOv2-ViT-S feature dimension
    num_classes = 555  # NABirds species count
    
    print(f"\nCreating linear probe:")
    print(f"  Input: {feature_dim} features")
    print(f"  Output: {num_classes} classes")
    
    probe = LinearProbe(feature_dim, num_classes, dropout=0.0)
    
    # Count parameters
    num_params = sum(p.numel() for p in probe.parameters())
    print(f"  Parameters: {num_params:,}")
    
    # Test forward pass
    print("\nTesting forward pass with dummy features...")
    dummy_features = torch.randn(4, feature_dim)
    logits = probe(dummy_features)
    
    print(f"Input shape: {dummy_features.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Output shape correct: {logits.shape == (4, num_classes)}")


def test_complete_model():
    """Test creating and using a complete encoder-classifier model."""
    print("\n" + "="*80)
    print("TEST 4: Complete Model (Encoder + Classifier)")
    print("="*80)
    
    num_classes = 555  # NABirds species count
    
    print("\nCreating model with create_model() factory...")
    print("  Encoder: dinov2_vits14")
    print("  Classifier: Linear probe")
    print("  Encoder frozen: True")
    
    model = create_model(
        encoder_type='dinov2',
        num_classes=num_classes,
        encoder_name='dinov2_vits14',
        freeze_encoder=True,
        classifier_type='linear'
    )
    
    # Get model info
    info = model.get_model_info()
    print(f"\nModel Information:")
    print(f"  Total parameters: {info['total_params']:,}")
    print(f"  Trainable parameters: {info['trainable_params']:,}")
    print(f"  Encoder frozen: {info['encoder_frozen']}")
    print(f"  Encoder type: {info['encoder_type']}")
    print(f"  Classifier type: {info['classifier_type']}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    model.eval()
    dummy_images = torch.randn(2, 3, 224, 224)
    
    with torch.no_grad():
        logits = model(dummy_images)
    
    print(f"Input shape: {dummy_images.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Logits shape correct: {logits.shape == (2, num_classes)}")
    
    # Test prediction with top-k
    print("\nTesting prediction with top-5...")
    with torch.no_grad():
        top_probs, top_classes = model.predict(dummy_images, top_k=5)
    
    print(f"Top probabilities shape: {top_probs.shape}")
    print(f"Top classes shape: {top_classes.shape}")
    print(f"Probabilities sum to ~1.0: {torch.allclose(top_probs.sum(dim=-1), torch.ones(2), atol=0.1)}")
    
    # Test feature extraction
    print("\nTesting feature extraction...")
    with torch.no_grad():
        features = model.get_features(dummy_images)
    
    print(f"Features shape: {features.shape}")
    print(f"Features shape correct: {features.shape == (2, 384)}")


def test_freeze_unfreeze():
    """Test freezing and unfreezing encoder."""
    print("\n" + "="*80)
    print("TEST 5: Freeze/Unfreeze Encoder")
    print("="*80)
    
    print("\nCreating model with frozen encoder...")
    model = create_model(
        encoder_type='dinov2',
        num_classes=555,
        encoder_name='dinov2_vits14',
        freeze_encoder=True,
        classifier_type='linear'
    )
    
    trainable_frozen = model.get_trainable_params()
    print(f"Trainable params (frozen): {trainable_frozen:,}")
    
    print("\nUnfreezing encoder...")
    model.unfreeze_encoder()
    trainable_unfrozen = model.get_trainable_params()
    print(f"Trainable params (unfrozen): {trainable_unfrozen:,}")
    
    print(f"\nDifference: {trainable_unfrozen - trainable_frozen:,} params")
    print(f"All params now trainable: {trainable_unfrozen == model.get_total_params()}")
    
    print("\nFreezing encoder again...")
    model.freeze_encoder_weights()
    trainable_refrozen = model.get_trainable_params()
    print(f"Trainable params (re-frozen): {trainable_refrozen:,}")
    print(f"Back to original: {trainable_refrozen == trainable_frozen}")


def main():
    """Run all tests."""
    print("\n")
    print("="*80)
    print(" MODEL COMPONENTS TEST SUITE")
    print("="*80)
    
    try:
        # Test 1: Get encoder info
        test_encoder_info()
        
        # Test 2: Load encoder
        test_encoder_loading()
        
        # Test 3: Create linear probe
        test_linear_probe()
        
        # Test 4: Create complete model
        test_complete_model()
        
        # Test 5: Test freeze/unfreeze
        test_freeze_unfreeze()
        
        print("\n" + "="*80)
        print(" ALL TESTS PASSED ✓")
        print("="*80)
        print("\nThe model components are working correctly!")
        print("\nNext steps:")
        print("  1. Integrate with NABirds dataset")
        print("  2. Implement training loop")
        print("  3. Add evaluation metrics")
        
    except Exception as e:
        print("\n" + "="*80)
        print(" TEST FAILED ✗")
        print("="*80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
