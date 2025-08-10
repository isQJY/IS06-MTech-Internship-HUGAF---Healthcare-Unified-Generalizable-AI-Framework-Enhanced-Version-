import torch
import numpy as np

# Add project path
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from transformers_model import MultimodalTransformer

def test_transformer_model_debug():
    """Debug Transformer model"""
    print("Debugging MultimodalTransformer model...")
    
    # Define modality dimensions
    modality_dims = {
        'demo': 10,
        'ts_ce': 50,
        'ts_le': 50,
        'ts_pe': 20,
        'vd': 1024,
        'vp': 18,
        'vmd': 1024,
        'vmp': 18,
        'n_ecg': 768,
        'n_ech': 768,
        'n_rad': 768,
        'genomic': 100,
        'wearable': 50,
        'pathology': 256
    }
    
    # Create model
    model = MultimodalTransformer(modality_dims, hidden_dim=256, num_heads=4, num_layers=3)
    
    # Create sample inputs - ensure each modality is a 1D vector
    batch_size = 2
    modality_inputs = {
        'demo': torch.randn(modality_dims['demo']),  # 1D
        'ts_ce': torch.randn(modality_dims['ts_ce']),  # 1D
        'ts_le': torch.randn(modality_dims['ts_le']),  # 1D
        'ts_pe': torch.randn(modality_dims['ts_pe']),  # 1D
        'vd': torch.randn(modality_dims['vd']),  # 1D
        'vp': torch.randn(modality_dims['vp']),  # 1D
        'vmd': torch.randn(modality_dims['vmd']),  # 1D
        'vmp': torch.randn(modality_dims['vmp']),  # 1D
        'n_ecg': torch.randn(modality_dims['n_ecg']),  # 1D
        'n_ech': torch.randn(modality_dims['n_ech']),  # 1D
        'n_rad': torch.randn(modality_dims['n_rad']),  # 1D
        'genomic': torch.randn(modality_dims['genomic']),  # 1D
        'wearable': torch.randn(modality_dims['wearable']),  # 1D
        'pathology': torch.randn(modality_dims['pathology'])  # 1D
    }
    
    # Print each modality shape
    print("Input modalities shapes:")
    for name, tensor in modality_inputs.items():
        print(f"  {name}: {tensor.shape}")
    
    # Test forward propagation
    try:
        output = model(modality_inputs)
        print(f"Model output shape: {output.shape}")
        print("MultimodalTransformer model test passed!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_transformer_model_debug()
