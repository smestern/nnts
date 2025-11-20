"""
Test script for Electrophysiology Transformer
Tests basic functionality before full training.
"""
import os
import sys
import joblib
import numpy as np
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from electrophysiology_transformer.model import ElectrophysiologyTransformer
from electrophysiology_transformer.tokenizer import ElectrophysiologyTokenizer




def test_model():
    """Test the model."""
    print("Testing Model...")
    
    CONTEXT_LENGTH = 128
    PREDICTION_LENGTH = 32
    MAX_LAG = 7
    BATCH_SIZE = 4
    
    model = ElectrophysiologyTransformer(
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        d_model=64,  # Smaller for testing
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=256,
        decoder_ffn_dim=256,
        dropout=0.1,
    )
    
    # Create dummy input
    past_length = CONTEXT_LENGTH + MAX_LAG
    past_values = torch.randn(BATCH_SIZE, past_length, 2)
    future_values = torch.randn(BATCH_SIZE, PREDICTION_LENGTH, 2)
    past_time_features = torch.randn(BATCH_SIZE, past_length, 1)
    future_time_features = torch.randn(BATCH_SIZE, PREDICTION_LENGTH, 1)
    past_observed_mask = torch.ones(BATCH_SIZE, past_length, 2)
    
    print(f"  Input shapes:")
    print(f"    past_values: {past_values.shape}")
    print(f"    future_values: {future_values.shape}")
    print(f"    past_time_features: {past_time_features.shape}")
    print(f"    future_time_features: {future_time_features.shape}")
    
    # Forward pass
    outputs = model(
        past_values=past_values,
        past_time_features=past_time_features,
        future_time_features=future_time_features,
        past_observed_mask=past_observed_mask,
        future_values=future_values,
    )
    
    print(f"  Output loss: {outputs.loss.item():.4f}")
    print(f"  Output shape: {outputs.encoder_last_hidden_state.shape}")
    print("  ✓ Model forward pass working!\n")
    
    # Test prediction
    print("Testing Prediction...")
    model.eval()
    with torch.no_grad():
        predictions = model.predict(
            past_values=past_values,
            past_time_features=past_time_features,
            future_time_features=future_time_features,
        )
    print(f"  Prediction encoder output shape: {predictions.encoder_last_hidden_state.shape}")
    print("  ✓ Model prediction working!\n")


def test_data_loading():
    """Test loading real data."""
    print("Testing Data Loading...")
    
    data_path = "../nn_ds_combined.joblib"
    if not os.path.exists(data_path):
        print(f"  ⚠ Data file not found: {data_path}")
        return
    
    data = joblib.load(data_path)
    print(f"  Data keys: {list(data.keys())}")
    print(f"  Number of trials: {len(data['voltages'])}")
    print(f"  Sweeps per trial: {len(data['voltages'][0])}")
    
    # Check first trial
    voltage_trial = data['voltages'][0]
    current_trial = data['currents'][0]
    
    print(f"  First sweep length (voltage): {len(voltage_trial[0])}")
    print(f"  First sweep length (current): {len(current_trial[0])}")
    print(f"  Voltage range: [{np.min([np.min(v) for v in voltage_trial]):.2f}, "
          f"{np.max([np.max(v) for v in voltage_trial]):.2f}]")
    print(f"  Current range: [{np.min([np.min(c) for c in current_trial]):.2f}, "
          f"{np.max([np.max(c) for c in current_trial]):.2f}]")
    print("  ✓ Data loading working!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Electrophysiology Transformer Test Suite")
    print("=" * 60 + "\n")
    
    test_data_loading()
    test_model()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
