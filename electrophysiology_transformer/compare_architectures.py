"""
Compare original vs enhanced multi-scale architecture performance

Visualizes how well each model captures:
1. Fast spikes (0-3ms)
2. Slow membrane dynamics (500ms)
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from electrophysiology_transformer.model import ElectrophysiologyTransformer
from electrophysiology_transformer.model_enhanced import EnhancedElectrophysiologyTransformer
from electrophysiology_transformer.tokenizer import ElectrophysiologyTokenizer
from dataset import ElectrophysiologyDataset


def analyze_timescale_errors(predictions, targets, sampling_rate=10000):
    """
    Separate error analysis for fast (spikes) and slow (membrane) components
    """
    # Convert to numpy
    pred_np = predictions.cpu().numpy()
    targ_np = targets.cpu().numpy()
    
    # Time axis
    time_ms = np.arange(pred_np.shape[1]) / sampling_rate * 1000
    
    # Identify spikes (fast events)
    spike_threshold = -20.0  # mV
    spike_mask = targ_np > spike_threshold
    
    # Fast error (during spikes)
    if spike_mask.any():
        fast_error = np.mean((pred_np[spike_mask] - targ_np[spike_mask]) ** 2)
        fast_mae = np.mean(np.abs(pred_np[spike_mask] - targ_np[spike_mask]))
    else:
        fast_error = 0.0
        fast_mae = 0.0
    
    # Slow error (subthreshold)
    membrane_mask = ~spike_mask
    if membrane_mask.any():
        slow_error = np.mean((pred_np[membrane_mask] - targ_np[membrane_mask]) ** 2)
        slow_mae = np.mean(np.abs(pred_np[membrane_mask] - targ_np[membrane_mask]))
    else:
        slow_error = 0.0
        slow_mae = 0.0
    
    # Frequency domain analysis (captures long-timescale patterns)
    pred_fft = np.fft.rfft(pred_np, axis=1)
    targ_fft = np.fft.rfft(targ_np, axis=1)
    freqs = np.fft.rfftfreq(pred_np.shape[1], 1.0 / sampling_rate)
    
    # Low frequency error (< 100 Hz, slow dynamics)
    low_freq_mask = freqs < 100
    low_freq_error = np.mean(np.abs(pred_fft[:, low_freq_mask] - targ_fft[:, low_freq_mask]) ** 2)
    
    # High frequency error (> 100 Hz, fast dynamics)
    high_freq_mask = freqs >= 100
    high_freq_error = np.mean(np.abs(pred_fft[:, high_freq_mask] - targ_fft[:, high_freq_mask]) ** 2)
    
    return {
        'fast_mse': float(fast_error),
        'fast_mae': float(fast_mae),
        'slow_mse': float(slow_error),
        'slow_mae': float(slow_mae),
        'low_freq_error': float(low_freq_error),
        'high_freq_error': float(high_freq_error),
        'spike_mask': spike_mask,
        'time_ms': time_ms,
    }


def visualize_comparison(
    original_pred,
    enhanced_pred,
    targets,
    sample_idx=0,
    save_path="model_comparison.png"
):
    """
    Create comprehensive comparison visualization
    """
    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    fig.suptitle("Original vs Enhanced Multi-Scale Architecture", fontsize=16, fontweight='bold')
    
    # Extract single sample
    orig_p = original_pred[sample_idx].cpu().numpy()
    enh_p = enhanced_pred[sample_idx].cpu().numpy()
    targ = targets[sample_idx].cpu().numpy()
    
    time_ms = np.arange(len(targ)) / 10.0  # 10kHz sampling
    
    # 1. Full trace comparison
    axes[0, 0].plot(time_ms, targ, 'k-', label='Ground Truth', alpha=0.7, linewidth=1.5)
    axes[0, 0].plot(time_ms, orig_p, 'b-', label='Original', alpha=0.6, linewidth=1)
    axes[0, 0].set_ylabel('Voltage (mV)')
    axes[0, 0].set_title('Original Model - Full Trace')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(time_ms, targ, 'k-', label='Ground Truth', alpha=0.7, linewidth=1.5)
    axes[0, 1].plot(time_ms, enh_p, 'r-', label='Enhanced', alpha=0.6, linewidth=1)
    axes[0, 1].set_ylabel('Voltage (mV)')
    axes[0, 1].set_title('Enhanced Multi-Scale Model - Full Trace')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 2. Zoom on spike (first 10ms)
    zoom_samples = min(100, len(targ))  # 10ms @ 10kHz
    axes[1, 0].plot(time_ms[:zoom_samples], targ[:zoom_samples], 'k-', label='Truth', linewidth=2)
    axes[1, 0].plot(time_ms[:zoom_samples], orig_p[:zoom_samples], 'b-', label='Original', linewidth=1.5)
    axes[1, 0].set_ylabel('Voltage (mV)')
    axes[1, 0].set_title('Fast Spike Detail (0-10ms)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(time_ms[:zoom_samples], targ[:zoom_samples], 'k-', label='Truth', linewidth=2)
    axes[1, 1].plot(time_ms[:zoom_samples], enh_p[:zoom_samples], 'r-', label='Enhanced', linewidth=1.5)
    axes[1, 1].set_ylabel('Voltage (mV)')
    axes[1, 1].set_title('Fast Spike Detail (0-10ms)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 3. Error over time
    orig_error = np.abs(orig_p - targ)
    enh_error = np.abs(enh_p - targ)
    
    axes[2, 0].plot(time_ms, orig_error, 'b-', alpha=0.6)
    axes[2, 0].fill_between(time_ms, 0, orig_error, alpha=0.3)
    axes[2, 0].set_ylabel('Absolute Error (mV)')
    axes[2, 0].set_xlabel('Time (ms)')
    axes[2, 0].set_title(f'Original Error (MAE: {orig_error.mean():.2f} mV)')
    axes[2, 0].grid(True, alpha=0.3)
    
    axes[2, 1].plot(time_ms, enh_error, 'r-', alpha=0.6)
    axes[2, 1].fill_between(time_ms, 0, enh_error, alpha=0.3)
    axes[2, 1].set_ylabel('Absolute Error (mV)')
    axes[2, 1].set_xlabel('Time (ms)')
    axes[2, 1].set_title(f'Enhanced Error (MAE: {enh_error.mean():.2f} mV)')
    axes[2, 1].grid(True, alpha=0.3)
    
    # 4. Frequency domain comparison
    targ_fft = np.abs(np.fft.rfft(targ))
    orig_fft = np.abs(np.fft.rfft(orig_p))
    enh_fft = np.abs(np.fft.rfft(enh_p))
    freqs = np.fft.rfftfreq(len(targ), 1.0 / 10000)
    
    # Only plot up to 500 Hz
    freq_mask = freqs <= 500
    
    axes[3, 0].semilogy(freqs[freq_mask], targ_fft[freq_mask], 'k-', label='Truth', alpha=0.7, linewidth=1.5)
    axes[3, 0].semilogy(freqs[freq_mask], orig_fft[freq_mask], 'b-', label='Original', alpha=0.6, linewidth=1)
    axes[3, 0].axvline(100, color='gray', linestyle='--', alpha=0.5, label='100 Hz')
    axes[3, 0].set_xlabel('Frequency (Hz)')
    axes[3, 0].set_ylabel('Amplitude (log)')
    axes[3, 0].set_title('Frequency Spectrum - Original')
    axes[3, 0].legend()
    axes[3, 0].grid(True, alpha=0.3)
    
    axes[3, 1].semilogy(freqs[freq_mask], targ_fft[freq_mask], 'k-', label='Truth', alpha=0.7, linewidth=1.5)
    axes[3, 1].semilogy(freqs[freq_mask], enh_fft[freq_mask], 'r-', label='Enhanced', alpha=0.6, linewidth=1)
    axes[3, 1].axvline(100, color='gray', linestyle='--', alpha=0.5, label='100 Hz')
    axes[3, 1].set_xlabel('Frequency (Hz)')
    axes[3, 1].set_ylabel('Amplitude (log)')
    axes[3, 1].set_title('Frequency Spectrum - Enhanced')
    axes[3, 1].legend()
    axes[3, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison saved to {save_path}")
    plt.close()


def main():
    """
    Load checkpoints and compare performance on test data
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=== Multi-Scale Architecture Comparison ===\n")
    
    # Load dataset
    tokenizer = ElectrophysiologyTokenizer(
        voltage_clip=(-120.0, 40.0),
        current_clip=(-4000.0, 4000.0),
        normalize=False,
    )
    
    dataset = ElectrophysiologyDataset(
        data_path="nngan_trace_dataset_2000.joblib",
        tokenizer=tokenizer,
        context_length=512,
        prediction_length=128,
        max_lag=7,
        data_length=10000,
        include_time_features=True,
        include_real_valued_features=True,
    )
    
    # Get test samples
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=False
    )
    batch = next(iter(test_loader))
    
    past_values = batch['past_values'].to(device)
    future_values = batch['future_values'].to(device).squeeze(-1)
    
    past_current = None
    if batch['past_time_features'].size(-1) > 1:
        past_current = batch['past_time_features'][:, :, 1:2].to(device)
    
    print("Test batch:")
    print(f"  Context: {past_values.shape}")
    print(f"  Targets: {future_values.shape}\n")
    
    # Try to load and evaluate original model
    original_pred = None
    orig_checkpoint_path = "new_ds_dec_17th.pt_epoch_155.pt_epoch_176.pt"  # Adjust path as needed
    
    if Path(orig_checkpoint_path).exists():
        print("Loading original model...")
        original_model = ElectrophysiologyTransformer(
            context_length=512,
            prediction_length=128,
            d_model=512,
            encoder_layers=8,
            decoder_layers=8,
            encoder_attention_heads=8,
            decoder_attention_heads=8,
            encoder_ffn_dim=512,
            decoder_ffn_dim=512,
            dropout=0.01,
            static_categorical_features=dataset.len_static_categorical_features,
            static_real_features=dataset.len_real_valued_features,
            scaling="std",
        ).to(device)
        
        try:
            checkpoint = torch.load(orig_checkpoint_path, map_location=device, weights_only=False)
            original_model.load_state_dict(checkpoint['model_state_dict'])
            original_model.eval()
            print("✓ Original model checkpoint loaded")
            with torch.no_grad():
                # Original model uses HF transformer interface
                outputs = original_model.generate(
                    past_values=past_values,
                    past_time_features=batch['past_time_features'].to(device),
                    future_time_features=batch['future_time_features'].to(device),
                    past_observed_mask=batch['past_observed_mask'].to(device),
                    static_categorical_features=batch['static_categorical_features'].to(device),
                    static_real_features=batch['static_real_features'].to(device),
                )
                original_pred = outputs.sequences.mean(dim=1)  # Average over samples
                print("✓ Original model loaded and evaluated\n")
        except Exception as e:
            print(f"⚠ Could not load original model: {e}\n")
    else:
        print(f"⚠ Original checkpoint not found at {orig_checkpoint_path}\n")
    
    # Evaluate enhanced model (even without checkpoint for architecture demo)
    print("Initializing enhanced model...")
    enhanced_model = EnhancedElectrophysiologyTransformer(
        context_length=512,
        prediction_length=128,
        d_model=256,
        num_layers=6,
        num_heads=8,
        window_size=64,
        global_stride=16,
        cutoff_freq=100.0,
    ).to(device)
    
    enhanced_checkpoint_path = "enhanced_model_checkpoint_epoch_100.pt"
    if Path(enhanced_checkpoint_path).exists():
        checkpoint = torch.load(enhanced_checkpoint_path, map_location=device, weights_only=False)
        enhanced_model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Enhanced model checkpoint loaded\n")
    else:
        print("⚠ No checkpoint found, using random initialization\n")
    
    enhanced_model.eval()
    with torch.no_grad():
        enhanced_pred = enhanced_model(past_values, past_current=past_current)
    
    # Analyze errors
    print("=== Timescale Error Analysis ===\n")
    
    if original_pred is not None:
        orig_errors = analyze_timescale_errors(original_pred, future_values)
        print("Original Model:")
        print(f"  Fast (spike) MSE: {orig_errors['fast_mse']:.4f}")
        print(f"  Slow (membrane) MSE: {orig_errors['slow_mse']:.4f}")
        print(f"  Low freq error: {orig_errors['low_freq_error']:.4f}")
        print(f"  High freq error: {orig_errors['high_freq_error']:.4f}\n")
    
    enh_errors = analyze_timescale_errors(enhanced_pred, future_values)
    print("Enhanced Multi-Scale Model:")
    print(f"  Fast (spike) MSE: {enh_errors['fast_mse']:.4f}")
    print(f"  Slow (membrane) MSE: {enh_errors['slow_mse']:.4f}")
    print(f"  Low freq error: {enh_errors['low_freq_error']:.4f}")
    print(f"  High freq error: {enh_errors['high_freq_error']:.4f}\n")
    
    # Visualize
    if original_pred is not None:
        visualize_comparison(
            original_pred,
            enhanced_pred,
            future_values,
            sample_idx=0,
            save_path="architecture_comparison.png"
        )
    else:
        print("Skipping visualization (original model not available)")
    
    print("\n=== Summary ===")
    print("The enhanced architecture addresses multi-scale challenges through:")
    print("  1. Multi-resolution temporal encoding (1x, 5x, 25x, 100x)")
    print("  2. Local + global hierarchical attention")
    print("  3. Explicit fast/slow frequency decomposition")
    print("  4. Separate loss terms for spikes vs membrane dynamics")


if __name__ == "__main__":
    main()
