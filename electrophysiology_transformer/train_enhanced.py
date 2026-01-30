"""
Training script for Enhanced Multi-Scale Electrophysiology Transformer

This script trains the enhanced model with multi-scale encoding specifically
designed to capture both fast spikes (0-3ms) and slow membrane dynamics (500ms).
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from electrophysiology_transformer.model_enhanced import EnhancedElectrophysiologyTransformer
from electrophysiology_transformer.tokenizer import ElectrophysiologyTokenizer
from dataset import ElectrophysiologyDataset


class MultiScaleLoss(nn.Module):
    """
    Loss function that separately evaluates fast and slow timescale predictions
    """
    
    def __init__(
        self,
        spike_weight: float = 0.1,
        membrane_weight: float = 1.0,
        spike_threshold: float = -20.0,  # mV
        frequency_weight: float = 0.1,  # Reduced from 0.5
    ):
        super().__init__()
        self.spike_weight = spike_weight
        self.membrane_weight = membrane_weight
        self.spike_threshold = spike_threshold
        self.frequency_weight = frequency_weight
        self.mse = nn.MSELoss()
        
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        sampling_rate: float = 10000.0,
    ) -> dict:
        """
        Args:
            predictions: [batch, length]
            targets: [batch, length]
            sampling_rate: Hz
        
        Returns:
            dict with total loss and individual components
        """
        # Basic MSE loss
        mse_loss = self.mse(predictions, targets)
        
        # Spike-specific loss (high-frequency events)
        spike_mask = targets > self.spike_threshold
        if spike_mask.any():
            spike_loss = self.mse(
                predictions[spike_mask],
                targets[spike_mask]
            )
        else:
            spike_loss = torch.tensor(0.0, device=predictions.device)
        
        # Membrane dynamics loss (exclude spikes)
        membrane_mask = ~spike_mask
        if membrane_mask.any():
            membrane_loss = self.mse(
                predictions[membrane_mask],
                targets[membrane_mask]
            )
        else:
            membrane_loss = torch.tensor(0.0, device=predictions.device)
        
        # Frequency domain loss (captures slow oscillations)
        pred_fft = torch.fft.rfft(predictions, dim=1)
        target_fft = torch.fft.rfft(targets, dim=1)
        
        # Focus on low frequencies (< 100 Hz) for membrane dynamics
        freqs = torch.fft.rfftfreq(targets.size(1), 1.0 / sampling_rate).to(predictions.device)
        low_freq_mask = freqs < 100.0
        
        if low_freq_mask.any():
            # Use magnitude and normalize by number of frequency bins
            freq_loss = torch.mean(
                torch.abs(pred_fft[:, low_freq_mask] - target_fft[:, low_freq_mask]) ** 2
            ) / (low_freq_mask.sum() * targets.size(1))  # Normalize by bins and sequence length
        else:
            freq_loss = torch.tensor(0.0, device=predictions.device)
        
        # Combine losses
        total_loss = (
            mse_loss +
            self.spike_weight * spike_loss +
            self.membrane_weight * membrane_loss +
            self.frequency_weight * freq_loss
        )
        
        return {
            'loss': total_loss,
            'mse': mse_loss.item(),
            'spike_loss': spike_loss.item(),
            'membrane_loss': membrane_loss.item(),
            'freq_loss': freq_loss.item(),
        }


def train_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    scaler=None,
    gradient_clip=1.0,
    accumulation_steps=1,
):
    """Train for one epoch with multi-scale loss"""
    model.train()
    total_losses = {
        'loss': 0,
        'mse': 0,
        'spike_loss': 0,
        'membrane_loss': 0,
        'freq_loss': 0,
    }
    grad_norms = []
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
        # Move to device
        past_values = batch['past_values'].to(device)
        future_values = batch['future_values'].to(device).squeeze(-1)  # [B, pred_len]
        
        # Extract current if available (from time features)
        past_current = None
        if 'past_time_features' in batch and batch['past_time_features'].size(-1) > 1:
            past_current = batch['past_time_features'][:, :, 1:2].to(device)  # [B, L, 1]
        
        # Forward pass with mixed precision
        with autocast(enabled=(scaler is not None)):
            predictions = model(past_values, past_current=past_current)
            loss_dict = criterion(predictions, future_values)
            loss = loss_dict['loss'] / accumulation_steps
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights with gradient accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                grad_norms.append(grad_norm.item())
                scaler.step(optimizer)
                scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                grad_norms.append(grad_norm.item())
                optimizer.step()
            
            optimizer.zero_grad(set_to_none=True)
        
        # Accumulate losses
        for key in total_losses.keys():
            if key in loss_dict:
                total_losses[key] += loss_dict[key] * accumulation_steps
            else:
                total_losses[key] += loss_dict['loss'].item() * accumulation_steps
    
    # Average losses
    num_batches = len(dataloader)
    for key in total_losses.keys():
        total_losses[key] /= num_batches
    
    avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0
    
    return total_losses, avg_grad_norm


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate on validation set"""
    model.eval()
    total_losses = {
        'loss': 0,
        'mse': 0,
        'spike_loss': 0,
        'membrane_loss': 0,
        'freq_loss': 0,
    }
    
    for batch in dataloader:
        past_values = batch['past_values'].to(device)
        future_values = batch['future_values'].to(device).squeeze(-1)
        
        past_current = None
        if 'past_time_features' in batch and batch['past_time_features'].size(-1) > 1:
            past_current = batch['past_time_features'][:, :, 1:2].to(device)
        
        predictions = model(past_values, past_current=past_current)
        loss_dict = criterion(predictions, future_values)
        
        for key in total_losses.keys():
            if key in loss_dict:
                total_losses[key] += loss_dict[key]
            else:
                total_losses[key] += loss_dict['loss'].item()
    
    # Average losses
    num_batches = max(len(dataloader), 1)
    for key in total_losses.keys():
        total_losses[key] /= num_batches
    
    return total_losses


def main(
    data_path=None,
    checkpoint_path=None,
    context_length=512,
    prediction_length=128,
    batch_size=16,
    epochs=200,
    learning_rate=3e-4,
    d_model=256,
    num_layers=6,
    num_heads=8,
    window_size=64,
    global_stride=16,
    cutoff_freq=100.0,
    spike_weight=2.0,
    membrane_weight=1.0,
    use_mixed_precision=True,
    gradient_clip=1.0,
    accumulation_steps=2,
    early_stopping_patience=60,
    warmup_epochs=5,
    fresh_start_lr=None,
):
    # Configuration
    DATA_PATH = "nngan_trace_dataset_2000.joblib" if data_path is None else data_path
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    VAL_SPLIT = 0.1
    
    print(f"Using device: {DEVICE}")
    print(f"\n=== Multi-Scale Architecture Configuration ===")
    print(f"Context: {context_length} samples ({context_length/10:.1f}ms @ 10kHz)")
    print(f"Prediction: {prediction_length} samples ({prediction_length/10:.1f}ms)")
    print(f"Window size: {window_size} samples ({window_size/10:.1f}ms)")
    print(f"Global stride: {global_stride} samples ({global_stride/10:.1f}ms)")
    print(f"Cutoff frequency: {cutoff_freq} Hz")
    print(f"Layers: {num_layers}, Heads: {num_heads}, d_model: {d_model}")
    
    # Create tokenizer
    tokenizer = ElectrophysiologyTokenizer(
        voltage_clip=(-120.0, 40.0),
        current_clip=(-4000.0, 4000.0),
        normalize=False,
    )
    
    # Create dataset
    dataset = ElectrophysiologyDataset(
        data_path=DATA_PATH,
        tokenizer=tokenizer,
        context_length=context_length,
        prediction_length=prediction_length,
        max_lag=7,
        data_length=10000,
        include_time_features=True,
        include_real_valued_features=True,
    )
    
    # Train/val split
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
    
    # Create enhanced model
    model = EnhancedElectrophysiologyTransformer(
        context_length=context_length,
        prediction_length=prediction_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        window_size=window_size,
        global_stride=global_stride,
        dropout=0.1,
        cutoff_freq=cutoff_freq,
    )
    
    model = model.to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Multi-scale loss
    criterion = MultiScaleLoss(
        spike_weight=spike_weight,
        membrane_weight=membrane_weight,
        frequency_weight=0.001,  # Lower weight for frequency loss
    )
    
    # Determine learning rate
    resuming_training = checkpoint_path is not None and os.path.isfile(checkpoint_path)
    start_epoch = 0
    effective_lr = learning_rate
    
    if not resuming_training and fresh_start_lr is not None:
        effective_lr = fresh_start_lr
        print(f"\nStarting fresh training with LR: {effective_lr:.2e}")
    elif resuming_training:
        print(f"\nResuming training with checkpoint LR: {learning_rate:.2e}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=effective_lr,
        weight_decay=0.01,
        betas=(0.9, 0.999),
    )
    
    # LR scheduler with warmup
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Mixed precision
    scaler = GradScaler() if use_mixed_precision and torch.cuda.is_available() else None
    if scaler:
        print("Using mixed precision training (AMP)")
    
    # Load checkpoint if resuming
    if resuming_training:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("✓ Restored optimizer state")
            except Exception as e:
                print(f"⚠ Could not restore optimizer: {e}")
        
        if 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("✓ Restored scheduler state")
            except Exception as e:
                print(f"⚠ Could not restore scheduler: {e}")
        
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"✓ Resumed from checkpoint (epoch {start_epoch})")
    else:
        print("Training from scratch")
    
    # Training loop
    print(f"\n=== Starting Training ===")
    print(f"Epochs: {start_epoch + 1} to {epochs}")
    print(f"Effective batch size: {batch_size * accumulation_steps}")
    
    best_val_loss = float('inf')
    patience_counter = 0
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'spike_loss': [],
        'membrane_loss': [],
        'grad_norm': [],
        'lr': [],
    }
    
    for epoch in range(start_epoch, epochs):
        # Train
        train_losses, grad_norm = train_epoch(
            model, train_loader, optimizer, criterion, DEVICE,
            scaler=scaler, gradient_clip=gradient_clip,
            accumulation_steps=accumulation_steps,
        )
        
        # Validate
        val_losses = evaluate(model, val_loader, criterion, DEVICE)
        
        # Log
        current_lr = optimizer.param_groups[0]['lr']
        training_history['train_loss'].append(train_losses['loss'])
        training_history['val_loss'].append(val_losses['loss'])
        training_history['spike_loss'].append(val_losses['spike_loss'])
        training_history['membrane_loss'].append(val_losses['membrane_loss'])
        training_history['grad_norm'].append(grad_norm)
        training_history['lr'].append(current_lr)
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_losses['loss']:.4f} | Val Loss: {val_losses['loss']:.4f}")
        print(f"  Spike: {val_losses['spike_loss']:.4f} | Membrane: {val_losses['membrane_loss']:.4f} | Freq: {val_losses['freq_loss']:.4f}")
        print(f"  Grad Norm: {grad_norm:.3f} | LR: {current_lr:.2e}")
        
        # Save best
        if val_losses['loss'] < best_val_loss:
            improvement = (best_val_loss - val_losses['loss']) / best_val_loss * 100
            best_val_loss = val_losses['loss']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'loss': val_losses['loss'],
                'training_history': training_history,
            }, f"enhanced_model_best_epoch_{epoch+1}.pt")
            
            print(f"  ✓ New best model saved (improved by {improvement:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"\n⚠ Early stopping triggered (no improvement for {patience_counter} epochs)")
                break
        
        scheduler.step()
        
        # Periodic checkpoints
        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'training_history': training_history,
            }, f"enhanced_model_checkpoint_epoch_{epoch+1}.pt")
            print(f"  ✓ Checkpoint saved")
    
    print("\n✓ Training complete!")
    return model, training_history


if __name__ == "__main__":
    model, history = main(
        context_length=512,
        prediction_length=128,
        batch_size=16,
        epochs=200,
        learning_rate=3e-4,  # Higher LR for fresh start
        d_model=256,
        num_layers=6,
        num_heads=8,
        window_size=64,  # ~6.4ms for spike capture
        global_stride=16,  # ~1.6ms global sampling
        cutoff_freq=100.0,  # 100Hz fast/slow boundary
        spike_weight=0.01,  # Emphasize spike accuracy
        membrane_weight=1.0,
        use_mixed_precision=True,
        gradient_clip=1.0,
        accumulation_steps=2,
        early_stopping_patience=100,
    )
