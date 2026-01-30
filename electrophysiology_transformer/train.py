"""
Training script for Electrophysiology Transformer
"""
import os
os.environ["CUADA_LAUNCH_BLOCKING"] = "1"  # For easier debugging
import sys
import torch
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from tqdm import tqdm
import numpy as np
# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from electrophysiology_transformer.model import ElectrophysiologyTransformer
from electrophysiology_transformer.tokenizer import ElectrophysiologyTokenizer
from dataset import ElectrophysiologyDataset


def detect_spikes(predictions, threshold=0.0):
    """Detect spikes in the predictions based on a threshold."""
    spike_indices = (predictions > threshold).nonzero(as_tuple=True)[1]
    return spike_indices


def train_epoch(model, dataloader, optimizer, device, scaler=None, gradient_clip=1.0, accumulation_steps=1):
    """Train for one epoch with mixed precision and gradient accumulation."""
    model.train()
    total_loss = 0
    grad_norms = []
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
        # Move to device
        past_values = batch['past_values'].to(device)
        future_values = batch['future_values'].to(device)
        past_time_features = batch['past_time_features'].to(device)
        future_time_features = batch['future_time_features'].to(device)
        past_observed_mask = batch['past_observed_mask'].to(device)
        static_categorical_features = batch['static_categorical_features'].to(device)
        static_real_features = batch['static_real_features'].to(device)
        # Forward pass with mixed precision
        with autocast(enabled=(scaler is not None)):
            outputs = model(
                past_values=past_values,
                past_time_features=past_time_features,
                future_time_features=future_time_features,
                past_observed_mask=past_observed_mask,
                future_values=future_values,
                static_categorical_features=static_categorical_features,
                static_real_features=static_real_features
            )
            
            loss = outputs.loss / accumulation_steps  # Scale loss for gradient accumulation
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights with gradient accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            if scaler is not None:
                # Gradient clipping with mixed precision
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                grad_norms.append(grad_norm.item())
                scaler.step(optimizer)
                scaler.update()
            else:
                # Gradient clipping without mixed precision
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                grad_norms.append(grad_norm.item())
                optimizer.step()
            
            optimizer.zero_grad(set_to_none=True)
        
        total_loss += loss.item() * accumulation_steps
    
    avg_loss = total_loss / len(dataloader)
    avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0
    return avg_loss, avg_grad_norm


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate on validation loader."""
    model.eval()
    total_loss = 0

    for batch in dataloader:
        past_values = batch['past_values'].to(device)
        future_values = batch['future_values'].to(device)
        past_time_features = batch['past_time_features'].to(device)
        future_time_features = batch['future_time_features'].to(device)
        past_observed_mask = batch['past_observed_mask'].to(device)
        static_categorical_features = batch['static_categorical_features'].to(device)
        static_real_features = batch['static_real_features'].to(device)
        # Forward pass
        outputs = model(
            past_values=past_values,
            past_time_features=past_time_features,
            future_time_features=future_time_features,
            past_observed_mask=past_observed_mask,
            future_values=future_values,
            static_categorical_features=static_categorical_features,
            static_real_features=static_real_features,
        )

        total_loss += outputs.loss.item()

    return total_loss / max(len(dataloader), 1)


def main(data_path=None, checkpoint_path="C:\\Users\\SMest\\Dropbox\\nnGAN\\best_val.pt_epoch_39.pt_epoch_56.pt_epoch_92.pt", context_length=1024, prediction_length=128, 
         batch_size=16, epochs=200, learning_rate=1e-6, use_mixed_precision=True, gradient_clip=5.0, 
         accumulation_steps=4, early_stopping_patience=100, warmup_epochs=5, override_lr=True, 
         use_swa=True, swa_start_epoch=50, swa_lr=1e-6):
    # Configuration
    DATA_PATH = "nngan_trace_dataset_2000.joblib" if data_path is None else data_path
    CONTEXT_LENGTH = context_length
    PREDICTION_LENGTH = prediction_length
    MAX_LAG = 7
    BATCH_SIZE = batch_size
    EPOCHS = epochs
    LEARNING_RATE = learning_rate
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CHECKPOINT = "best_val.pt_epoch_39.pt_epoch_56.pt" if checkpoint_path is None else checkpoint_path
    VAL_SPLIT = 0.1  # Fraction of windows reserved for validation (set to 0 to disable)
    BEST_CHECKPOINT_PATH = "best_val.pt_epoch_39.pt_epoch_56.pt" if checkpoint_path is None else checkpoint_path
    print(f"Using device: {DEVICE}")
    
    # Create tokenizer
    tokenizer = ElectrophysiologyTokenizer(
        voltage_clip=(-120.0, 40.0),
        current_clip=(-4000.0, 4000.0),
        normalize=False,
    )
    
    # Create dataset and dataloader
    dataset = ElectrophysiologyDataset(
        data_path=DATA_PATH,
        tokenizer=tokenizer,
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        max_lag=MAX_LAG,
        data_length=10000,
        include_time_features=True,
        include_real_valued_features=True,
    )
    
    if VAL_SPLIT is not None and VAL_SPLIT > 0:
        val_size = int(len(dataset) * VAL_SPLIT)
    else:
        val_size = 0
    
    if val_size > 0:
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
    else:
        train_dataset = dataset
        val_dataset = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Use 0 for Windows compatibility
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
        )
    
    # Create model
    model = ElectrophysiologyTransformer(
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        d_model=512,
        encoder_layers=8,
        decoder_layers=8,
        encoder_attention_heads=16,
        decoder_attention_heads=16,
        encoder_ffn_dim=512,
        decoder_ffn_dim=512,
        dropout=0.01,
        static_categorical_features=dataset.len_static_categorical_features,
        static_real_features=dataset.len_real_valued_features,
        scaling="std",
    )
    
    model = model.to(DEVICE)

    # Determine if we're resuming or starting fresh
    resuming_training = CHECKPOINT is not None and os.path.isfile(CHECKPOINT)
    start_epoch = 0
    
    # Use fresh_start_lr if provided and not resuming, otherwise use learning_rate
    
    if not resuming_training and learning_rate is not None:
        effective_lr = learning_rate
        print(f"Starting fresh training with LR: {effective_lr:.2e}")
    elif resuming_training and override_lr and learning_rate is not None:
        effective_lr = learning_rate
        print(f"Resuming training with overridden LR: {effective_lr:.2e}")
    elif resuming_training:
        effective_lr = 1e-5  # Temporary placeholder
         # Will be updated after loading checkpoint
        print(f"Resuming training with checkpoint LR: {learning_rate:.2e}")
    
    # Optimizer with better defaults
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=effective_lr,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # # Learning rate scheduler with warmup
    # warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    #     optimizer,
    #     start_factor=1e-12,
    #     end_factor=0.999,
    #     total_iters=warmup_epochs,
    # )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs - warmup_epochs,
        eta_min=1e-12,
    )
    # scheduler = torch.optim.lr_scheduler.ChainedScheduler(
    #     [warmup_scheduler, 
    #      cosine_scheduler]
    # )
    
    # Stochastic Weight Averaging (SWA)
    swa_model = None
    swa_scheduler = None
    if use_swa:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)
        print(f"SWA enabled: will start at epoch {swa_start_epoch} with LR {swa_lr:.2e}")
    
    # Mixed precision scaler
    scaler = GradScaler() if use_mixed_precision and torch.cuda.is_available() else None
    if scaler:
        print("Using mixed precision training (AMP)")
    
    # Load checkpoint after optimizer/scheduler initialization
    if resuming_training:
        checkpoint = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore optimizer state if available (preserves momentum, LR, etc.)
        if 'optimizer_state_dict' in checkpoint and not override_lr:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("✓ Restored optimizer state")
            except Exception as e:
                print(f"⚠ Could not restore optimizer state: {e}")
        elif override_lr:
            print(f"⚠ Skipping optimizer state restoration (override_lr=True)")
        
        # Override learning rate if requested
        if override_lr and learning_rate is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
            print(f"✓ Overridden LR to: {learning_rate:.2e}")
        
        # Restore scheduler state if available
        if 'scheduler_state_dict' in checkpoint and not override_lr:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("✓ Restored scheduler state")
            except Exception as e:
                print(f"⚠ Could not restore scheduler state: {e}")
        
        # Restore scaler state if available
        if scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
            try:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                print("✓ Restored mixed precision scaler state")
            except Exception as e:
                print(f"⚠ Could not restore scaler state: {e}")
        
        # Get the starting epoch
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"✓ Resumed from checkpoint: {CHECKPOINT} (epoch {start_epoch})")
    else:
        print("No checkpoint found, training from scratch.")

    # Training loop
    print("\nStarting training...")
    print(f"Initial LR: {optimizer.param_groups[0]['lr']:.2e}, Warmup: {warmup_epochs} epochs, Gradient Clip: {gradient_clip}")
    print(f"Batch Size: {batch_size}, Accumulation Steps: {accumulation_steps}, Effective Batch: {batch_size * accumulation_steps}")
    print(f"Training from epoch {start_epoch + 1} to {epochs}")
    
    best_val_loss = float('inf') if val_loader is not None else None
    patience_counter = 0
    training_history = {'train_loss': [], 'val_loss': [], 'grad_norm': [], 'lr': []}

    for epoch in range(start_epoch, epochs):
        train_loss, grad_norm = train_epoch(
            model, train_loader, optimizer, DEVICE, 
            scaler=scaler, gradient_clip=gradient_clip, 
            accumulation_steps=accumulation_steps
        )
        
        current_lr = optimizer.param_groups[0]['lr']
        training_history['train_loss'].append(train_loss)
        training_history['grad_norm'].append(grad_norm)
        training_history['lr'].append(current_lr)

        log_message = f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} | Grad Norm: {grad_norm:.3f} | LR: {current_lr:.2e}"

        val_loss = None
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, DEVICE)
            training_history['val_loss'].append(val_loss)
            log_message += f" | Val Loss: {val_loss:.4f}"

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict() if scaler else None,
                    'loss': val_loss,
                    'training_history': training_history,
                }, f"{BEST_CHECKPOINT_PATH}_epoch_{epoch+1}.pt")
                print(f"✓ New best validation loss: {val_loss:.4f} (improved by {(1 - val_loss/best_val_loss)*100:.2f}%) @ epoch {epoch+1} with LR: {current_lr:.2e}")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\n⚠ Early stopping triggered after {patience_counter} epochs without improvement")
                    break

        print(log_message)
        
        # Update learning rate (use SWA scheduler after swa_start_epoch)
        if use_swa and epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % 50 == 0:
            checkpoint_path = f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            # save tokenizer config
            #tokenizer.save_pretrained(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Finalize SWA model
    if use_swa and swa_model is not None:
        print("\nFinalizing SWA model...")
        # Update batch normalization statistics for SWA model
        update_bn(train_loader, swa_model, device=DEVICE)
        
        # Save SWA model
        swa_path = "final_model_swa.pt"
        torch.save(swa_model.module.state_dict(), swa_path)
        print(f"SWA model saved to {swa_path}")
        
        # Evaluate SWA model if validation set exists
        if val_loader is not None:
            swa_val_loss = evaluate(swa_model, val_loader, DEVICE)
            print(f"SWA validation loss: {swa_val_loss:.4f}")
    
    # Save final model
    final_path = "final_model.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete! Model saved to {final_path}")


if __name__ == "__main__":
    main()
