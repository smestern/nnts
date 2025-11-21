"""
Training script for Electrophysiology Transformer
"""
import os
os.environ["CUADA_LAUNCH_BLOCKING"] = "1"  # For easier debugging
import sys
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from electrophysiology_transformer.model import ElectrophysiologyTransformer
from electrophysiology_transformer.tokenizer import ElectrophysiologyTokenizer
from dataset import ElectrophysiologyDataset


def detect_spikes(predictions, threshold=0.0):
    """Detect spikes in the predictions based on a threshold."""
    spike_indices = (predictions > threshold).nonzero(as_tuple=True)[1]
    return spike_indices


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        # Move to device
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
            static_real_features=static_real_features
        )
        
        loss = outputs.loss

        # #weird fix but we want to reward the model for predicting near spikes
        # true_spikes = (future_values > 0.0).any(axis=1)
        # #just enhanve the loss if true spikes are present
        # if true_spikes.any():
        #     loss = loss * (1.1 * true_spikes.float().mean().item())
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


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


def main(data_path="nn_ds_combined.joblib", checkpoint_path=None):
    # Configuration
    DATA_PATH = "nngan_trace_dataset_2000.joblib" if data_path is None else data_path
    CONTEXT_LENGTH = 512
    PREDICTION_LENGTH = 128
    MAX_LAG = 7
    BATCH_SIZE = 16 
    EPOCHS = 2000
    LEARNING_RATE = 1e-6
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CHECKPOINT = None #"best_val.pt"
    VAL_SPLIT = 0.1  # Fraction of windows reserved for validation (set to 0 to disable)
    BEST_CHECKPOINT_PATH = "best_val.pt" if checkpoint_path is None else checkpoint_path
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
        encoder_attention_heads=8,
        decoder_attention_heads=8,
        encoder_ffn_dim=512,
        decoder_ffn_dim=512,
        dropout=0.01,
        static_categorical_features=dataset.len_static_categorical_features,
        static_real_features=dataset.len_real_valued_features,
        scaling="std",
    )
    
    model = model.to(DEVICE)

    if CHECKPOINT is not None and os.path.isfile(CHECKPOINT):
        # Load checkpoint
        checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Resumed training from checkpoint: {CHECKPOINT}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    #LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf') if val_loader is not None else None

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
        scheduler.step()

        log_message = f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f}"

        val_loss = None
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, DEVICE)
            log_message += f" | Val Loss: {val_loss:.4f}"

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                }, BEST_CHECKPOINT_PATH)
                print(f"New best validation loss; checkpointed to {BEST_CHECKPOINT_PATH}")

        print(log_message)
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
    
    # Save final model
    final_path = "final_model.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete! Model saved to {final_path}")


if __name__ == "__main__":
    main()
