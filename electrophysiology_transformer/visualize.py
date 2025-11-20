"""
Inference and visualization script for trained models
"""
import os
import sys
import joblib
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from electrophysiology_transformer import tokenizer
from electrophysiology_transformer.model import ElectrophysiologyTransformer
from electrophysiology_transformer.tokenizer import ElectrophysiologyTokenizer
from dataset import ElectrophysiologyDataset

def load_model(checkpoint_path, device='cpu'):
    """Load a trained model from checkpoint."""
    # Create model with same config as training
    model = ElectrophysiologyTransformer(
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
        scaling="std",)
    
    # Load weights
    if checkpoint_path.endswith('.pt'):
        if 'final_model' in checkpoint_path:
            # Direct state dict
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            # Checkpoint format
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from epoch {checkpoint['epoch']+1}")
    
    model = model.to(device)
    model.eval()
    return model


def visualize_prediction(
    model,
    tokenizer,
    voltage_data,
    current_data,
    cell_id=None,
    context_length=256,
    prediction_length=32,
    max_lag=7,
    device='cpu',
):
    """
    Make a prediction and visualize it.
    
    Args:
        model: Trained ElectrophysiologyTransformer
        tokenizer: ElectrophysiologyTokenizer
        voltage_data: numpy array of voltage values
        current_data: numpy array of current values
        context_length: Context length
        prediction_length: Prediction length
        max_lag: Maximum lag
        device: torch device
    """
    # Process data
    voltage_proc, current_proc = tokenizer(voltage_data, current_data)
    
    # Prepare inputs
    total_length = context_length + max_lag + prediction_length
    if len(voltage_proc) < total_length:
        print(f"Warning: Data too short ({len(voltage_proc)} < {total_length})")
        return
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    #make a rolling visualization over the data
    for i in range(0, len(voltage_proc) - total_length + 1, prediction_length):
        # Take a segment
        voltage_seg = voltage_proc[i:total_length+i]
        current_seg = current_proc[i:total_length+i]
        
        # Split into past and future
        past_length = context_length + max_lag
        past_voltage = voltage_seg[:past_length]
        past_current = current_seg[:past_length]
        future_voltage_true = voltage_seg[past_length:]
        future_current = current_seg[past_length:]

        if i > 0:
            #append the predictions to the future voltage true for continuous plotting
            past_voltage_in = np.concatenate([past_voltage[:-prediction_length], predicted_voltage_orig ])
        else:
            past_voltage_in = past_voltage
        
        # Create input tensors
        past_values = torch.tensor(past_voltage_in).unsqueeze(0).to(device)  # (1, past_length)
        
        past_time = np.arange(past_length, dtype=np.float32).reshape(-1, 1) / past_length
        past_time = torch.tensor(past_time ,  dtype=torch.float32).unsqueeze(0).to(device)

        future_time = np.arange(past_length, past_length + prediction_length, dtype=np.float32).reshape(-1, 1) / past_length
        future_time = torch.tensor(future_time, dtype=torch.float32).unsqueeze(0).to(device)
        
        #stack the current as dynamic real features
        past_current_feat = past_current.reshape(-1, 1)
        future_current_feat = future_current.reshape(-1, 1)

        past_time_features = np.concatenate([past_time.cpu().numpy().squeeze(0), past_current_feat], axis=-1)
        future_time_features = np.concatenate([future_time.cpu().numpy().squeeze(0), future_current_feat], axis=-1)
        past_time_features = torch.tensor(past_time_features, dtype=torch.float32).unsqueeze(0).to(device)
        future_time_features = torch.tensor(future_time_features, dtype=torch.float32).unsqueeze(0).to(device)

        static_categorical_features = torch.tensor([[cell_id]], dtype=torch.long).to(device) if cell_id is not None else torch.randint(0, 1000, (1,1), dtype=torch.long).to(device)

        # Make prediction
        with torch.no_grad():
            outputs = model.generate(
                past_values=past_values,
                past_time_features=past_time_features,
                future_time_features=future_time_features,
                static_categorical_features=static_categorical_features,
            )
        
        # Get the decoder output - this contains the prediction
        # The actual prediction is in the distribution parameters
        # For now, let's just visualize what we have
        
        # Decode the data back to original scale
        past_voltage_orig = tokenizer.decode_voltage(past_voltage)
        past_current_orig = tokenizer.decode_current(past_current)
        future_voltage_orig = tokenizer.decode_voltage(future_voltage_true)
        future_current_orig = tokenizer.decode_current(future_current)
        predicted_voltage = outputs.sequences.squeeze(0).cpu().numpy()[3, :]
        predicted_voltage_orig = tokenizer.decode_voltage(predicted_voltage)
        # Create time axis
        time_past = np.arange(past_length)+i
        time_future = np.arange(past_length, past_length + prediction_length)+i



        # Voltage plot
        axes[0].plot(time_past, past_voltage_orig, 'b-', label='Past Voltage', linewidth=1.5)
        axes[0].plot(time_future, future_voltage_orig, 'g-', label='True Future Voltage', linewidth=1.5)
        axes[0].plot(time_future, predicted_voltage_orig, 'r--', label='Predicted Voltage', linewidth=1.5)
        axes[0].axvline(x=past_length, color='r', linestyle='--', alpha=0.5, label='Prediction Start')
        axes[0].set_ylabel('Voltage (mV)')
        axes[0].set_title('Voltage Response')
        #axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(-80, 20)
        # Current plot
        axes[1].plot(time_past, past_current_orig, 'b-', label='Past Current', linewidth=1.5)
        axes[1].plot(time_future, future_current_orig, 'orange', label='Future Current', linewidth=1.5)
        axes[1].axvline(x=past_length, color='r', linestyle='--', alpha=0.5, label='Prediction Start')
        axes[1].set_ylabel('Current (pA)')
        axes[1].set_xlabel('Time Steps')
        axes[1].set_title('Applied Current')
        #axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
    return fig

def test_denovo(model, tokenizer,   context_length=512,
    prediction_length=128,  max_lag=7, cell_id=None, device="cuda"):
    total_length = context_length + max_lag + prediction_length
    # Dummy test function for de novo prediction
    #square pulse current injection
    sr = 10000  # 10 kHz
    t = np.arange(0, 1, 1.0/sr)
    current_injection = np.zeros_like(t)
    current_injection[(t >= 0.5) & (t < 1.5)] = 2000.0  # 2000 pA pulse from 0.5s to 1.5s
    voltage_response = -65.0 * np.ones(total_length).astype(np.float32)  # baseline at -65 mV
    # Add some noise
    voltage_response += np.random.normal(0, 0.5, size=voltage_response.shape)
    
    #predict with model
    # Visualize prediction
    # Process data
    voltage_proc, current_proc = tokenizer(voltage_response, current_injection)
    volt_out = []

    for i in range(0, len(current_proc) - total_length + 1, prediction_length):
        
        print(f"De novo prediction step {i}/{len(current_proc) - total_length}")
        current_seg = current_proc[i:total_length+i].astype(np.float32)
        past_length = context_length + max_lag

        past_current = current_seg[:past_length]
        future_current = current_seg[past_length:]

        if i > 0:
            #append the predictions to the future voltage true for continuous plotting
            past_voltage_in = np.concatenate([past_voltage_in, predicted_voltage])[-past_length:]
        else:
            past_voltage_in = voltage_proc
                # Create input tensors
        past_values = torch.tensor(past_voltage_in).unsqueeze(0).to(device)  # (1, past_length)
        
        dt = 1.0 / sr
        past_time = np.arange(0, past_length * dt, dt, dtype=np.float32).reshape(-1, 1)
        future_time = np.arange(past_length * dt, (past_length + prediction_length) * dt, dt, dtype=np.float32).reshape(-1, 1)
        #makes the shape match just in case of rounding errors
        past_time = past_time[:past_length]
        future_time = future_time[:prediction_length]


        #stack the current as dynamic real features
        past_current_feat = past_current.reshape(-1, 1)
        future_current_feat = future_current.reshape(-1, 1)

        past_time_features = np.concatenate([past_time, past_current_feat], axis=-1)
        future_time_features = np.concatenate([future_time, future_current_feat], axis=-1)
        past_time_features = torch.tensor(past_time_features, dtype=torch.float32).unsqueeze(0).to(device)
        future_time_features = torch.tensor(future_time_features, dtype=torch.float32).unsqueeze(0).to(device)

        static_categorical_features = torch.tensor([[cell_id]], dtype=torch.long).to(device) if cell_id is not None else torch.randint(0, 500, (1,1), dtype=torch.long).to(device)

        # Make prediction
        with torch.no_grad():
            outputs = model.generate(
                past_values=past_values,
                past_time_features=past_time_features,
                future_time_features=future_time_features,
                static_categorical_features=static_categorical_features,
            )
        
        # Get the decoder output - this contains the prediction
        # The actual prediction is in the distribution parameters
        # For now, let's just visualize what we have
        
        # Decode the data back to original scale
        past_current_orig = tokenizer.decode_current(past_current)
        future_current_orig = tokenizer.decode_current(future_current)
        predicted_voltage = outputs.sequences.squeeze(0).cpu().numpy()[3, :]
        predicted_voltage_orig = tokenizer.decode_voltage(predicted_voltage)

        volt_out.extend(predicted_voltage_orig)
    
    volt_out = np.array(volt_out)
    # Plot the results
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    time_axis = np.arange(len(volt_out)) / sr
    axes[0].plot(time_axis, volt_out, 'r-', label='Predicted Voltage', linewidth=1.5)
    axes[0].set_ylabel('Voltage (mV)')
    axes[0].set_title('De Novo Predicted Voltage Response to Square Pulse Current Injection')
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(time_axis, current_injection[:len(volt_out)], 'orange', label='Input Current', linewidth=1.5)
    axes[1].set_ylabel('Current (pA)')
    fig.show()



def main():
    """Main inference script."""
    # Configuration
    DATA_PATH = "nn_ds_combined.joblib"
    CHECKPOINT_PATH = "best_val.pt"  # Change to your checkpoint
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {DEVICE}")
    
    # Check if checkpoint exists
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Checkpoint not found: {CHECKPOINT_PATH}")
        print("Please train a model first using train.py")
        return
    
    # Load model
    print("Loading model...")
    model = load_model(CHECKPOINT_PATH, device=DEVICE)
    print("Model loaded!")
    
    # Load data
    print("Loading data...")
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
        context_length= 512,
        prediction_length = 128,
        max_lag=7,
        data_length=10000,
    )

    #test_denovo(model, tokenizer)
    
    # Get a sample trial
    for trial_idx in range(50):
        data = dataset[trial_idx]
        
        voltage_sweep = data['past_values'][:, 0].numpy()
        current_sweep = data['past_time_features'][:, 1].numpy()
        volt_future = data['future_values'][:, 0].numpy()
        curr_future = data['future_time_features'][:, 1].numpy()
        voltage_sweep = np.concatenate([voltage_sweep, volt_future])
        current_sweep = np.concatenate([current_sweep, curr_future])
        print(f"Data loaded: {len(voltage_sweep)} time steps")
        if np.max(voltage_sweep) < -30:
            continue
        # Visualize prediction
        print("Making prediction...")
        fig = visualize_prediction(
            model=model,
            tokenizer=tokenizer,
            voltage_data=voltage_sweep,
            current_data=current_sweep,
            context_length=512,
            prediction_length=128,
            device=DEVICE,
        )
        plt.pause(10)
    
    if fig is not None:
        output_path = "prediction_visualization.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
        plt.show()


if __name__ == "__main__":
    main()
