# Electrophysiology Transformer

A simplified transformer model for patch-clamp electrophysiology data, built on the HuggingFace TimeSeriesTransformer.

## Overview

This project implements a transformer model to predict voltage responses from current stimuli in patch-clamp recordings. The model uses:
- **Past voltage + current** (context)
- **Future current** (known stimulus)

To predict:
- **Future voltage** (neuronal response)

## Components

### 1. Tokenizer (`tokenizer.py`)
Simple preprocessing that:
- Clips voltage to physiological range (-120 to 40 mV)
- Clips current to reasonable range (-1000 to 1000 pA)
- Optionally normalizes to [-1, 1] range

### 2. Model (`model.py`)
Subclasses `TimeSeriesTransformerForPrediction` with:
- Encoder-decoder architecture
- Multi-head attention
- Multivariate input (voltage + current)
- Probabilistic predictions (mean and std)

### 3. Training Script (`train.py`)
Complete training pipeline with:
- Data loading from `.joblib` files
- Batching and data augmentation
- Checkpoint saving
- Progress tracking

## Data Format

Expected data structure in `.joblib` file:
```python
{
    'voltages': [  # List of trials
        [sweep1, sweep2, ...],  # Each trial has multiple sweeps
        ...
    ],
    'currents': [  # List of trials
        [sweep1, sweep2, ...],
        ...
    ]
}
```

## Usage

### Testing
Run the test suite to verify everything works:
```bash
python test.py
```

### Training
Train the model:
```bash
python train.py
```

Configuration in `train.py`:
- `CONTEXT_LENGTH = 128` - Past time steps for context
- `PREDICTION_LENGTH = 32` - Future time steps to predict
- `BATCH_SIZE = 16` - Batch size
- `EPOCHS = 10` - Number of training epochs
- `LEARNING_RATE = 1e-4` - Adam learning rate

### Custom Training
```python
from model import ElectrophysiologyTransformer
from tokenizer import ElectrophysiologyTokenizer

# Create tokenizer
tokenizer = ElectrophysiologyTokenizer(normalize=True)

# Create model
model = ElectrophysiologyTransformer(
    context_length=128,
    prediction_length=32,
    d_model=128,
    encoder_layers=4,
    decoder_layers=4,
)

# Process data
voltage_proc, current_proc = tokenizer(voltage_data, current_data)

# Make predictions
predictions = model.predict(
    past_values=past_values,
    past_time_features=past_time_features,
    future_time_features=future_time_features,
)
```

## Model Architecture

```
Input: [past_voltage, past_current, future_current]
       │
       ▼
   Tokenizer (clip + normalize)
       │
       ▼
   Transformer Encoder
       │
       ▼
   Transformer Decoder
       │
       ▼
   Distribution Head (Normal)
       │
       ▼
   Output: future_voltage (mean, std)
```

## Key Parameters

- **context_length**: Number of past time steps (default: 128)
- **prediction_length**: Number of future time steps (default: 32)
- **d_model**: Model dimension (default: 128)
- **encoder/decoder_layers**: Number of transformer layers (default: 4)
- **attention_heads**: Number of attention heads (default: 8)
- **scaling**: Data scaling method ("std", "mean", or None)

## Files

- `model.py` - Transformer model definition
- `tokenizer.py` - Data preprocessing
- `train.py` - Training script
- `test.py` - Test suite
- `README.md` - This file

## Requirements

```
torch
transformers
numpy
joblib
tqdm
```

## Notes

- The model uses the TimeSeriesTransformer from HuggingFace, which is designed for time series forecasting
- Input size is 2 (voltage + current as multivariate series)
- The model learns a probabilistic distribution over future voltage
- Built-in scaling helps with numerical stability
- Supports CUDA acceleration if available

## Future Enhancements

Possible extensions (not implemented yet):
- FFT-based feature augmentation
- Multi-scale temporal encoding
- Physics-informed losses
- Spike detection and timing losses
- More sophisticated time features

## License

See main repository LICENSE.
