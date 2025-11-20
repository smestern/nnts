# Quick Start Guide

## What We Built

A simplified transformer model for electrophysiology data with three core components:

1. **Tokenizer** (`tokenizer.py`) - Clips and normalizes voltage/current data
2. **Model** (`model.py`) - TimeSeriesTransformer subclass for multivariate time series
3. **Training** (`train.py`) - Complete training pipeline

## Quick Test

```bash
cd electrophysiology_transformer
python test.py
```

You should see:
```
============================================================
Electrophysiology Transformer Test Suite
============================================================

Testing Tokenizer...
  ✓ Tokenizer working!

Testing Data Loading...
  ✓ Data loading working!

Testing Model...
  ✓ Model forward pass working!
  ✓ Model prediction working!

============================================================
All tests passed! ✓
============================================================
```

## Training

```bash
python train.py
```

This will:
- Load data from `../nn_ds_combined.joblib`
- Train for 10 epochs
- Save checkpoints every 5 epochs
- Save final model as `final_model.pt`

## Key Concepts

### Data Flow
```
Input Data (lists of lists)
    ↓
Concatenate sweeps
    ↓
Tokenizer (clip + normalize)
    ↓
Create batches
    ↓
Model forward pass
    ↓
Loss calculation
    ↓
Backprop + optimize
```

### Model Input/Output
```
Inputs:
  - past_values: (batch, context_length+max_lag, 2) [voltage, current]
  - past_time_features: (batch, context_length+max_lag, 1)
  - future_time_features: (batch, prediction_length, 1)
  - future_values: (batch, prediction_length, 2) [for training only]

Output:
  - loss: scalar (during training)
  - encoder_last_hidden_state: encoded representations
```

### TimeSeriesTransformer Specifics

**Important differences from standard transformers:**

1. **Multivariate Input**: Uses `input_size=2` for voltage + current
2. **Time Features**: Requires explicit time features (not learned positions)
3. **Scaling**: Built-in data scaling (mean, std, or none)
4. **Lags**: Automatically creates lagged features from past values
5. **Distribution Output**: Predicts distribution parameters (mean, std)
6. **Context vs Prediction**: Separate context and prediction lengths

**Key config parameters:**
- `context_length`: Past time steps for context
- `prediction_length`: Future time steps to predict
- `input_size`: Number of variates (2 for voltage + current)
- `lags_sequence`: Which lags to use [1,2,3,4,5,6,7]
- `num_time_features`: Number of time features (1 = simple index)

## Configuration

Edit these in `train.py`:
```python
CONTEXT_LENGTH = 128      # Past context window
PREDICTION_LENGTH = 32    # Future prediction window
BATCH_SIZE = 16          # Batch size
EPOCHS = 10              # Training epochs
LEARNING_RATE = 1e-4     # Adam learning rate
```

Edit these in model creation:
```python
d_model = 128            # Model dimension
encoder_layers = 4       # Encoder depth
decoder_layers = 4       # Decoder depth
encoder_attention_heads = 8  # Attention heads
```

## Files Overview

```
electrophysiology_transformer/
├── __init__.py              # Package init
├── model.py                 # ElectrophysiologyTransformer
├── tokenizer.py            # ElectrophysiologyTokenizer
├── train.py                # Training script
├── test.py                 # Test suite
├── visualize.py            # Inference & visualization
├── README.md               # Full documentation
└── QUICKSTART.md           # This file
```

## Common Issues

**"Data too short" warning:**
- Increase length of data or decrease `CONTEXT_LENGTH + MAX_LAG + PREDICTION_LENGTH`

**Out of memory:**
- Reduce `BATCH_SIZE`
- Reduce `d_model` or number of layers
- Reduce sequence lengths

**Poor predictions:**
- Train for more epochs
- Adjust learning rate
- Check data preprocessing (clipping ranges)
- Verify data quality

## Next Steps

1. Run `python test.py` to verify installation ✓
2. Run `python train.py` to train a model
3. Run `python visualize.py` to see predictions (after training)
4. Adjust hyperparameters based on results
5. Add custom evaluation metrics

## Learning Resources

- HuggingFace TimeSeriesTransformer: [docs](https://huggingface.co/docs/transformers/model_doc/time_series_transformer)
- Original paper: "Temporal Fusion Transformers"
- See `modeling_time_series_transformer.py` for implementation details

## Need Help?

Check:
1. `README.md` - Full documentation
2. `test.py` - Working examples
3. Comments in source code
4. TimeSeriesTransformer documentation
