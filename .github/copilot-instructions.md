# Copilot Coding Agent Instructions for nnGAN Electrophysiology Project

## Project Overview
This codebase implements a physics-informed transformer for patch-clamp electrophysiology data. The project has a **dual structure**: 
- **Root directory** (`nnGAN/`): Contains legacy models, data files, and utilities
- **Modern transformer** (`electrophysiology_transformer/`): Advanced physics-informed transformer with multi-scale encoding and specialized losses

The architecture processes high-frequency (10-50kHz) voltage/current recordings and supports multi-modal input with biophysical constraints.

## Key Architecture & Structure

### Root Directory (`nnGAN/`)
- **Data files**: `nn_ds*.joblib`, `temp_short_data.joblib` - Core electrophysiology datasets
- **legacy/**: Earlier transformer implementations, VAE models, and dataset utilities
- **utils.py**: Spike detection, feature extraction using IPFX (Allen Institute)
- **dataset_concat.py**: Data concatenation utilities

### Transformer Module (`electrophysiology_transformer/`)
- **models/physics_informed_transformer.py**: Multi-scale temporal encoder with `BiophysicalConstraintLayer`
- **data/enhanced_dataset.py**: Flexible data loader supporting `.joblib`, `.npy`, `.npz` with voltage/current
- **data/electrophysiology_tokenizer.py**: Adaptive tokenizer for voltage, derivatives, and biological events
- **config.py**: Dataclass-based configuration with predefined profiles (`get_debug_config()`, `get_production_config()`)
- **train_enhanced.py**: Main training with hierarchical learning and custom losses
- **visualize_checkpoint.py**: Comprehensive model evaluation and visualization

## Critical Data Patterns & Workflows

### Data Loading (Multi-format Support)
```python
# Joblib format (primary): voltage_data = data['voltage_traces'] or data['voltages']
# NPZ format: voltage = data['voltage'], current = data['current'] 
# Data paths are typically in root: "C:/Users/SMest/Dropbox/nnGAN/nn_ds_combined.joblib"
```

### Essential Commands
- **Train transformer**: `cd electrophysiology_transformer && python train_enhanced.py`
- **Test tokenizer**: `cd electrophysiology_transformer && python tests/test_tokenizer_roundtrip.py`
- **Visualize checkpoint**: `python visualize_checkpoint.py --checkpoint path/to/checkpoint`
- **Quick debug training**: Use `config = get_debug_config()` for fast iteration

### Path Management Pattern
```python
# Always add parent path when working in subdirectories:
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

## Physics-Informed Architecture

### Multi-Scale Temporal Processing
- **Encoder scales**: [1x, 5x, 10x, 50x] downsampling for different temporal resolutions
- **Biophysical constraints**: Voltage ranges (-120mV to 40mV), membrane time constants, action potential kinetics
- **Tokenizer categories**: 50% voltage levels, 25% derivatives, 25% biological events

### Specialized Training Features
- **Hierarchical stages**: Multi-resolution training via `downsample_factors` in config
- **Curriculum learning**: Gradual complexity increase (`use_curriculum`, `curriculum_steps`)
- **Custom losses**: Spike timing, frequency domain, continuity, physics constraints
- **Multi-modal**: Voltage + current processing with correlation analysis

## Developer Workflows
- **Install dependencies**: `pip install torch transformers scikit-learn scipy matplotlib seaborn pandas wandb joblib`
- **Train model**: `python train_enhanced.py` (or use `main(config)` for custom configs)
- **Evaluate**: Use `ElectrophysiologyEvaluator` and `VisualizationTools` from `evaluation.py`
- **Visualize checkpoint**: `python visualize_checkpoint.py --checkpoint <path>`
- **Data formats**: Accepts `.joblib`, `.npy`, `.npz` (with voltage/current keys)
- **Multi-modal input**: Set `include_current=True` in config for voltage + current training

## Integration Points
- **Hugging Face Transformers**: Extends `TimeSeriesTransformerForPrediction`
- **WandB**: Optional experiment tracking (`report_to` in config)
- **PyTorch**: All model, training, and evaluation logic

## Conventions & Tips
- **Config-driven**: Always use config objects for reproducibility
- **Multi-scale**: Encoder and loss support multiple temporal resolutions
- **Specialized Losses**: Use `ElectrophysiologyLoss` for domain-specific training
- **Debugging**: Use `get_debug_config()` for fast iteration
- **Performance**: Enable `fp16`, `tf32`, and optimize DataLoader for speed

## Example: Quick Training
```python
from config import get_debug_config
config = get_debug_config()
config.data.data_path = "your_data.joblib"
from train_enhanced import main
main(config)
```

## Troubleshooting
- **CUDA OOM**: Lower batch size, enable gradient accumulation, use smaller model
- **Data issues**: Check format, file paths, and memory
- **Spike detection**: Adjust `spike_loss_weight` and `spike_threshold`

---
For more details, see `README.md` and config examples in `config.py`. If any section is unclear or missing, please provide feedback for further refinement.
