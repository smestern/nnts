"""
Training script for Electrophysiology Transformer
"""
import os
import sys
import joblib
import numpy as np
import torch
from torch.utils.data import Dataset
# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



class ElectrophysiologyDataset(Dataset):
    """Dataset for electrophysiology time series data."""
    
    def __init__(
        self,
        data_path,
        tokenizer,
        context_length=128,
        prediction_length=32,
        max_lag=7,
        sample_rate=10000,
        data_length=None,
    ):
        """
        Args:
            data_path: Path to .joblib file with 'voltages' and 'currents' keys
            tokenizer: ElectrophysiologyTokenizer instance
            context_length: Number of past time steps
            prediction_length: Number of future time steps to predict
            max_lag: Maximum lag used by the model
        """
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.max_lag = max_lag
        self.sample_rate = sample_rate
        self.data_length = data_length
        self.segment_stride = prediction_length
        self.total_length = self.context_length + self.max_lag + self.prediction_length

        data = joblib.load(data_path)
        voltages = data['voltages']
        currents = data['currents']

        self.samples = []
        self._prepare_samples(voltages, currents)
        self.index = self._build_index()

        if not self.index:
            raise ValueError("No valid samples found after filtering.")

        print(f"Prepared {len(self.index)} windows from {len(self.samples)} trials")

    def _prepare_samples(self, voltages, currents):
        """Clean trials and concatenate sweeps once they pass basic quality checks."""
        for v_trial, c_trial in zip(voltages, currents):
            if np.isscalar(v_trial):
                continue

            cleaned_voltage = []
            cleaned_current = []

            for v, c in zip(v_trial, c_trial):
                v_arr = np.asarray(v, dtype=np.float32)
                c_arr = np.asarray(c, dtype=np.float32)

                if v_arr.shape != c_arr.shape:
                    continue
                if np.isnan(v_arr).any() or np.isnan(c_arr).any():
                    continue

                cleaned_voltage.append(v_arr)
                cleaned_current.append(c_arr)

            if not cleaned_voltage:
                continue

            voltage_concat = np.concatenate(cleaned_voltage)
            current_concat = np.concatenate(cleaned_current)

            if voltage_concat.shape[0] < self.total_length:
                continue

            self.samples.append(
                {
                    'voltage': voltage_concat,
                    'current': current_concat,
                }
            )

    def _build_index(self):
        """Pre-compute start indices so __getitem__ mapping is deterministic."""
        spans = []
        for trial_idx, sample in enumerate(self.samples):
            max_start = sample['voltage'].shape[0] - self.total_length
            starts = range(0, max_start + 1, self.segment_stride)
            for start in starts:
                spans.append((trial_idx, start))

        if self.data_length is not None and self.data_length < len(spans):
            rng = np.random.default_rng(0)
            selected = rng.choice(len(spans), size=self.data_length, replace=False)
            spans = [spans[i] for i in sorted(selected)]

        return spans

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        """Return one deterministic window of (voltage, current) data."""
        trial_idx, start_idx = self.index[idx]
        sample = self.samples[trial_idx]

        end_idx = start_idx + self.total_length
        voltage_window = sample['voltage'][start_idx:end_idx]
        current_window = sample['current'][start_idx:end_idx]

        voltage_processed, current_processed = self.tokenizer(voltage_window, current_window)

        past_length = self.context_length + self.max_lag

        past_voltage = voltage_processed[:past_length]
        past_current = current_processed[:past_length]
        future_voltage = voltage_processed[past_length:]
        future_current = current_processed[past_length:]

        past_values = np.expand_dims(past_voltage, axis=-1)
        future_values = np.expand_dims(future_voltage, axis=-1)

        dt = 1.0 / self.sample_rate
        past_time = np.arange(0, past_length * dt, dt, dtype=np.float32).reshape(-1, 1)
        future_time = np.arange(past_length * dt, (past_length + self.prediction_length) * dt, dt, dtype=np.float32).reshape(-1, 1)

        #makes the shape match just in case of rounding errors
        past_time = past_time[:past_length]
        future_time = future_time[:self.prediction_length]


        past_current_feat = past_current.reshape(-1, 1)
        future_current_feat = future_current.reshape(-1, 1)
        past_time_features = np.concatenate([past_time, past_current_feat], axis=-1)
        future_time_features = np.concatenate([future_time, future_current_feat], axis=-1)

        past_observed_mask = np.ones_like(past_values)

        return {
            'past_values': torch.tensor(past_values, dtype=torch.float32),
            'future_values': torch.tensor(future_values, dtype=torch.float32),
            'past_time_features': torch.tensor(past_time_features, dtype=torch.float32),
            'future_time_features': torch.tensor(future_time_features, dtype=torch.float32),
            'past_observed_mask': torch.tensor(past_observed_mask, dtype=torch.float32),
            'static_categorical_features': torch.tensor([trial_idx], dtype=torch.long), # Added static categorical feature, technically the cell-of-origin
        }
