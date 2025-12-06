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
        include_time_features=False,
        include_real_valued_features=False,
        real_features_list=['input_resistance', 'tau', 'v_baseline', 'rheobase_i', 'ap_mean_threshold_v_0_long_square','ap_1_peak_v_0_long_square']
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
        self.include_time_features = include_time_features
        self.include_real_valued_features = include_real_valued_features
        self.list_real_valued_features = real_features_list
        data = joblib.load(data_path)
        self.upsample_unique_data = True
        voltages = data['responses']
        currents = data['commands']
        info = data.get('info', None)

        self.samples = []
        self._prepare_samples(voltages, currents, info)
        self.index = self._build_index()
        self._upsample_unique_data() if self.upsample_unique_data else None
        self.len_real_valued_features = len(self.list_real_valued_features)
        self.len_static_categorical_features = 1  # cell-of-origin as static categorical feature

        if not self.index:
            raise ValueError("No valid samples found after filtering.")

        print(f"Prepared {len(self.index)} windows from {len(self.samples)} trials")

    def _prepare_samples(self, voltages, currents, info=None):
        """Clean trials and concatenate sweeps once they pass basic quality checks."""
        for i, (v_trial, c_trial) in enumerate(zip(voltages, currents)):
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

            if self.include_real_valued_features and info is not None:
                trial_info = info[i]
                for feature in self.list_real_valued_features:
                    if feature in trial_info:
                        value = trial_info[feature]
                        if np.isnan(value):
                            value = 0.0
                    else:
                        value = 0.0
                    # Here you can store or process the real-valued feature as needed
                    # For simplicity, we are not using them further in this example

            self.samples.append(
                {
                    'voltage': voltage_concat,
                    'current': current_concat,
                    'real_valued_features': {feature: trial_info.get(feature, 0.0) for feature in self.list_real_valued_features} if self.include_real_valued_features and info is not None else {}
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
    
    def _upsample_unique_data(self):
        """Upsample unique data samples to balance the dataset. Essentially we want to get segments where current is changing more often."""
        #use the precomputed index to find unique samples
        print("Upsampling unique data segments...")
        unique_spans = set()
        for span in self.index:
            trial_idx, start_idx = span
            sample = self.samples[trial_idx]
            current_window = sample['current'][start_idx:start_idx + self.total_length]
            if np.any(np.diff(current_window) != 0):
                unique_spans.add(span)
        unique_spans = list(unique_spans)
        #upsample by jittering the idxs a little
        augmented_spans = []
        for span in unique_spans:
            trial_idx, start_idx = span
            for shift in [-2, -1, 0, 1, 2]:
                new_start = start_idx + shift
                sample = self.samples[trial_idx]
                if 0 <= new_start <= sample['voltage'].shape[0] - self.total_length:
                    augmented_spans.append((trial_idx, new_start))
        

        #only grow to double the dataset size
        target_size = min(len(self.index) * 2, len(self.index) + len(augmented_spans))
        current_size = len(self.index)
        if current_size < target_size:
            needed = target_size - current_size
            self.index.extend(augmented_spans[:needed])
            print(f"Upsampled {len(unique_spans)} unique spans to {len(augmented_spans[:needed])} spans.")
        else:
            self.index.extend(augmented_spans)
            print(f"Upsampled {len(unique_spans)} unique spans to {len(augmented_spans)} spans.")
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
            'static_real_features': torch.tensor(
                [sample['real_valued_features'].get(feature, 0.0) for feature in self.list_real_valued_features],
                dtype=torch.float32
            ) if self.include_real_valued_features else torch.tensor([] , dtype=torch.float32),
        }

    