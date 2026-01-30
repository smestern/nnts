"""
Enhanced Multi-Scale Electrophysiology Transformer

Addresses the challenge of modeling both fast spikes (0-3ms) and slow membrane 
dynamics (500ms) at 10kHz sampling rate through:
1. Multi-scale temporal encoding
2. Hierarchical local + global attention
3. Learnable timescale decomposition
4. Learned absolute positional encodings (preserves fixed biological timescales)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class MultiScaleTemporalEncoder(nn.Module):
    """
    Parallel pathways for different temporal scales:
    - Fast (1x): 0-10ms - spikes, action potentials
    - Medium (5x): 10-50ms - burst patterns, ISI
    - Slow (25x): 50-250ms - adaptation, subthreshold
    - Ultra-slow (100x): 250ms+ - long-term trends, steady state
    """
    
    def __init__(self, d_model: int = 256, dropout: float = 0.1):
        super().__init__()
        
        quarter_dim = d_model // 4
        
        # Fast pathway - captures spikes (kernel ~0.3ms @ 10kHz)
        self.fast_conv = nn.Sequential(
            nn.Conv1d(1, quarter_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Medium pathway - captures burst patterns (kernel ~1.5ms)
        self.medium_conv = nn.Sequential(
            nn.Conv1d(1, quarter_dim, kernel_size=15, stride=5, padding=7),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Slow pathway - captures adaptation (kernel ~7.5ms)
        self.slow_conv = nn.Sequential(
            nn.Conv1d(1, quarter_dim, kernel_size=75, stride=25, padding=37),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Ultra-slow pathway - captures long trends (kernel ~30ms)
        self.ultra_slow_conv = nn.Sequential(
            nn.Conv1d(1, quarter_dim, kernel_size=300, stride=100, padding=150),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Fusion and normalization
        self.fusion = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, length, 1] voltage trace
        Returns:
            [batch, length, d_model] multi-scale features
        """
        x_t = x.transpose(1, 2)  # [B, 1, L]
        target_length = x.size(1)
        
        # Process each pathway
        fast = self.fast_conv(x_t)  # [B, d/4, L]
        
        medium = self.medium_conv(x_t)  # [B, d/4, L/5]
        medium = F.interpolate(medium, size=target_length, mode='linear', align_corners=False)
        
        slow = self.slow_conv(x_t)  # [B, d/4, L/25]
        slow = F.interpolate(slow, size=target_length, mode='linear', align_corners=False)
        
        ultra = self.ultra_slow_conv(x_t)  # [B, d/4, L/100]
        ultra = F.interpolate(ultra, size=target_length, mode='linear', align_corners=False)
        
        # Concatenate all scales
        multi_scale = torch.cat([fast, medium, slow, ultra], dim=1)  # [B, d, L]
        multi_scale = multi_scale.transpose(1, 2)  # [B, L, d]
        
        # Fuse and return
        return self.fusion(multi_scale)


class LocalGlobalAttention(nn.Module):
    """
    Hierarchical attention combining:
    - Local windowed attention for spikes (efficient, O(n*w))
    - Sparse global attention for long-range dependencies (O(n*s))
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        window_size: int = 64,  # ~6.4ms @ 10kHz
        global_stride: int = 16,  # Sample every 1.6ms for global
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.window_size = window_size
        self.global_stride = global_stride
        
        # Local attention for fine-grained features
        self.local_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        
        # Global attention for long-range dependencies
        self.global_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        
        # Projection to combine local and global
        self.output_proj = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, length, d_model]
            mask: Optional attention mask
        Returns:
            [batch, length, d_model]
        """
        B, L, D = x.shape
        
        # ===== Local Windowed Attention =====
        # Pad sequence to be divisible by window_size
        pad_len = (self.window_size - L % self.window_size) % self.window_size
        if pad_len > 0:
            x_padded = F.pad(x, (0, 0, 0, pad_len))
        else:
            x_padded = x
        
        # Reshape into windows
        num_windows = x_padded.size(1) // self.window_size
        x_windowed = x_padded.view(B, num_windows, self.window_size, D)
        x_windowed = x_windowed.reshape(B * num_windows, self.window_size, D)
        
        # Apply local attention within each window
        local_out, _ = self.local_attn(x_windowed, x_windowed, x_windowed)
        local_out = local_out.reshape(B, num_windows * self.window_size, D)
        
        # Remove padding
        local_features = local_out[:, :L, :]
        
        # ===== Sparse Global Attention =====
        # Sample tokens at regular stride for global context
        global_indices = torch.arange(0, L, self.global_stride, device=x.device)
        global_tokens = x[:, global_indices, :]  # [B, L//stride, D]
        
        # Apply global attention
        global_out, _ = self.global_attn(global_tokens, global_tokens, global_tokens)
        
        # Interpolate global context back to full resolution
        global_out_t = global_out.transpose(1, 2)  # [B, D, L//stride]
        global_features = F.interpolate(global_out_t, size=L, mode='linear', align_corners=False)
        global_features = global_features.transpose(1, 2)  # [B, L, D]
        
        # ===== Combine Local + Global =====
        combined = torch.cat([local_features, global_features], dim=-1)  # [B, L, 2*D]
        output = self.output_proj(combined)
        output = self.norm(output + x)  # Residual connection
        
        return output


class TimescaleDecompositionLayer(nn.Module):
    """
    Explicitly separate fast (spikes) and slow (membrane) components
    using learnable frequency-domain filtering
    """
    
    def __init__(self, d_model: int = 256, cutoff_freq: float = 100.0, dropout: float = 0.1):
        super().__init__()
        self.cutoff_freq = cutoff_freq  # Hz
        
        # Separate processing for fast and slow components
        self.fast_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.slow_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Learnable mixing weights
        self.fast_weight = nn.Parameter(torch.tensor(1.0))
        self.slow_weight = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x: torch.Tensor, sampling_rate: float = 10000.0) -> torch.Tensor:
        """
        Args:
            x: [batch, length, d_model]
            sampling_rate: Hz (default 10kHz)
        Returns:
            [batch, length, d_model]
        """
        B, L, D = x.shape
        
        # FFT along time dimension (per feature)
        x_fft = torch.fft.rfft(x, dim=1)  # [B, L//2+1, D]
        freqs = torch.fft.rfftfreq(L, 1.0 / sampling_rate).to(x.device)  # [L//2+1]
        
        # Create frequency masks
        low_mask = freqs < self.cutoff_freq  # Slow components
        high_mask = freqs >= self.cutoff_freq  # Fast components
        
        # Separate frequency components
        slow_fft = x_fft.clone()
        slow_fft[:, high_mask, :] = 0
        slow_signal = torch.fft.irfft(slow_fft, n=L, dim=1)
        
        fast_fft = x_fft.clone()
        fast_fft[:, low_mask, :] = 0
        fast_signal = torch.fft.irfft(fast_fft, n=L, dim=1)
        
        # Project and weight each component
        fast_out = self.fast_weight * self.fast_proj(fast_signal)
        slow_out = self.slow_weight * self.slow_proj(slow_signal)
        
        return fast_out + slow_out


class LearnedAbsolutePositionalEncoding(nn.Module):
    """
    Learned absolute positional encoding that preserves fixed biological timescales.
    
    Unlike relative encoding, this allows the model to learn that:
    - Spike width is always ~1-2ms (10-20 samples @ 10kHz)
    - Refractory period is always ~2-5ms
    - Membrane time constants are fixed values
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Learnable positional embeddings (more flexible than sinusoidal)
        self.position_embeddings = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, length, d_model]
        Returns:
            [batch, length, d_model] with positional information added
        """
        seq_len = x.size(1)
        # Add positional encoding (absolute position from start of sequence)
        x = x + self.position_embeddings[:, :seq_len, :]
        return self.dropout(x)


class EnhancedElectrophysiologyTransformer(nn.Module):
    """
    Multi-scale transformer for electrophysiology with:
    - Multi-resolution temporal encoding
    - Local + global hierarchical attention
    - Explicit fast/slow timescale decomposition
    - Learned absolute positional encoding (preserves biological timescales)
    """
    
    def __init__(
        self,
        context_length: int = 512,
        prediction_length: int = 128,
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        window_size: int = 64,
        global_stride: int = 16,
        dropout: float = 0.1,
        cutoff_freq: float = 100.0,
    ):
        super().__init__()
        
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.d_model = d_model
        
        # Multi-scale input encoding
        self.multi_scale_encoder = MultiScaleTemporalEncoder(d_model, dropout)
        
        # Absolute positional encoding (learns fixed biological timescales)
        self.pos_encoder = LearnedAbsolutePositionalEncoding(
            d_model, 
            max_len=max(context_length, prediction_length) * 2,  # Extra headroom
            dropout=dropout
        )
        
        # Current injection as additional feature
        self.current_proj = nn.Linear(1, d_model)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': LocalGlobalAttention(d_model, num_heads, window_size, global_stride, dropout),
                'decomposition': TimescaleDecompositionLayer(d_model, cutoff_freq, dropout),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model),
                    nn.Dropout(dropout),
                ),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model),
            })
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, prediction_length),
        )
        
    def forward(
        self,
        past_values: torch.Tensor,
        past_current: Optional[torch.Tensor] = None,
        future_current: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            past_values: [batch, context_length, 1] voltage
            past_current: [batch, context_length, 1] injected current (optional)
            future_current: [batch, prediction_length, 1] future current (optional)
        
        Returns:
            predictions: [batch, prediction_length] predicted voltage
        """
        # Multi-scale encoding of voltage
        x = self.multi_scale_encoder(past_values)  # [B, L, d]
        
        # Add current information if available
        if past_current is not None:
            current_features = self.current_proj(past_current)
            x = x + current_features
        
        # Apply encoder layers
        for layer in self.encoder_layers:
            # Multi-head attention
            attn_out = layer['attention'](layer['norm1'](x))
            x = x + attn_out
            
            # Timescale decomposition
            decomp_out = layer['decomposition'](x)
            x = x + decomp_out
            
            # Feedforward
            ffn_out = layer['ffn'](layer['norm2'](x))
            x = x + ffn_out
        
        # Global pooling across time + projection to predictions
        # Use both mean (slow trends) and max (spike peaks)
        pooled = torch.cat([x.mean(dim=1), x.max(dim=1)[0]], dim=-1)  # [B, 2*d]
        pooled = pooled[:, :self.d_model]  # Truncate to d_model
        
        # Generate predictions
        predictions = self.output_head(pooled)  # [B, prediction_length]
        
        return predictions
