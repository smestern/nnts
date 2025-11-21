"""Electrophysiology Transformer wrapper.

Configures ``TimeSeriesTransformerForPrediction`` for patch-clamp electrophysiology with
voltage as the ONLY predicted target and injected current provided as a known dynamic
real covariate (available in the past and future windows).

Key configuration differences from the previous version:
* input_size=1 (voltage only)
* num_time_features=1 (time index / age feature)
* num_dynamic_real_features=1 (current trace)
* Loss computed only over voltage values
"""

import torch
import torch.nn as nn
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction


class ElectrophysiologyTransformer(nn.Module):
    """Forecast future membrane voltage given past voltage and known current profile.

    Current values (past + future) are passed as dynamic real features inside the
    ``past_time_features`` and ``future_time_features`` tensors (second column after the
    time index).
    """
    
    def __init__(
        self,
        context_length: int = 128,
        prediction_length: int = 32,
        d_model: int = 128,
        encoder_layers: int = 4,
        decoder_layers: int = 4,
        encoder_attention_heads: int = 8,
        decoder_attention_heads: int = 8,
        encoder_ffn_dim: int = 512,
        decoder_ffn_dim: int = 512,
        dropout: float = 0.1,
        lags_sequence = [1, 2, 3, 4, 5, 6, 7],
        static_categorical_features = 1,
        static_real_features = 0,
        cardinality = [2000],
        embedding_dimension = [16],
        scaling: str = "mean",  # "mean", "std", or None
    ) -> None:
        """
        Args:
            context_length: Number of past time steps to use as context
            prediction_length: Number of future time steps to predict
            d_model: Dimension of the model
            encoder_layers: Number of encoder layers
            decoder_layers: Number of decoder layers
            encoder_attention_heads: Number of attention heads in encoder
            decoder_attention_heads: Number of attention heads in decoder
            encoder_ffn_dim: Dimension of encoder feedforward network
            decoder_ffn_dim: Dimension of decoder feedforward network
            dropout: Dropout rate
            scaling: Type of scaling to use ("mean", "std", or None)
        """
        super().__init__()
        
        # We predict ONLY voltage -> input_size=1. Current is a dynamic real covariate.
        config = TimeSeriesTransformerConfig(
            prediction_length=prediction_length,
            context_length=context_length,
            input_size=1,  # voltage only
            lags_sequence=lags_sequence,  # Use recent lags
            num_time_features=1,  # Simple time index / age feature
            num_dynamic_real_features=1,  # current as known future covariate
            num_static_categorical_features=static_categorical_features,
            num_static_real_features=static_real_features,
            cardinality=cardinality,
            embedding_dimension=embedding_dimension,
            d_model=d_model,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            encoder_attention_heads=encoder_attention_heads,
            decoder_attention_heads=decoder_attention_heads,
            encoder_ffn_dim=encoder_ffn_dim,
            decoder_ffn_dim=decoder_ffn_dim,
            dropout=dropout,
            activation_function="gelu",
            scaling=scaling,
            distribution_output="student_t",  # Predict mean and std
            loss="nll",  # Negative log likelihood
        )

        self.config = config
        self.model = TimeSeriesTransformerForPrediction(config)
        
    def forward(
        self,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        future_time_features: torch.Tensor,
        past_observed_mask: torch.Tensor | None = None,
        future_values: torch.Tensor | None = None,
        static_categorical_features: torch.Tensor | None = None,
        static_real_features: torch.Tensor | None = None,
    ):
        """
        Forward pass.
        
        Args:
            past_values: (batch_size, past_length, 1) voltage only
            past_time_features: (batch_size, past_length, 2) [time_index, current]
            future_time_features: (batch_size, prediction_length, 2) [time_index, future_current]
            past_observed_mask: (batch_size, past_length, 1 or past_length) mask of observed voltage
            future_values: (batch_size, prediction_length, 1) voltage targets
            
        Returns:
            Model outputs including predictions and loss (if future_values provided)
        """
        return self.model(
            past_values=past_values.squeeze(-1)  if past_values.ndim >= 3 else past_values,  # Remove last dim for compatibility
            past_time_features=past_time_features,
            future_time_features=future_time_features,
            past_observed_mask=past_observed_mask.squeeze(-1)  if past_observed_mask is not None else None,
            future_values=future_values.squeeze(-1) if future_values is not None and future_values.ndim >=3 else future_values,
            static_categorical_features=static_categorical_features,
            static_real_features=static_real_features,
        )
    
    def predict(
        self,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        future_time_features: torch.Tensor,
        past_observed_mask: torch.Tensor | None = None,
    ):
        """
        Generate predictions.
        
        Args:
            past_values: (batch_size, past_length, 1)
            past_time_features: (batch_size, past_length, 2)
            future_time_features: (batch_size, prediction_length, 2)
            past_observed_mask: (batch_size, past_length, 1 or past_length)
            
        Returns:
            Model outputs with predictions
        """
        outputs = self.model(
            past_values=past_values,
            past_time_features=past_time_features,
            future_time_features=future_time_features,
            past_observed_mask=past_observed_mask,
        )
        return outputs

    def generate(
        self,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        future_time_features: torch.Tensor,
        past_observed_mask: torch.Tensor | None = None,
        static_categorical_features: torch.Tensor | None = None,
        static_real_features: torch.Tensor | None = None,
        **kwargs,
    ):
        """
        Generate future predictions.
        
        Args:
            past_values: (batch_size, past_length, 1)
            past_time_features: (batch_size, past_length, 2)
            future_time_features: (batch_size, prediction_length, 2)
            past_observed_mask: (batch_size, past_length, 1 or past_length)
            
        Returns:
            Model outputs with predictions
        """
        outputs = self.model.generate(
            past_values=past_values.squeeze(-1) if past_values.ndim >= 3 else past_values,
            past_time_features=past_time_features,
            future_time_features=future_time_features,
            past_observed_mask=past_observed_mask.squeeze(-1) if past_observed_mask is not None else None,
            static_categorical_features=static_categorical_features,
            static_real_features=static_real_features,
            **kwargs,
        )
        return outputs