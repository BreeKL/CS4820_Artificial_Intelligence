"""
Transformer Model for Light Curve Classification

Implements a time-series transformer with temporal positional encoding,
multi-head attention, and convolutional embedding layers.

References:
- Vaswani et al. (2017): "Attention Is All You Need"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class TemporalPositionalEncoding(nn.Module):
    """Positional encoding using actual observation timestamps."""
    
    def __init__(self, d_model: int, max_len: int = 10000):
        """
        Initialize temporal positional encoding.
        
        Args:
            d_model: Dimension of the model
            max_len: Maximum sequence length
        """
        super().__init__()
        self.d_model = d_model
        
        # Create time embedding projection
        self.time_proj = nn.Linear(1, d_model)
        
    def forward(self, x: torch.Tensor, timestamps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            timestamps: Optional tensor of observation times (batch, seq_len)
            
        Returns:
            Tensor with positional encoding added
        """
        batch_size, seq_len, _ = x.shape
        
        if timestamps is not None:
            # Use actual timestamps
            # Normalize timestamps to [0, 1] range
            timestamps = timestamps.unsqueeze(-1)  # (batch, seq_len, 1)
            time_min = timestamps.min(dim=1, keepdim=True)[0]
            time_max = timestamps.max(dim=1, keepdim=True)[0]
            timestamps_norm = (timestamps - time_min) / (time_max - time_min + 1e-8)
            
            pos_encoding = self.time_proj(timestamps_norm)
        else:
            # Fall back to standard sinusoidal positional encoding
            position = torch.arange(seq_len, dtype=torch.float32, device=x.device).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, self.d_model, 2, dtype=torch.float32, device=x.device) 
                * -(np.log(10000.0) / self.d_model)
            )
            
            pos_encoding = torch.zeros(seq_len, self.d_model, device=x.device)
            pos_encoding[:, 0::2] = torch.sin(position * div_term)
            pos_encoding[:, 1::2] = torch.cos(position * div_term)
            pos_encoding = pos_encoding.unsqueeze(0).expand(batch_size, -1, -1)
        
        return x + pos_encoding


class ConvEmbedding(nn.Module):
    """Convolutional embedding layer for time series."""
    
    def __init__(self, input_dim: int, d_model: int, kernel_size: int = 3):
        """
        Initialize convolutional embedding.
        
        Args:
            input_dim: Input feature dimension
            d_model: Output embedding dimension
            kernel_size: Convolution kernel size
        """
        super().__init__()
        
        padding = (kernel_size - 1) // 2
        
        self.conv1 = nn.Conv1d(
            input_dim, d_model // 2, 
            kernel_size=kernel_size, 
            padding=padding
        )
        self.conv2 = nn.Conv1d(
            d_model // 2, d_model, 
            kernel_size=kernel_size, 
            padding=padding
        )
        self.activation = nn.ReLU()
        self.norm = nn.BatchNorm1d(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            
        Returns:
            Embedded tensor of shape (batch, seq_len, d_model)
        """
        # Transpose for Conv1d: (batch, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm(x)
        x = self.activation(x)
        
        # Transpose back: (batch, seq_len, d_model)
        x = x.transpose(1, 2)
        
        return x


class TransformerEncoderBlock(nn.Module):
    """Single transformer encoder block with attention and feed-forward."""
    
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        d_ff: int, 
        dropout: float = 0.1
    ):
        """
        Initialize transformer encoder block.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Tuple of (output tensor, attention weights)
        """
        # Self-attention with residual connection
        attn_output, attn_weights = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        
        return x, attn_weights


class LightCurveTransformer(nn.Module):
    """Complete transformer model for light curve classification."""
    
    def __init__(
        self,
        input_dim: int = 1,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        n_classes: int = 2,
        dropout: float = 0.1,
        max_len: int = 5000,
        use_temporal_encoding: bool = True
    ):
        """
        Initialize light curve transformer.
        
        Args:
            input_dim: Input feature dimension (1 for flux only)
            d_model: Model embedding dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            n_classes: Number of output classes
            dropout: Dropout rate
            max_len: Maximum sequence length
            use_temporal_encoding: Use temporal positional encoding
        """
        super().__init__()
        
        self.d_model = d_model
        self.use_temporal_encoding = use_temporal_encoding
        
        # Convolutional embedding
        self.embedding = ConvEmbedding(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = TemporalPositionalEncoding(d_model, max_len)
        
        # Transformer encoder blocks
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Global pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Store attention weights for visualization
        self.attention_weights = []
        
    def forward(
        self, 
        x: torch.Tensor, 
        timestamps: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            timestamps: Optional observation timestamps (batch, seq_len)
            return_attention: Whether to store attention weights
            
        Returns:
            Class logits of shape (batch, n_classes)
        """
        # Embedding
        x = self.embedding(x)
        x = self.dropout(x)
        
        # Positional encoding
        if self.use_temporal_encoding:
            x = self.pos_encoding(x, timestamps)
        
        # Transformer encoder blocks
        self.attention_weights = []
        for block in self.encoder_blocks:
            x, attn = block(x)
            if return_attention:
                self.attention_weights.append(attn.detach())
        
        # Global average pooling
        # (batch, seq_len, d_model) -> (batch, d_model, seq_len)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)  # (batch, d_model)
        
        # Classification
        logits = self.classifier(x)
        
        return logits
    
    def get_attention_weights(self) -> list:
        """Get stored attention weights from last forward pass."""
        return self.attention_weights


def create_model(config: dict) -> LightCurveTransformer:
    """
    Create model from configuration dictionary.
    
    Args:
        config: Dictionary with model hyperparameters
        
    Returns:
        Initialized LightCurveTransformer model
    """
    model = LightCurveTransformer(
        input_dim=config.get('input_dim', 1),
        d_model=config.get('d_model', 128),
        n_heads=config.get('n_heads', 4),
        n_layers=config.get('n_layers', 4),
        d_ff=config.get('d_ff', 512),
        n_classes=config.get('n_classes', 2),
        dropout=config.get('dropout', 0.1),
        max_len=config.get('max_len', 5000),
        use_temporal_encoding=config.get('use_temporal_encoding', True)
    )
    
    return model


if __name__ == "__main__":
    # Test model
    batch_size = 8
    seq_len = 1000
    input_dim = 1
    
    model = LightCurveTransformer(
        input_dim=input_dim,
        d_model=128,
        n_heads=4,
        n_layers=3,
        n_classes=2
    )
    
    # Create dummy data
    x = torch.randn(batch_size, seq_len, input_dim)
    timestamps = torch.linspace(0, 100, seq_len).unsqueeze(0).expand(batch_size, -1)
    
    # Forward pass
    output = model(x, timestamps, return_attention=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Attention weights stored: {len(model.get_attention_weights())} layers")