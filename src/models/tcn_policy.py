"""
TCN-based policy network for direct portfolio weight learning.
Uses temporal convolutional layers with optional attention for sequence modeling.
PyTorch implementation.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class TemporalBlock(nn.Module):
    """
    Temporal Convolutional Block with dilated causal convolutions.
    """
    def __init__(self, n_inputs: int, n_filters: int, kernel_size: int, dilation_rate: int, dropout: float = 0.1):
        super().__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        
        # Causal padding (we'll do manual padding)
        self.pad_size = (kernel_size - 1) * dilation_rate
        
        self.conv1 = nn.Conv1d(
            in_channels=n_inputs,
            out_channels=n_filters,
            kernel_size=kernel_size,
            dilation=dilation_rate,
            padding=0  # We'll add padding manually for causal
        )
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            in_channels=n_filters,
            out_channels=n_filters,
            kernel_size=kernel_size,
            dilation=dilation_rate,
            padding=0
        )
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection (projection if needed)
        self.downsample = None
        if n_inputs != n_filters:
            self.downsample = nn.Conv1d(n_inputs, n_filters, kernel_size=1)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x shape: (batch, channels, time)
        
        # First conv with causal padding
        out = F.pad(x, (self.pad_size, 0))
        out = self.conv1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        
        # Second conv with causal padding
        out = F.pad(out, (self.pad_size, 0))
        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        
        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        
        return self.relu(out + res)


class AttentionLayer(nn.Module):
    """Simple attention mechanism over time steps."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = np.sqrt(hidden_dim)
    
    def forward(self, x):
        # x shape: (batch, time, features)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(scores, dim=-1)
        
        output = torch.matmul(attention_weights, v)
        return output


class TCNPolicyNetwork(nn.Module):
    """
    TCN-based policy network for portfolio optimization.
    
    Input: (batch, window_len, n_features)
    Output: (batch, n_assets) logits for portfolio weights
    """
    def __init__(self,
                 n_assets: int,
                 n_features: int,
                 hidden_dim: int = 64,
                 num_blocks: int = 2,
                 kernel_size: int = 5,
                 dropout: float = 0.1,
                 use_attention: bool = True):
        super().__init__()
        
        self.n_assets = n_assets
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.use_attention = use_attention
        
        # Initial projection
        self.input_proj = nn.Linear(n_features, hidden_dim)
        
        # TCN blocks with increasing dilation
        self.tcn_blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation = 2 ** i
            n_inputs = hidden_dim  # All blocks have same channel size
            block = TemporalBlock(
                n_inputs=n_inputs,
                n_filters=hidden_dim,
                kernel_size=kernel_size,
                dilation_rate=dilation,
                dropout=dropout
            )
            self.tcn_blocks.append(block)
        
        # Optional attention
        if use_attention:
            self.attention = AttentionLayer(hidden_dim)
        
        # Output projection to logits
        self.output_dense1 = nn.Linear(hidden_dim, hidden_dim)
        self.output_dropout = nn.Dropout(dropout)
        self.output_dense2 = nn.Linear(hidden_dim, n_assets)  # Logits for weights
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x shape: (batch, window_len, n_features)
        
        # Initial projection
        x = self.input_proj(x)  # (batch, window_len, hidden_dim)
        x = self.relu(x)
        
        # TCN blocks expect (batch, channels, time)
        x = x.transpose(1, 2)  # (batch, hidden_dim, window_len)
        
        for block in self.tcn_blocks:
            x = block(x)
        
        # Back to (batch, time, channels) for attention
        x = x.transpose(1, 2)  # (batch, window_len, hidden_dim)
        
        # Attention over time
        if self.use_attention:
            x = self.attention(x)
        
        # Take last time step
        x = x[:, -1, :]  # (batch, hidden_dim)
        
        # Output layers
        x = self.output_dense1(x)
        x = self.relu(x)
        x = self.output_dropout(x)
        logits = self.output_dense2(x)  # (batch, n_assets)
        
        return logits
    
    def predict_weights(self, x, max_weight: float = 0.20):
        """
        Predict portfolio weights with constraints.
        
        Args:
            x: Input features (batch, window_len, n_features)
            max_weight: Maximum weight per asset
            
        Returns:
            weights: (batch, n_assets) with constraints applied
        """
        self.eval()
        with torch.no_grad():
            logits = self(x)
            
            # Softmax to get initial weights
            weights = F.softmax(logits, dim=-1)
            
            # Apply cap constraint
            weights_np = weights.cpu().numpy()
            
            # Project to caps
            from src.constraints import project_to_caps
            projected = np.array([project_to_caps(w, max_weight) for w in weights_np])
            
            return projected


def create_tcn_policy_model(config: dict, n_assets: int, n_features: int) -> TCNPolicyNetwork:
    """
    Create TCN policy model from config.
    
    Args:
        config: Model configuration dict
        n_assets: Number of assets
        n_features: Number of features
        
    Returns:
        model: TCNPolicyNetwork instance
    """
    model = TCNPolicyNetwork(
        n_assets=n_assets,
        n_features=n_features,
        hidden_dim=config.get("hidden_dim", 64),
        num_blocks=config.get("tcn_blocks", 2),
        kernel_size=config.get("kernel_size", 5),
        dropout=config.get("dropout", 0.1),
        use_attention=config.get("use_attention", True)
    )
    
    return model


def test_tcn_model():
    """Test TCN model construction."""
    batch_size = 32
    window_len = 60
    n_assets = 15
    n_features = 90  # 6 features per asset * 15 assets
    
    # Create model
    model = TCNPolicyNetwork(
        n_assets=n_assets,
        n_features=n_features,
        hidden_dim=64,
        num_blocks=2,
        kernel_size=5,
        dropout=0.1,
        use_attention=True
    )
    
    # Test forward pass
    x = torch.randn(batch_size, window_len, n_features)
    model.eval()
    logits = model(x)
    
    print(f"TCN Model Test:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test weight prediction
    weights = model.predict_weights(x, max_weight=0.20)
    print(f"  Predicted weights shape: {weights.shape}")
    print(f"  Weights sum: {weights.sum(axis=1)[:5]}")  # First 5
    print(f"  Max weight: {weights.max():.4f}")
    
    print("\n✓ TCN model test passed!")


if __name__ == "__main__":
    test_tcn_model()
