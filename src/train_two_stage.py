"""
Training script for Track B: Two-stage predict-then-optimize.
Train forecaster, then use with convex allocator for portfolio construction.
PyTorch implementation.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import yaml
import json
from datetime import datetime
from typing import Dict, Tuple

print("⚠ Running with PyTorch (converted from TensorFlow)")
print(f"PyTorch version: {torch.__version__}")

# Use MPS if available (Apple Silicon), otherwise CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✓ Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("✓ Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("✓ Using CPU")

from src.features import load_data, build_feature_matrix, create_sequences
from src.models.two_stage import create_forecaster, ConvexAllocator, TwoStagePortfolio
from src.backtest_loop import run_backtest, get_rebalance_dates, load_backtest_config
from src.metrics import compute_all_metrics


def set_seeds(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    import random
    random.seed(seed)


def load_configs() -> Tuple[dict, dict]:
    """Load backtest and model configs."""
    backtest_cfg = load_backtest_config("configs/backtest.yaml")
    with open("configs/model_two_stage.yaml", "r") as f:
        model_cfg = yaml.safe_load(f)
    return backtest_cfg, model_cfg


def train_forecaster(
    forecaster: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict,
    device: torch.device
) -> float:
    """
    Train return forecaster with MSE loss.
    
    Returns best validation MSE.
    """
    forecaster = forecaster.to(device)
    
    # Setup optimizer and loss
    optimizer = torch.optim.Adam(forecaster.parameters(), lr=float(config["lr"]))
    criterion = nn.MSELoss()
    
    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)
    
    # Training loop
    print("\nTraining Return Forecaster...")
    print("=" * 60)
    
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    patience = 3
    best_val_mse = float('inf')
    patience_counter = 0
    best_weights = None
    
    for epoch in range(epochs):
        # Training
        forecaster.train()
        train_losses = []
        
        # Shuffle training data
        indices = torch.randperm(len(X_train_t))
        
        for i in range(0, len(X_train_t), batch_size):
            batch_indices = indices[i:i+batch_size]
            X_batch = X_train_t[batch_indices]
            y_batch = y_train_t[batch_indices]
            
            # Forward pass
            predictions = forecaster(X_batch)
            loss = criterion(predictions, y_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        train_mse = np.mean(train_losses)
        
        # Validation
        forecaster.eval()
        with torch.no_grad():
            val_predictions = forecaster(X_val_t)
            val_mse = criterion(val_predictions, y_val_t).item()
        
        print(f"Epoch {epoch+1}/{epochs} | Train MSE: {train_mse:.6f} | Val MSE: {val_mse:.6f}")
        
        # Early stopping
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            patience_counter = 0
            best_weights = {k: v.cpu().clone() for k, v in forecaster.state_dict().items()}
            print(f"  → New best Val MSE: {best_val_mse:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  → Early stopping triggered after {epoch+1} epochs")
                break
    
    # Restore best weights
    if best_weights is not None:
        forecaster.load_state_dict(best_weights)
    
    print("=" * 60)
    print(f"Training complete. Best Val MSE: {best_val_mse:.6f}\n")
    
    return best_val_mse


def evaluate_two_stage(
    two_stage: TwoStagePortfolio,
    returns_df: pd.DataFrame,
    features: pd.DataFrame,
    config_bt: dict,
    config_model: dict,
    split_name: str = "test",
    risk_aversion: float = 1.0
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Evaluate two-stage system using backtest engine.
    """
    # Filter to split
    if split_name == "val":
        start, end = config_bt["val_start"], config_bt["val_end"]
    else:
        start, end = config_bt["test_start"], config_bt["test_end"]
    
    returns_split = returns_df[(returns_df.index >= start) & (returns_df.index <= end)]
    
    # Weight function using two-stage system
    window_len = config_model["window_len"]
    n_assets = returns_split.shape[1]
    max_weight = config_bt["max_weight_per_asset"]
    
    def two_stage_weight_function(date, hist_returns):
        """Generate weights using two-stage system."""
        # Find date in features
        if date not in features.index:
            return np.ones(n_assets) / n_assets
        
        date_idx = features.index.get_loc(date)
        
        if date_idx < window_len or len(hist_returns) < 60:
            return np.ones(n_assets) / n_assets
        
        # Extract feature window
        window = features.iloc[date_idx-window_len:date_idx].values
        
        # Get historical returns for covariance (last 252 days)
        lookback = min(252, len(hist_returns))
        hist_ret_array = hist_returns.iloc[-lookback:].values
        
        # Predict weights
        try:
            weights = two_stage.predict_weights(
                window, hist_ret_array, risk_aversion=risk_aversion
            )
            return weights
        except Exception as e:
            print(f"Warning: Two-stage prediction failed at {date}: {e}")
            return np.ones(n_assets) / n_assets
    
    # Get rebalance dates
    rebal_dates = get_rebalance_dates(
        returns_split.index,
        frequency=config_bt["rebalance"],
        start_date=start,
        end_date=end
    )
    
    # Run backtest
    portfolio_returns, weights_history = run_backtest(
        returns_split,
        two_stage_weight_function,
        rebal_dates,
        cost_bps=config_bt["cost_bps_per_side"]
    )
    
    return portfolio_returns, weights_history


def main():
    """Main training pipeline for Track B."""
    # Set seeds
    bt_cfg, model_cfg = load_configs()
    set_seeds(bt_cfg["seed"])
    
    print("Track B: Two-Stage Predict-Then-Optimize")
    print("=" * 60)
    print(f"Config loaded:")
    print(f"  Train: {bt_cfg['train_start']} to {bt_cfg['train_end']}")
    print(f"  Val:   {bt_cfg['val_start']} to {bt_cfg['val_end']}")
    print(f"  Test:  {bt_cfg['test_start']} to {bt_cfg['test_end']}")
    print(f"  Allocator: {model_cfg['allocator']}")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    prices, rf = load_data()
    returns = prices["adj_close"].pct_change().dropna()
    
    # Build features (use same as Track A for fair comparison)
    print("Building features...")
    with open("configs/model_policy.yaml", "r") as f:
        policy_cfg = yaml.safe_load(f)
    
    features, targets = build_feature_matrix(prices, rf, policy_cfg["features"])
    
    # Prepare data splits
    print("Preparing training data...")
    window_len = model_cfg["window_len"]
    
    from src.features import create_sequences
    X_train, y_train, dates_train = create_sequences(
        features, targets, window_len,
        bt_cfg["train_start"], bt_cfg["train_end"]
    )
    
    X_val, y_val, dates_val = create_sequences(
        features, targets, window_len,
        bt_cfg["val_start"], bt_cfg["val_end"]
    )
    
    X_test, y_test, dates_test = create_sequences(
        features, targets, window_len,
        bt_cfg["test_start"], bt_cfg["test_end"]
    )
    
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Val:   {X_val.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")
    
    # Create forecaster
    n_assets = returns.shape[1]
    n_features = X_train.shape[2]
    
    print(f"\nCreating forecaster with {n_assets} assets and {n_features} features...")
    forecaster = create_forecaster(model_cfg, n_assets, n_features)
    
    # Train forecaster
    best_val_mse = train_forecaster(
        forecaster, X_train, y_train, X_val, y_val, model_cfg, device
    )
    
    # Create allocator
    print("\nCreating convex allocator...")
    allocator = ConvexAllocator(
        n_assets=n_assets,
        max_weight=bt_cfg["max_weight_per_asset"],
        shrinkage_alpha=model_cfg["shrinkage_alpha"]
    )
    
    # Create two-stage system
    two_stage = TwoStagePortfolio(forecaster, allocator, window_len)
    
    # Evaluate on validation
    print("\nEvaluating on validation set...")
    val_returns, val_weights = evaluate_two_stage(
        two_stage, returns, features, bt_cfg, model_cfg, split_name="val"
    )
    val_metrics = compute_all_metrics(val_returns, val_weights, periods_per_year=252)
    
    print("\nValidation Metrics:")
    for key, value in val_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Evaluate on test
    print("\nEvaluating on test set...")
    test_returns, test_weights = evaluate_two_stage(
        two_stage, returns, features, bt_cfg, model_cfg, split_name="test"
    )
    test_metrics = compute_all_metrics(test_returns, test_weights, periods_per_year=252)
    
    print("\nTest Metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Save results
    print("\nSaving results...")
    Path("out/reports").mkdir(parents=True, exist_ok=True)
    Path("out/models").mkdir(parents=True, exist_ok=True)
    
    # Save forecaster
    torch.save(forecaster.state_dict(), "out/models/two_stage_forecaster.pt")
    
    # Save metrics
    val_metrics_df = pd.DataFrame([val_metrics])
    val_metrics_df.to_csv("out/reports/two_stage_val_metrics.csv", index=False)
    
    test_metrics_df = pd.DataFrame([test_metrics])
    test_metrics_df.to_csv("out/reports/two_stage_test_metrics.csv", index=False)
    
    # Save returns and weights
    test_returns.to_csv("out/reports/two_stage_test_returns.csv")
    test_weights.to_csv("out/reports/two_stage_test_weights.csv")
    
    # Save config provenance
    provenance = {
        "timestamp": datetime.now().isoformat(),
        "model": "Two_Stage",
        "backtest_config": bt_cfg,
        "model_config": model_cfg,
        "best_val_mse": float(best_val_mse),
        "versions": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "device": str(device)
    }
    
    with open("out/reports/two_stage_provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)
    
    print("\n✓ Track B training complete!")
    print(f"  Results saved to out/reports/")
    print(f"  Model saved to out/models/two_stage_forecaster.pt")


if __name__ == "__main__":
    main()
