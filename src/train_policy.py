"""
Training script for Track A: Direct policy learning with TCN.
Walk-forward training with early stopping on validation Sharpe.
PyTorch implementation.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import yaml
import json
from datetime import datetime
from typing import Dict, Tuple

print("⚠ Running with PyTorch (converted from TensorFlow)")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS (Apple Silicon) available: {torch.backends.mps.is_available()}")

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
from src.models.tcn_policy import create_tcn_policy_model
from src.losses import combined_portfolio_loss, compute_portfolio_returns
from src.constraints import project_weights_full
from src.backtest_loop import run_backtest, get_rebalance_dates, load_backtest_config
from src.metrics import compute_all_metrics

print("✓ Imports successful")


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
    with open("configs/model_policy.yaml", "r") as f:
        model_cfg = yaml.safe_load(f)
    return backtest_cfg, model_cfg


def prepare_training_data(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    config_bt: dict,
    config_model: dict
) -> Dict:
    """
    Prepare training, validation, and test sets.
    
    Returns dict with X, y, dates for each split.
    """
    window_len = config_model["window_len"]
    
    # Create sequences for each split
    X_train, y_train, dates_train = create_sequences(
        features, targets, window_len,
        config_bt["train_start"], config_bt["train_end"]
    )
    
    X_val, y_val, dates_val = create_sequences(
        features, targets, window_len,
        config_bt["val_start"], config_bt["val_end"]
    )
    
    X_test, y_test, dates_test = create_sequences(
        features, targets, window_len,
        config_bt["test_start"], config_bt["test_end"]
    )
    
    return {
        "train": {"X": X_train, "y": y_train, "dates": dates_train},
        "val": {"X": X_val, "y": y_val, "dates": dates_val},
        "test": {"X": X_test, "y": y_test, "dates": dates_test},
    }


class PolicyTrainer:
    """
    Trainer for TCN policy network.
    """
    def __init__(self, model: nn.Module, config_model: dict, config_bt: dict, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.config_model = config_model
        self.config_bt = config_bt
        
        # Loss hyperparameters
        self.lambda_turn = config_model["lambda_turn"]
        self.lambda_cvar = config_model["lambda_cvar"]
        self.alpha_cvar = config_model["alpha_cvar"]
        
        # Optimizer with gradient clipping
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(config_model["lr"])
        )
        
        # Tracking
        self.train_losses = []
        self.val_sharpes = []
    
    def train_step(self, X_batch, y_batch, weights_prev_batch):
        """Single training step."""
        self.model.train()
        
        # Forward pass
        logits = self.model(X_batch)
        
        # Apply softmax to get weights
        weights = F.softmax(logits, dim=-1)
        
        # Project to caps (match backtest constraints)
        max_weight = float(self.config_bt["max_weight_per_asset"])
        # Clip and renormalize
        weights_clipped = torch.clamp(weights, 0.0, max_weight)
        weights = weights_clipped / (weights_clipped.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Compute portfolio returns
        portfolio_returns = compute_portfolio_returns(
            weights, y_batch,
            transaction_cost_bps=float(self.config_bt["cost_bps_per_side"]),
            weights_prev=weights_prev_batch
        )
        
        # Compute loss
        loss = combined_portfolio_loss(
            portfolio_returns, weights, weights_prev_batch,
            lambda_turn=self.lambda_turn,
            lambda_cvar=self.lambda_cvar,
            alpha_cvar=self.alpha_cvar
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        
        return loss.item(), weights.detach()
    
    def train_epoch(self, X_train, y_train, batch_size: int):
        """Train for one epoch in time order, carrying weights between batches."""
        n_samples = X_train.shape[0]
        n_assets = y_train.shape[1]
        
        epoch_losses = []
        
        # Initialize previous weights to equal weight
        weights_prev = torch.ones(batch_size, n_assets, device=self.device) / n_assets
        
        # Iterate in time order (no shuffling)
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            actual_batch_size = end_idx - i
            
            X_batch = torch.tensor(X_train[i:end_idx], dtype=torch.float32, device=self.device)
            y_batch = torch.tensor(y_train[i:end_idx], dtype=torch.float32, device=self.device)
            
            # Use carried weights from previous batch
            if actual_batch_size < batch_size:
                # Last batch might be smaller
                weights_prev_batch = weights_prev[:actual_batch_size]
            else:
                weights_prev_batch = weights_prev
            
            # Train step returns loss and predicted weights
            loss, weights_pred = self.train_step(X_batch, y_batch, weights_prev_batch)
            epoch_losses.append(loss)
            
            # Carry predicted weights to next batch
            if actual_batch_size == batch_size:
                weights_prev = weights_pred
        
        return np.mean(epoch_losses)
    
    def evaluate_sharpe(self, X, y):
        """Evaluate Sharpe ratio with sequential pass that charges costs properly."""
        if len(X) == 0:
            return 0.0
        
        self.model.eval()
        
        n_samples = len(X)
        n_assets = y.shape[1]
        max_weight = float(self.config_bt["max_weight_per_asset"])
        
        # Sequential pass in time order
        portfolio_returns = []
        weights_prev = torch.ones(n_assets, device=self.device) / n_assets
        
        batch_size = 128
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                X_batch = torch.tensor(X[i:end_idx], dtype=torch.float32, device=self.device)
                y_batch = y[i:end_idx]
                
                # Predict weights
                logits = self.model(X_batch)
                weights = F.softmax(logits, dim=-1)
                
                # Project to caps
                weights_clipped = torch.clamp(weights, 0.0, max_weight)
                weights = weights_clipped / (weights_clipped.sum(dim=-1, keepdim=True) + 1e-8)
                weights_np = weights.cpu().numpy()
                
                # Compute returns with costs vs actual prev weights
                for j in range(len(weights_np)):
                    w = weights_np[j]
                    r = y_batch[j]
                    
                    # Gross return
                    gross_ret = np.dot(w, r)
                    
                    # Transaction cost
                    turnover = np.sum(np.abs(w - weights_prev.cpu().numpy()))
                    cost = (float(self.config_bt["cost_bps_per_side"]) / 10000.0) * turnover
                    
                    # Net return
                    net_ret = gross_ret - cost
                    portfolio_returns.append(net_ret)
                    
                    # Update prev weights for next step
                    weights_prev = torch.tensor(w, device=self.device)
        
        # Sharpe
        returns_np = np.array(portfolio_returns)
        sharpe = (returns_np.mean() / (returns_np.std() + 1e-8)) * np.sqrt(252)
        
        return sharpe
    
    def train(self, data_dict: Dict, epochs: int, early_stopping_patience: int = 3):
        """
        Train with early stopping on validation Sharpe.
        """
        X_train = data_dict["train"]["X"]
        y_train = data_dict["train"]["y"]
        X_val = data_dict["val"]["X"]
        y_val = data_dict["val"]["y"]
        
        batch_size = self.config_model["batch_size"]
        
        best_val_sharpe = -np.inf
        patience_counter = 0
        best_weights = None
        
        print("\nTraining TCN Policy Network...")
        print("=" * 60)
        
        # Log training setup
        steps_per_epoch = (len(X_train) + batch_size - 1) // batch_size
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        import time
        epoch_times = []
        
        for epoch in range(epochs):
            epoch_start = time.time()
            # Train
            train_loss = self.train_epoch(X_train, y_train, batch_size)
            self.train_losses.append(train_loss)
            
            # Validate
            val_sharpe = self.evaluate_sharpe(X_val, y_val)
            self.val_sharpes.append(val_sharpe)
            
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Sharpe: {val_sharpe:.4f} | Time: {epoch_time:.1f}s")
            
            # Early stopping
            if val_sharpe > best_val_sharpe:
                best_val_sharpe = val_sharpe
                patience_counter = 0
                best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                print(f"  → New best Val Sharpe: {best_val_sharpe:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"  → Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Restore best weights
        if best_weights is not None:
            self.model.load_state_dict(best_weights)
        
        print("=" * 60)
        print(f"Training complete. Best Val Sharpe: {best_val_sharpe:.4f}")
        if epoch_times:
            print(f"Average time per epoch: {np.mean(epoch_times):.1f}s")
        print()
        
        return best_val_sharpe


def evaluate_on_backtest(
    model: nn.Module,
    returns_df: pd.DataFrame,
    config_bt: dict,
    config_model: dict,
    device: torch.device,
    split_name: str = "test"
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Evaluate model using backtest engine.
    """
    # Prepare features for the entire period
    prices, rf = load_data()
    features, targets = build_feature_matrix(prices, rf, config_model["features"])
    
    # Filter to split
    if split_name == "val":
        start, end = config_bt["val_start"], config_bt["val_end"]
    else:
        start, end = config_bt["test_start"], config_bt["test_end"]
    
    returns_split = returns_df[(returns_df.index >= start) & (returns_df.index <= end)]
    
    # Weight function using model
    window_len = config_model["window_len"]
    n_assets = returns_split.shape[1]
    max_weight = config_bt["max_weight_per_asset"]
    
    model.eval()
    
    def model_weight_function(date, hist_returns):
        """Generate weights using trained model."""
        # Find date in features
        if date not in features.index:
            # Default to equal weight
            return np.ones(n_assets) / n_assets
        
        date_idx = features.index.get_loc(date)
        
        if date_idx < window_len:
            # Not enough history
            return np.ones(n_assets) / n_assets
        
        # Extract window
        window = features.iloc[date_idx-window_len:date_idx].values
        
        # Predict
        X = torch.tensor(window.reshape(1, window_len, -1), dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = model(X)
        
        # Project to feasible set
        from src.constraints import softmax_weights, project_to_caps
        weights = softmax_weights(logits.cpu().numpy()[0])
        weights = project_to_caps(weights, max_weight=max_weight)
        
        return weights
    
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
        model_weight_function,
        rebal_dates,
        cost_bps=config_bt["cost_bps_per_side"]
    )
    
    return portfolio_returns, weights_history


def main():
    """Main training pipeline for Track A."""
    # Set seeds
    bt_cfg, model_cfg = load_configs()
    set_seeds(bt_cfg["seed"])
    
    print("Track A: TCN Policy Network Training")
    print("=" * 60)
    print(f"Config loaded:")
    print(f"  Train: {bt_cfg['train_start']} to {bt_cfg['train_end']}")
    print(f"  Val:   {bt_cfg['val_start']} to {bt_cfg['val_end']}")
    print(f"  Test:  {bt_cfg['test_start']} to {bt_cfg['test_end']}")
    print(f"  Rebalance: {bt_cfg['rebalance']}")
    print(f"  Cost: {bt_cfg['cost_bps_per_side']} bps")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    prices, rf = load_data()
    returns = prices["adj_close"].pct_change().dropna()
    
    # Build features
    print("Building features...")
    features, targets = build_feature_matrix(prices, rf, model_cfg["features"])
    
    # Prepare data splits
    print("Preparing training data...")
    data_dict = prepare_training_data(features, targets, bt_cfg, model_cfg)
    
    print(f"  Train: {data_dict['train']['X'].shape[0]} samples")
    print(f"  Val:   {data_dict['val']['X'].shape[0]} samples")
    print(f"  Test:  {data_dict['test']['X'].shape[0]} samples")
    
    # Create model
    n_assets = returns.shape[1]
    n_features = data_dict["train"]["X"].shape[2]
    
    print(f"\nCreating TCN model with {n_assets} assets and {n_features} features...")
    model = create_tcn_policy_model(model_cfg, n_assets, n_features)
    
    # Train
    trainer = PolicyTrainer(model, model_cfg, bt_cfg, device)
    best_val_sharpe = trainer.train(
        data_dict,
        epochs=model_cfg["epochs"],
        early_stopping_patience=model_cfg["early_stopping_patience"]
    )
    
    # Evaluate on validation
    print("Evaluating on validation set...")
    val_returns, val_weights = evaluate_on_backtest(
        model, returns, bt_cfg, model_cfg, device, split_name="val"
    )
    val_metrics = compute_all_metrics(val_returns, val_weights, periods_per_year=252)
    
    print("\nValidation Metrics:")
    for key, value in val_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Evaluate on test
    print("\nEvaluating on test set...")
    test_returns, test_weights = evaluate_on_backtest(
        model, returns, bt_cfg, model_cfg, device, split_name="test"
    )
    test_metrics = compute_all_metrics(test_returns, test_weights, periods_per_year=252)
    
    print("\nTest Metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Save results
    print("\nSaving results...")
    Path("out/reports").mkdir(parents=True, exist_ok=True)
    Path("out/models").mkdir(parents=True, exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), "out/models/tcn_policy_model.pt")
    
    # Save metrics
    val_metrics_df = pd.DataFrame([val_metrics])
    val_metrics_df.to_csv("out/reports/tcn_policy_val_metrics.csv", index=False)
    
    test_metrics_df = pd.DataFrame([test_metrics])
    test_metrics_df.to_csv("out/reports/tcn_policy_test_metrics.csv", index=False)
    
    # Save returns and weights
    test_returns.to_csv("out/reports/tcn_policy_test_returns.csv")
    test_weights.to_csv("out/reports/tcn_policy_test_weights.csv")
    
    # Save config provenance
    provenance = {
        "timestamp": datetime.now().isoformat(),
        "model": "TCN_Policy",
        "backtest_config": bt_cfg,
        "model_config": model_cfg,
        "best_val_sharpe": float(best_val_sharpe),
        "versions": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "device": str(device),
        "optimizations": {
            "gradient_clipping": 1.0,
            "temporal_order_preserved": True,
            "caps_projection_in_training": True,
            "sequential_validation": True
        }
    }
    
    with open("out/reports/tcn_policy_provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)
    
    # Save run config
    run_config = {
        "backtest": bt_cfg,
        "model": model_cfg
    }
    with open("out/reports/run_config.yaml", "w") as f:
        yaml.dump(run_config, f, default_flow_style=False)
    
    print("\n✓ Track A training complete!")
    print(f"  Results saved to out/reports/")
    print(f"  Model saved to out/models/tcn_policy_model.pt")


if __name__ == "__main__":
    main()
