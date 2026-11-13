"""
Train the TCN policy on 2007-2017, early-stop on validation Sharpe (2018-2020),
then backtest the selected checkpoint on validation and test.

    python -m src.train --seeds 0 1 2 3 4

Writes per-seed daily returns, weights and turnover to results/.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import yaml
import argparse
from typing import Dict, Tuple

from src.features import load_data, build_feature_matrix, create_sequences
from src.models.tcn_policy import create_tcn_policy_model
from src.losses import combined_portfolio_loss, compute_portfolio_returns
from src.backtest_loop import run_backtest, get_rebalance_dates, load_backtest_config
from src.constraints import softmax_weights, project_to_caps


def set_seeds(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    import random
    random.seed(seed)


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
        
        
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(X_train, y_train, batch_size)
            self.train_losses.append(train_loss)
            
            # Validate
            val_sharpe = self.evaluate_sharpe(X_val, y_val)
            self.val_sharpes.append(val_sharpe)
            
            print(f"  epoch {epoch+1:2d}  train loss {train_loss:+.5f}  val Sharpe {val_sharpe:.4f}")
            
            # Early stopping
            if val_sharpe > best_val_sharpe:
                best_val_sharpe = val_sharpe
                patience_counter = 0
                best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self.best_epoch = epoch + 1
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    break
        
        # Restore best weights
        if best_weights is not None:
            self.model.load_state_dict(best_weights)
        
        return best_val_sharpe


def evaluate_on_backtest(model, features, returns_df, config_bt, config_model, device,
                         start, end):
    """
    Weekly backtest of a trained model over [start, end].

    On rebalance date t the model sees the `window_len` feature rows strictly before t
    (features at t-1 use closes up to t-1), matching the engine's timing convention.
    """
    window_len = config_model["window_len"]
    n_assets = returns_df.shape[1]
    max_weight = config_bt["max_weight_per_asset"]
    model.eval()

    def weight_fn(date, hist_returns):
        if date not in features.index:
            return np.ones(n_assets) / n_assets
        i = features.index.get_loc(date)
        window = features.iloc[i - window_len:i].ffill(limit=5).values
        if i < window_len or np.isnan(window).any():
            return np.ones(n_assets) / n_assets
        X = torch.tensor(window[None], dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = model(X).cpu().numpy()[0]
        return project_to_caps(softmax_weights(logits), max_weight=max_weight)

    rebal = get_rebalance_dates(returns_df.index, config_bt["rebalance"], start, end)
    return run_backtest(returns_df, weight_fn, rebal, cost_bps=config_bt["cost_bps_per_side"],
                        eval_start=start, eval_end=end)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--backtest-config", default="configs/backtest.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    bt_cfg = load_backtest_config(args.backtest_config)
    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)
    device = torch.device(args.device)
    out = Path(args.out)
    (out / "runs").mkdir(parents=True, exist_ok=True)

    prices, rf = load_data()
    returns = prices["adj_close"].pct_change().dropna()
    features, targets = build_feature_matrix(prices, rf, model_cfg["features"])
    data = prepare_training_data(features, targets, bt_cfg, model_cfg)
    print(f"samples: train {len(data['train']['X'])}, val {len(data['val']['X'])}, "
          f"features per step {data['train']['X'].shape[2]}")

    log_rows = []
    for seed in args.seeds:
        set_seeds(seed)
        torch.use_deterministic_algorithms(True)
        model = create_tcn_policy_model(model_cfg, returns.shape[1], data["train"]["X"].shape[2])
        trainer = PolicyTrainer(model, model_cfg, bt_cfg, device)
        trainer.best_epoch = 0
        print(f"seed {seed}")
        best = trainer.train(data, epochs=model_cfg["epochs"],
                             early_stopping_patience=model_cfg["early_stopping_patience"])
        for e, (l, v) in enumerate(zip(trainer.train_losses, trainer.val_sharpes), 1):
            log_rows.append({"seed": seed, "epoch": e, "train_loss": l, "val_sharpe_daily": v})
        print(f"  selected epoch {trainer.best_epoch} (val Sharpe {best:.4f})")
        for split in ["val", "test"]:
            r, w, t = evaluate_on_backtest(model, features, returns, bt_cfg, model_cfg, device,
                                           bt_cfg[f"{split}_start"], bt_cfg[f"{split}_end"])
            pd.DataFrame({"ret": r, "turnover": t}).to_csv(out / "runs" / f"tcn_seed{seed}_{split}.csv")
            if split == "test":
                w[t.index.isin(get_rebalance_dates(returns.index, bt_cfg["rebalance"],
                   bt_cfg["test_start"], bt_cfg["test_end"]))].round(5).to_csv(
                    out / "runs" / f"tcn_seed{seed}_test_weights.csv")
    pd.DataFrame(log_rows).to_csv(out / "training_log.csv", index=False)


if __name__ == "__main__":
    main()
