"""
Stage 2: Progressive Fine-tuning - Optimize gradually.
Start with Sharpe only, then add turnover, then CVaR.
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
from typing import Dict

from src.features import load_data, build_feature_matrix
from src.models.tcn_policy import create_tcn_policy_model
from src.losses import sharpe_loss, turnover_penalty, cvar_pinball_loss, compute_portfolio_returns
from src.backtest_loop import load_backtest_config
from src.train_policy import prepare_training_data, evaluate_on_backtest, device, set_seeds
from src.metrics import compute_all_metrics


class ProgressiveFineTuner:
    """
    Progressive fine-tuner with curriculum on loss complexity.
    """
    def __init__(self, model: nn.Module, config: dict, bt_config: dict, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.bt_config = bt_config
        
        # Very low LR for fine-tuning
        self.lr = float(config.get("lr", 1e-4))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Loss weights (will be adjusted progressively)
        self.lambda_turn = 0.0  # Start at 0
        self.lambda_cvar = 0.0  # Start at 0
        self.alpha_cvar = config.get("alpha_cvar", 0.95)
        
        self.train_losses = []
        self.val_sharpes = []
    
    def train_step(self, X_batch, y_batch, weights_prev_batch):
        """Single training step."""
        self.model.train()
        
        logits = self.model(X_batch)
        weights = F.softmax(logits, dim=-1)
        
        # Project to caps
        max_weight = float(self.bt_config["max_weight_per_asset"])
        weights_clipped = torch.clamp(weights, 0.0, max_weight)
        weights = weights_clipped / (weights_clipped.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Compute portfolio returns
        portfolio_returns = compute_portfolio_returns(
            weights, y_batch,
            transaction_cost_bps=float(self.bt_config["cost_bps_per_side"]),
            weights_prev=weights_prev_batch
        )
        
        # PROGRESSIVE LOSS
        # Start with Sharpe only, then add turnover, then CVaR
        loss_sharpe = sharpe_loss(portfolio_returns, epsilon=1e-6)
        loss_turn = turnover_penalty(weights, weights_prev_batch)
        loss_cvar = cvar_pinball_loss(portfolio_returns, alpha=self.alpha_cvar)
        
        loss = loss_sharpe + self.lambda_turn * loss_turn + self.lambda_cvar * loss_cvar
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item(), weights.detach()
    
    def train_epoch(self, X_train, y_train, batch_size: int):
        """Train one epoch."""
        n_samples = X_train.shape[0]
        n_assets = y_train.shape[1]
        
        epoch_losses = []
        weights_prev = torch.ones(batch_size, n_assets, device=self.device) / n_assets
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            actual_batch_size = end_idx - i
            
            X_batch = torch.tensor(X_train[i:end_idx], dtype=torch.float32, device=self.device)
            y_batch = torch.tensor(y_train[i:end_idx], dtype=torch.float32, device=self.device)
            
            if actual_batch_size < batch_size:
                weights_prev_batch = weights_prev[:actual_batch_size]
            else:
                weights_prev_batch = weights_prev
            
            loss, weights_pred = self.train_step(X_batch, y_batch, weights_prev_batch)
            epoch_losses.append(loss)
            
            if actual_batch_size == batch_size:
                weights_prev = weights_pred
        
        return np.mean(epoch_losses)
    
    def evaluate_sharpe(self, X, y):
        """Simple Sharpe evaluation."""
        from src.train_policy import PolicyTrainer
        # Reuse the evaluation logic
        temp_trainer = PolicyTrainer(self.model, self.config, self.bt_config, self.device)
        return temp_trainer.evaluate_sharpe(X, y)
    
    def train_progressive(self, data_dict: Dict, total_epochs: int = 50):
        """
        Progressive training with 3 phases.
        """
        X_train = data_dict["train"]["X"]
        y_train = data_dict["train"]["y"]
        X_val = data_dict["val"]["X"]
        y_val = data_dict["val"]["y"]
        
        batch_size = self.config.get("batch_size", 64)
        
        print("\n" + "=" * 60)
        print("PROGRESSIVE FINE-TUNING (3 PHASES)")
        print("=" * 60)
        
        best_val_sharpe = -np.inf
        best_weights = None
        patience = 10
        patience_counter = 0
        
        # PHASE 1: Sharpe only (epochs 1-20)
        print("\nPHASE 1: Sharpe Loss Only")
        print("-" * 60)
        self.lambda_turn = 0.0
        self.lambda_cvar = 0.0
        phase1_epochs = min(20, total_epochs // 3)
        
        for epoch in range(phase1_epochs):
            loss = self.train_epoch(X_train, y_train, batch_size)
            val_sharpe = self.evaluate_sharpe(X_val, y_val)
            
            print(f"Epoch {epoch+1}/{phase1_epochs} | Loss: {loss:.6f} | Val Sharpe: {val_sharpe:.4f}")
            
            if val_sharpe > best_val_sharpe:
                best_val_sharpe = val_sharpe
                best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
                print(f"  → New best: {best_val_sharpe:.4f}")
            else:
                patience_counter += 1
        
        # PHASE 2: Add turnover penalty (epochs 21-40)
        print("\nPHASE 2: Sharpe + Turnover")
        print("-" * 60)
        self.lambda_turn = float(self.config.get("lambda_turn", 1.0))
        print(f"  λ_turnover = {self.lambda_turn}")
        phase2_epochs = min(20, total_epochs // 3)
        
        for epoch in range(phase2_epochs):
            loss = self.train_epoch(X_train, y_train, batch_size)
            val_sharpe = self.evaluate_sharpe(X_val, y_val)
            
            print(f"Epoch {phase1_epochs+epoch+1}/{phase1_epochs+phase2_epochs} | Loss: {loss:.6f} | Val Sharpe: {val_sharpe:.4f}")
            
            if val_sharpe > best_val_sharpe:
                best_val_sharpe = val_sharpe
                best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
                print(f"  → New best: {best_val_sharpe:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  → Early stop at phase 2")
                    break
        
        # PHASE 3: Add CVaR (epochs 41-50) - only if not stopped
        if patience_counter < patience:
            print("\nPHASE 3: Sharpe + Turnover + CVaR")
            print("-" * 60)
            self.lambda_cvar = float(self.config.get("lambda_cvar", 0.3))
            print(f"  λ_CVaR = {self.lambda_cvar}")
            phase3_epochs = total_epochs - phase1_epochs - phase2_epochs
            
            for epoch in range(phase3_epochs):
                loss = self.train_epoch(X_train, y_train, batch_size)
                val_sharpe = self.evaluate_sharpe(X_val, y_val)
                
                total_epoch = phase1_epochs + phase2_epochs + epoch + 1
                print(f"Epoch {total_epoch}/{total_epochs} | Loss: {loss:.6f} | Val Sharpe: {val_sharpe:.4f}")
                
                if val_sharpe > best_val_sharpe:
                    best_val_sharpe = val_sharpe
                    best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                    print(f"  → New best: {best_val_sharpe:.4f}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"  → Early stop at phase 3")
                        break
        
        # Restore best
        if best_weights is not None:
            self.model.load_state_dict(best_weights)
        
        print("\n" + "=" * 60)
        print(f"Progressive Fine-tuning Complete")
        print(f"Best Val Sharpe: {best_val_sharpe:.4f}")
        print("=" * 60)
        
        return best_val_sharpe


def main():
    """Progressive fine-tuning from distilled model."""
    print("STAGE 2: PROGRESSIVE FINE-TUNING")
    print("=" * 60)
    
    # Load configs
    bt_cfg = load_backtest_config("configs/backtest.yaml")
    with open("configs/model_policy_distill_v2.yaml", "r") as f:
        model_cfg = yaml.safe_load(f)
    
    set_seeds(bt_cfg["seed"])
    
    # Load data
    print("\nLoading data...")
    prices, rf = load_data()
    returns = prices["adj_close"].pct_change().dropna()
    features, targets = build_feature_matrix(prices, rf, model_cfg["features"])
    
    # Prepare data
    print("Preparing data...")
    data_dict = prepare_training_data(features, targets, bt_cfg, model_cfg)
    
    print(f"  Train: {data_dict['train']['X'].shape[0]} samples")
    print(f"  Val: {data_dict['val']['X'].shape[0]} samples")
    print(f"  Test: {data_dict['test']['X'].shape[0]} samples")
    
    # Create model and load distilled checkpoint
    n_assets = returns.shape[1]
    n_features = data_dict["train"]["X"].shape[2]
    
    print(f"\nLoading distilled model ({n_assets} assets, {n_features} features)...")
    model = create_tcn_policy_model(model_cfg, n_assets, n_features)
    
    checkpoint = torch.load("out/models/tcn_distilled_momentum.pt", map_location=device)
    model.load_state_dict(checkpoint)
    print("✓ Loaded distilled checkpoint (corr=0.765)")
    
    # Progressive fine-tuning
    trainer = ProgressiveFineTuner(model, model_cfg, bt_cfg, device)
    best_val_sharpe = trainer.train_progressive(data_dict, total_epochs=50)
    
    # Evaluate on test
    print("\nEvaluating on test set...")
    test_returns, test_weights = evaluate_on_backtest(
        model, returns, bt_cfg, model_cfg, device, split_name="test"
    )
    test_metrics = compute_all_metrics(test_returns, test_weights, periods_per_year=252)
    
    print("\nTest Metrics (Progressive Fine-tuned):")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Save
    print("\nSaving model...")
    Path("out/models").mkdir(parents=True, exist_ok=True)
    Path("out/reports").mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), "out/models/tcn_progressive_finetuned.pt")
    
    # Save results
    test_metrics_df = pd.DataFrame([test_metrics])
    test_metrics_df.to_csv("out/reports/tcn_progressive_test_metrics.csv", index=False)
    test_returns.to_csv("out/reports/tcn_progressive_test_returns.csv")
    test_weights.to_csv("out/reports/tcn_progressive_test_weights.csv")
    
    print(f"\n✓ Progressive fine-tuning complete!")
    print(f"  Model: out/models/tcn_progressive_finetuned.pt")
    print(f"  Test Sharpe: {test_metrics['sharpe']:.4f}")


if __name__ == "__main__":
    main()

