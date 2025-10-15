"""
Stage 1: Distillation Training - Learn to imitate Momentum oracle.
Uses cosine similarity loss to match predicted weights to oracle weights.
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

from src.features import load_data, build_feature_matrix, create_sequences
from src.models.tcn_policy import create_tcn_policy_model
from src.backtest_loop import load_backtest_config
from src.generate_oracle_labels import align_oracle_to_sequences

# Device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✓ Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def cosine_distance_loss(pred_weights: torch.Tensor, 
                         target_weights: torch.Tensor) -> torch.Tensor:
    """
    Cosine distance loss for weight matching.
    Loss = 1 - cosine_similarity
    
    Range: [0, 2] where 0 is perfect match.
    """
    # Normalize
    pred_norm = F.normalize(pred_weights, p=2, dim=1)
    target_norm = F.normalize(target_weights, p=2, dim=1)
    
    # Cosine similarity
    cos_sim = (pred_norm * target_norm).sum(dim=1)
    
    # Distance (to minimize)
    loss = (1 - cos_sim).mean()
    
    return loss


def l1_weight_distance(pred_weights: torch.Tensor,
                      target_weights: torch.Tensor) -> torch.Tensor:
    """
    L1 distance between weight vectors.
    Interpretable as "turnover" if treating target as previous weights.
    """
    return torch.abs(pred_weights - target_weights).sum(dim=1).mean()


class DistillationTrainer:
    """
    Trainer for supervised distillation from oracle.
    """
    def __init__(self, model: nn.Module, config: dict, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(config.get("distill_lr", 1e-3))
        )
        
        # Tracking
        self.train_losses = []
        self.val_corrs = []
    
    def train_step(self, X_batch, target_weights_batch):
        """Single training step."""
        self.model.train()
        
        # Forward pass
        logits = self.model(X_batch)
        
        # Softmax to get weights
        pred_weights = F.softmax(logits, dim=-1)
        
        # Cosine loss
        loss_cosine = cosine_distance_loss(pred_weights, target_weights_batch)
        
        # Optional: Add small L1 penalty for smoothness
        loss_l1 = l1_weight_distance(pred_weights, target_weights_batch)
        
        # Combined loss
        lambda_l1 = self.config.get("lambda_l1", 0.1)
        loss = loss_cosine + lambda_l1 * loss_l1
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item(), pred_weights.detach()
    
    def train_epoch(self, X_train, target_weights_train, batch_size: int):
        """Train for one epoch."""
        n_samples = X_train.shape[0]
        indices = torch.randperm(n_samples)  # Shuffle is OK for supervised learning
        
        epoch_losses = []
        
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:min(i+batch_size, n_samples)]
            
            X_batch = torch.tensor(X_train[batch_indices], dtype=torch.float32, device=self.device)
            target_batch = torch.tensor(target_weights_train[batch_indices], dtype=torch.float32, device=self.device)
            
            loss, _ = self.train_step(X_batch, target_batch)
            epoch_losses.append(loss)
        
        return np.mean(epoch_losses)
    
    def evaluate_correlation(self, X_val, target_weights_val):
        """
        Evaluate correlation between predicted and target weights.
        Target: correlation > 0.9 for good distillation.
        """
        self.model.eval()
        
        with torch.no_grad():
            X_val_t = torch.tensor(X_val, dtype=torch.float32, device=self.device)
            logits = self.model(X_val_t)
            pred_weights = F.softmax(logits, dim=-1).cpu().numpy()
        
        # Compute correlation per sample, then average
        correlations = []
        for i in range(len(pred_weights)):
            corr = np.corrcoef(pred_weights[i], target_weights_val[i])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
        
        avg_corr = np.mean(correlations) if correlations else 0.0
        
        return avg_corr
    
    def train(self, X_train, target_train, X_val, target_val, epochs: int, batch_size: int):
        """
        Train distillation model.
        """
        print("\nTraining Distillation Model...")
        print("=" * 60)
        print(f"Train samples: {len(X_train)}")
        print(f"Val samples: {len(X_val)}")
        
        best_corr = -1.0
        best_weights = None
        patience = self.config.get("distill_patience", 10)
        patience_counter = 0
        
        import time
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch(X_train, target_train, batch_size)
            self.train_losses.append(train_loss)
            
            # Validate
            val_corr = self.evaluate_correlation(X_val, target_val)
            self.val_corrs.append(val_corr)
            
            epoch_time = time.time() - epoch_start
            
            print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.6f} | Val Corr: {val_corr:.4f} | Time: {epoch_time:.1f}s")
            
            # Track best
            if val_corr > best_corr:
                best_corr = val_corr
                patience_counter = 0
                best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                print(f"  → New best correlation: {best_corr:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  → Early stopping after {epoch+1} epochs")
                    break
        
        # Restore best
        if best_weights is not None:
            self.model.load_state_dict(best_weights)
        
        print("=" * 60)
        print(f"Distillation complete. Best Val Correlation: {best_corr:.4f}")
        
        if best_corr >= 0.9:
            print("✅ TARGET ACHIEVED: Correlation > 0.9")
        elif best_corr >= 0.8:
            print("⚠️  Good progress but below 0.9 target")
        else:
            print("❌ Correlation too low - may need more capacity or better features")
        
        return best_corr


def main():
    """Main distillation training."""
    print("STAGE 1: DISTILLATION FROM MOMENTUM ORACLE (V2 - IMPROVED)")
    print("=" * 60)
    
    # Load configs
    bt_cfg = load_backtest_config("configs/backtest.yaml")
    with open("configs/model_policy_distill_v2.yaml", "r") as f:
        model_cfg = yaml.safe_load(f)
    
    print(f"Model capacity: {model_cfg['hidden_dim']} hidden, {model_cfg['tcn_blocks']} blocks")
    print(f"Distillation LR: {model_cfg['distill_lr']}")
    print(f"Max epochs: {model_cfg['distill_epochs']}")
    
    # Load data
    print("\nLoading data and features...")
    prices, rf = load_data()
    returns = prices["adj_close"].pct_change().dropna()
    features, targets = build_feature_matrix(prices, rf, model_cfg["features"])
    
    # Load oracle weights
    print("Loading oracle labels...")
    oracle_weights = pd.read_parquet("data/oracle/momentum_weights.parquet")
    print(f"  Oracle dates: {len(oracle_weights)}")
    
    # Create sequences
    print("\nPreparing sequences...")
    X_train, y_train, dates_train = create_sequences(
        features, targets, model_cfg["window_len"],
        bt_cfg["train_start"], bt_cfg["train_end"]
    )
    
    X_val, y_val, dates_val = create_sequences(
        features, targets, model_cfg["window_len"],
        bt_cfg["val_start"], bt_cfg["val_end"]
    )
    
    print(f"  Train: {X_train.shape}")
    print(f"  Val: {X_val.shape}")
    
    # Align oracle to sequences
    print("\nAligning oracle to sequences...")
    oracle_train = align_oracle_to_sequences(oracle_weights, dates_train)
    oracle_val = align_oracle_to_sequences(oracle_weights, dates_val)
    
    print(f"  Oracle train: {oracle_train.shape}")
    print(f"  Oracle val: {oracle_val.shape}")
    
    # Create model
    n_assets = returns.shape[1]
    n_features = X_train.shape[2]
    
    print(f"\nCreating TCN model ({n_assets} assets, {n_features} features)...")
    model = create_tcn_policy_model(model_cfg, n_assets, n_features)
    
    # Train distillation
    trainer = DistillationTrainer(model, model_cfg, device)
    best_corr = trainer.train(
        X_train, oracle_train,
        X_val, oracle_val,
        epochs=model_cfg.get("distill_epochs", 100),
        batch_size=128
    )
    
    # Save distilled model
    print("\nSaving distilled model...")
    Path("out/models").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), "out/models/tcn_distilled_momentum.pt")
    
    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "stage": "distillation",
        "oracle": "momentum",
        "best_val_correlation": float(best_corr),
        "model_config": model_cfg,
        "n_train_samples": len(X_train),
        "n_val_samples": len(X_val)
    }
    
    with open("out/models/distillation_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\n✓ Distillation stage complete!")
    print(f"  Model: out/models/tcn_distilled_momentum.pt")
    print(f"  Val correlation: {best_corr:.4f}")
    
    return best_corr


if __name__ == "__main__":
    main()

