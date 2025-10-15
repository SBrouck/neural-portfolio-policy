"""
Stage 2: Simple Fine-tuning - Sharpe + Turnover only (NO CVaR).
CVaR term is unstable - skip it.
"""
from src.train_finetune_progressive import *


def main():
    """Simple 2-phase fine-tuning (no CVaR)."""
    print("STAGE 2: SIMPLE FINE-TUNING (Sharpe + Turnover, NO CVaR)")
    print("=" * 60)
    
    # Load configs
    bt_cfg = load_backtest_config("configs/backtest.yaml")
    with open("configs/model_policy_distill_v2.yaml", "r") as f:
        model_cfg = yaml.safe_load(f)
    
    # Override: NO CVaR
    model_cfg["lambda_cvar"] = 0.0
    
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
    
    # Load distilled model
    n_assets = returns.shape[1]
    n_features = data_dict["train"]["X"].shape[2]
    
    print(f"\nLoading distilled model...")
    model = create_tcn_policy_model(model_cfg, n_assets, n_features)
    checkpoint = torch.load("out/models/tcn_distilled_momentum.pt", map_location=device)
    model.load_state_dict(checkpoint)
    print("✓ Loaded (corr=0.765)")
    
    # Train with Sharpe + Turnover only
    trainer = ProgressiveFineTuner(model, model_cfg, bt_cfg, device)
    
    X_train = data_dict["train"]["X"]
    y_train = data_dict["train"]["y"]
    X_val = data_dict["val"]["X"]
    y_val = data_dict["val"]["y"]
    
    batch_size = 64
    best_val_sharpe = -np.inf
    best_weights = None
    patience = 10
    patience_counter = 0
    
    print("\n" + "=" * 60)
    print("TRAINING: 2-Phase (Sharpe only → Sharpe+Turnover)")
    print("=" * 60)
    
    # PHASE 1: Sharpe only (20 epochs)
    print("\nPHASE 1: Sharpe Only")
    print("-" * 60)
    trainer.lambda_turn = 0.0
    trainer.lambda_cvar = 0.0
    
    for epoch in range(20):
        loss = trainer.train_epoch(X_train, y_train, batch_size)
        val_sharpe = trainer.evaluate_sharpe(X_val, y_val)
        
        print(f"Epoch {epoch+1}/20 | Loss: {loss:.6f} | Val Sharpe: {val_sharpe:.4f}")
        
        if val_sharpe > best_val_sharpe:
            best_val_sharpe = val_sharpe
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"  → New best: {best_val_sharpe:.4f}")
        else:
            patience_counter += 1
    
    # PHASE 2: Add turnover (30 more epochs)
    print("\nPHASE 2: Sharpe + Turnover")
    print("-" * 60)
    trainer.lambda_turn = 1.0
    print(f"  λ_turnover = {trainer.lambda_turn}")
    patience_counter = 0
    
    for epoch in range(30):
        loss = trainer.train_epoch(X_train, y_train, batch_size)
        val_sharpe = trainer.evaluate_sharpe(X_val, y_val)
        
        print(f"Epoch {21+epoch}/50 | Loss: {loss:.6f} | Val Sharpe: {val_sharpe:.4f}")
        
        if val_sharpe > best_val_sharpe:
            best_val_sharpe = val_sharpe
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"  → New best: {best_val_sharpe:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  → Early stop")
                break
    
    # Restore best
    if best_weights is not None:
        model.load_state_dict(best_weights)
    
    print("\n" + "=" * 60)
    print(f"Fine-tuning Complete. Best Val Sharpe: {best_val_sharpe:.4f}")
    print("=" * 60)
    
    # Evaluate on test
    print("\nEvaluating on test set...")
    test_returns, test_weights = evaluate_on_backtest(
        model, returns, bt_cfg, model_cfg, device, split_name="test"
    )
    test_metrics = compute_all_metrics(test_returns, test_weights, periods_per_year=252)
    
    print("\nTest Metrics (Simple Fine-tuned):")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Save
    Path("out/models").mkdir(parents=True, exist_ok=True)
    Path("out/reports").mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), "out/models/tcn_simple_finetuned.pt")
    test_metrics_df = pd.DataFrame([test_metrics])
    test_metrics_df.to_csv("out/reports/tcn_simple_test_metrics.csv", index=False)
    test_returns.to_csv("out/reports/tcn_simple_test_returns.csv")
    test_weights.to_csv("out/reports/tcn_simple_test_weights.csv")
    
    print(f"\n✓ Simple fine-tuning complete!")
    print(f"  Model: out/models/tcn_simple_finetuned.pt")
    print(f"  Test Sharpe: {test_metrics['sharpe']:.4f}")


if __name__ == "__main__":
    main()

