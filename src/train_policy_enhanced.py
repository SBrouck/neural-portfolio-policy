"""
Training with enhanced features - testing feature improvements.
"""
from src.train_policy import *

def main():
    """Main training with enhanced features."""
    # Load enhanced config instead of default
    bt_cfg = load_backtest_config("configs/backtest.yaml")
    with open("configs/model_policy_v2.yaml", "r") as f:
        model_cfg = yaml.safe_load(f)
    
    set_seeds(bt_cfg["seed"])
    
    print("Track A: TCN Policy Network Training (ENHANCED FEATURES)")
    print("=" * 60)
    print(f"Config loaded:")
    print(f"  Train: {bt_cfg['train_start']} to {bt_cfg['train_end']}")
    print(f"  Val:   {bt_cfg['val_start']} to {bt_cfg['val_end']}")
    print(f"  Test:  {bt_cfg['test_start']} to {bt_cfg['test_end']}")
    print(f"  Features: {len(model_cfg['features'])}")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    prices, rf = load_data()
    returns = prices["adj_close"].pct_change().dropna()
    
    # Build enhanced features
    print("Building ENHANCED features...")
    features, targets = build_feature_matrix(prices, rf, model_cfg["features"])
    print(f"  Feature matrix: {features.shape}")
    
    # Prepare data splits
    print("Preparing training data...")
    data_dict = prepare_training_data(features, targets, bt_cfg, model_cfg)
    
    print(f"  Train: {data_dict['train']['X'].shape[0]} samples")
    print(f"  Val:   {data_dict['val']['X'].shape[0]} samples")
    print(f"  Test:  {data_dict['test']['X'].shape[0]} samples")
    if len(data_dict['train']['X'].shape) > 2:
        print(f"  Features per sample: {data_dict['train']['X'].shape[2]}")
    
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
    torch.save(model.state_dict(), "out/models/tcn_policy_enhanced.pt")
    
    # Save metrics
    val_metrics_df = pd.DataFrame([val_metrics])
    val_metrics_df.to_csv("out/reports/tcn_policy_enhanced_val_metrics.csv", index=False)
    
    test_metrics_df = pd.DataFrame([test_metrics])
    test_metrics_df.to_csv("out/reports/tcn_policy_enhanced_test_metrics.csv", index=False)
    
    # Save returns and weights
    test_returns.to_csv("out/reports/tcn_policy_enhanced_test_returns.csv")
    test_weights.to_csv("out/reports/tcn_policy_enhanced_test_weights.csv")
    
    # Save config provenance
    provenance = {
        "timestamp": datetime.now().isoformat(),
        "model": "TCN_Policy_Enhanced",
        "backtest_config": bt_cfg,
        "model_config": model_cfg,
        "best_val_sharpe": float(best_val_sharpe),
        "versions": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "device": str(device),
        "features": model_cfg["features"],
        "n_features_total": n_features
    }
    
    with open("out/reports/tcn_policy_enhanced_provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)
    
    print("\n✓ Track A training complete (ENHANCED)!")
    print(f"  Results saved to out/reports/")
    print(f"  Model saved to out/models/tcn_policy_enhanced.pt")


if __name__ == "__main__":
    main()

