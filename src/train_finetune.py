"""
Stage 2: Fine-tuning - Optimize end-to-end with portfolio loss.
Starts from distilled model and fine-tunes on Sharpe + turnover + CVaR.
"""
from src.train_policy import *


def main():
    """Fine-tune distilled model end-to-end."""
    print("STAGE 2: FINE-TUNING WITH PORTFOLIO LOSS")
    print("=" * 60)
    
    # Load configs
    bt_cfg = load_backtest_config("configs/backtest.yaml")
    with open("configs/model_policy_v2.yaml", "r") as f:
        model_cfg = yaml.safe_load(f)
    
    set_seeds(bt_cfg["seed"])
    
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
    
    # Create model and LOAD DISTILLED WEIGHTS
    n_assets = returns.shape[1]
    n_features = data_dict["train"]["X"].shape[2]
    
    print(f"\nCreating TCN model and loading distilled weights...")
    model = create_tcn_policy_model(model_cfg, n_assets, n_features)
    
    # Load distilled checkpoint
    try:
        checkpoint = torch.load("out/models/tcn_distilled_momentum.pt", map_location=device)
        model.load_state_dict(checkpoint)
        print("✓ Loaded distilled model checkpoint")
    except Exception as e:
        print(f"⚠️  Could not load distilled model: {e}")
        print("   Training from scratch instead...")
    
    # Fine-tune with portfolio loss
    print("\nFine-tuning with portfolio loss...")
    trainer = PolicyTrainer(model, model_cfg, bt_cfg, device)
    
    # Use slightly lower learning rate for fine-tuning
    original_lr = model_cfg["lr"]
    model_cfg["lr"] = float(original_lr) * 0.5  # Half the LR
    trainer.optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(model_cfg["lr"])
    )
    print(f"  Using fine-tuning LR: {model_cfg['lr']}")
    
    best_val_sharpe = trainer.train(
        data_dict,
        epochs=model_cfg["epochs"],
        early_stopping_patience=model_cfg["early_stopping_patience"]
    )
    
    # Evaluate on validation
    print("\nEvaluating on validation set...")
    val_returns, val_weights = evaluate_on_backtest(
        model, returns, bt_cfg, model_cfg, device, split_name="val"
    )
    val_metrics = compute_all_metrics(val_returns, val_weights, periods_per_year=252)
    
    print("\nValidation Metrics (Fine-tuned):")
    for key, value in val_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Evaluate on test
    print("\nEvaluating on test set...")
    test_returns, test_weights = evaluate_on_backtest(
        model, returns, bt_cfg, model_cfg, device, split_name="test"
    )
    test_metrics = compute_all_metrics(test_returns, test_weights, periods_per_year=252)
    
    print("\nTest Metrics (Fine-tuned):")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Save results
    print("\nSaving fine-tuned model...")
    Path("out/reports").mkdir(parents=True, exist_ok=True)
    Path("out/models").mkdir(parents=True, exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), "out/models/tcn_finetuned.pt")
    
    # Save metrics
    val_metrics_df = pd.DataFrame([val_metrics])
    val_metrics_df.to_csv("out/reports/tcn_finetuned_val_metrics.csv", index=False)
    
    test_metrics_df = pd.DataFrame([test_metrics])
    test_metrics_df.to_csv("out/reports/tcn_finetuned_test_metrics.csv", index=False)
    
    # Save returns and weights
    test_returns.to_csv("out/reports/tcn_finetuned_test_returns.csv")
    test_weights.to_csv("out/reports/tcn_finetuned_test_weights.csv")
    
    # Save config provenance
    provenance = {
        "timestamp": datetime.now().isoformat(),
        "model": "TCN_Policy_Finetuned",
        "stage": "fine-tuning",
        "initialized_from": "distilled_momentum",
        "backtest_config": bt_cfg,
        "model_config": model_cfg,
        "best_val_sharpe": float(best_val_sharpe),
        "versions": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "device": str(device)
    }
    
    with open("out/reports/tcn_finetuned_provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)
    
    print("\n✓ Fine-tuning complete!")
    print(f"  Model: out/models/tcn_finetuned.pt")
    print(f"  Test Sharpe: {test_metrics['sharpe']:.4f}")


if __name__ == "__main__":
    main()

