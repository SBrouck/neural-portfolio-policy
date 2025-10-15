# Quick Start Guide

## Installation

```bash
# Clone repository
git clone https://github.com/SBrouck/FINNET-Port-weights-opt
cd "FINNET Port wieghts opt"

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
- Python 3.12+
- PyTorch 2.0+
- 2GB RAM minimum
- Apple Silicon / CUDA / CPU supported

---

## Run Best Model (30 seconds)

```bash
# Train TCN Enhanced (our best model)
python3 -m src.train_policy_enhanced

# Expected output:
# ✓ Using Apple Silicon GPU (MPS)
# Training TCN Policy Network...
# Epoch 1/30 | Train Loss: -0.010761 | Val Sharpe: 0.3941
# ...
# ✓ Track A training complete (ENHANCED)!
# Test Sharpe: 0.8245 (naive) / 0.5192 (excess)
```

**Outputs:**
- Model: `out/models/tcn_policy_enhanced.pt`
- Metrics: `out/reports/tcn_policy_enhanced_test_metrics.csv`
- Returns: `out/reports/tcn_policy_enhanced_test_returns.csv`

---

## Compare to Baselines

```bash
# Generate all baselines (Equal Weight, Momentum, Min Var, etc.)
python3 -m src.baselines

# Validate with proper excess Sharpe
python3 -m src.validate_metrics

# Expected output:
# Rank  Strategy        Excess Sharpe
#   1   Momentum            0.5485
#   2   TCN Enhanced        0.5192  ← Our model
#   3   Equal Weight        0.5164
```

**Output:** `out/validation/comparison_excess_sharpe.csv`

---

## Export Results

```bash
# Export all figure data as CSV
python3 -m src.export_figure_data

# Files created in out/figs/:
# - nav_timeseries.csv       (NAV curves)
# - drawdown_timeseries.csv  (Drawdown analysis)
# - rolling_sharpe_tcn.csv   (6-month rolling)
# - weights_matrix_tcn.csv   (1201 × 15 weights)
# - turnover_analysis.csv    (Costs over time)
# - performance_table.csv    (Summary stats)
```

---

## Advanced: Curriculum Learning

### Stage 1: Generate Oracle Labels

```bash
python3 -m src.generate_oracle_labels

# Output: data/oracle/momentum_weights.parquet (660 labels)
```

### Stage 2: Distillation

```bash
python3 -m src.train_distillation

# Expected: Val correlation ~0.76
# Output: out/models/tcn_distilled_momentum.pt
```

### Stage 3: Fine-tuning

```bash
python3 -m src.train_finetune_simple

# Note: Currently overfits (use TCN Enhanced instead)
```

---

## File Locations

### Key Files

| File | Description |
|------|-------------|
| `out/models/tcn_policy_enhanced.pt` | **Best model** (Sharpe 0.519) |
| `out/validation/comparison_excess_sharpe.csv` | **Main results table** |
| `out/reports/tcn_policy_enhanced_test_returns.csv` | Daily returns |
| `out/reports/tcn_policy_enhanced_test_weights.csv` | Portfolio weights |
| `out/figs/nav_timeseries.csv` | NAV data for plotting |

### Config Files

| File | Purpose |
|------|---------|
| `configs/backtest.yaml` | Train/val/test splits, costs |
| `configs/model_policy_v2.yaml` | Enhanced features (150) |
| `configs/model_policy_distill_v2.yaml` | Distillation config |

---

## Interpret Results

### Metrics Explained

**Excess Sharpe** = (Mean return - Risk-free) / Std × √52
- Uses DTB3 (3-month T-Bill) as risk-free
- Annualized for weekly returns
- **Higher is better**

**Turnover** = Average Σ|w_t - w_{t-1}| per rebalance
- As percentage of portfolio value
- **Lower is better** (less trading costs)

**Maximum Drawdown** = Largest peak-to-trough decline
- As percentage
- **Smaller magnitude is better**

**Beta(SPY)** = Systematic risk exposure to S&P 500
- 1.0 = matches SPY
- < 1.0 = lower systematic risk
- **Diversification benefit if < 1**

### Performance Tiers

| Excess Sharpe | Rating |
|---------------|--------|
| > 0.6 | Excellent |
| 0.5 - 0.6 | Good ✅ (TCN Enhanced: 0.519) |
| 0.4 - 0.5 | Acceptable |
| < 0.4 | Poor |

---

## Common Issues

### 1. "No module named 'torch'"

```bash
pip install torch
```

### 2. "Cannot import TensorFlow"

Fixed! Project uses PyTorch now.

### 3. "MPS not available"

```bash
# Check if MPS is available
python3 -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"

# If false, model will use CPU (slower but works)
```

### 4. "Matplotlib architecture error"

Matplotlib has issues - use CSV exports instead:
```bash
python3 -m src.export_figure_data
# Then visualize CSVs in Excel or R
```

---

## Reproduce Paper Results

```bash
# 1. Train best model
python3 -m src.train_policy_enhanced

# 2. Generate baselines
python3 -m src.baselines

# 3. Validate (excess Sharpe)
python3 -m src.validate_metrics

# 4. Export data
python3 -m src.export_figure_data

# Results match Table 1 in paper:
# TCN Enhanced: Rank #2, Excess Sharpe 0.519
```

---

## Customize

### Change Features

Edit `configs/model_policy_v2.yaml`:
```yaml
features:
  - ret_1d
  - ewma_vol
  # Add your custom features here
```

### Adjust Model Capacity

```yaml
hidden_dim: 128  # 64 → 128 for more capacity
tcn_blocks: 3    # 2 → 3 for deeper network
```

### Modify Loss Function

Edit `src/train_policy.py`:
```python
# Change loss weights
lambda_turn: 1.0  # Lower for less turnover penalty
lambda_cvar: 0.0  # Disable CVaR if unstable
```

---

## Support

**Issues:** https://github.com/SBrouck/FINNET-Port-weights-opt/issues  
**Contact:** sbrouck.org I sbrouck@uw.edu

---

**Time to Results:** < 1 minute   
**Rank:** #2 out of 7 strategies
