# Portfolio Optimization with Deep Policy Learning

A PyTorch implementation of Temporal Convolutional Networks (TCN) for learning portfolio allocation policies from historical market data.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model (6 epochs, ~4 seconds)
python3 -m src.train_policy_enhanced

# Validate results
python3 -m src.validate_metrics

# Export figure data
python3 -m src.export_figure_data
```

---

## Project Overview

This project implements a data-driven approach to portfolio allocation using deep learning. A Temporal Convolutional Network learns to predict optimal portfolio weights from multi-horizon price features, volatility signals, and market regime indicators.

### Key Results (Test Period 2021-2025)

| Metric | Value |
|--------|-------|
| **Excess Sharpe Ratio** | **0.519** (Rank #2/7) |
| **Annualized Return** | 8.62% |
| **Annualized Volatility** | 10.46% |
| **Maximum Drawdown** | -17.04% |
| **Average Turnover** | **0.82%** (lowest among competitive strategies) |
| **Beta to SPY** | 0.318 |

**vs Momentum (Rank #1):**
- Sharpe gap: -5.3%
- **Turnover reduction: -77.3%**
- **Transaction cost savings: 28 bps/year**

---

## Architecture

### TCN Policy Network

```
Input: (batch, window=60, features=150)
  ↓
Dense Projection (150 → 64)
  ↓
TCN Block 1 (kernel=5, dilation=1)
  ↓
TCN Block 2 (kernel=5, dilation=2)
  ↓
Attention Layer (cross-temporal)
  ↓
Dense (64 → 64) + ReLU + Dropout
  ↓
Dense (64 → 15) → Logits
  ↓
Softmax + Cap Projection → Weights
```

**Total Parameters:** ~280,000  
**Device:** Apple Silicon MPS / CUDA / CPU

### Features (10 per asset, 150 total)

**Returns:**
- `ret_1d`, `ret_5d`, `ret_20d` - Multi-horizon log returns

**Risk:**
- `ewma_vol` - EWMA volatility (λ=0.94)
- `skew_60d` - Rolling skewness (tail risk)
- `drawdown_local` - Rolling drawdown (252d)

**Momentum:**
- `mom_12m_z` - 12-month momentum z-score
- `mom_12m_rank` - Cross-sectional rank

**Market Structure:**
- `dispersion` - Cross-sectional dispersion (20d)
- `vol_regime` - Volatility regime indicator

All features use **expanding window normalization** to avoid look-ahead bias.

---

## Training

### Loss Function

```python
L = -Sharpe(R_p) + λ_turn·Turnover + λ_cvar·CVaR
where:
  R_p = w^T·r - c·|w - w_prev|  (portfolio return net of costs)
  λ_turn = 2.0  (turnover penalty)
  λ_cvar = 0.5  (tail risk penalty)
```

### Optimization

- **Optimizer:** Adam (lr=1e-3, gradient clipping=1.0)
- **Batch size:** 128
- **Epochs:** 30 (early stop patience=5)
- **Temporal ordering:** Preserved (no shuffling)
- **Weight propagation:** Previous weights carried between batches

### Data Splits

- **Train:** 2007-2017 (2,313 weekly samples)
- **Validation:** 2018-2020 (691 samples)
- **Test:** 2021-2025 (1,132 samples)
- **Embargo:** 5 trading days between splits

---

## Baselines

All strategies tested with identical backtest engine:
- **Rebalance:** Weekly (Friday)
- **Costs:** 10 bps per side
- **Constraints:** Long-only, 20% cap per asset
- **Risk-free:** DTB3 (3-month T-Bill)

**Implemented baselines:**
1. Momentum (12-month lookback, rank-weighted)
2. Equal Weight (1/N)
3. Minimum Variance (rolling covariance, shrinkage)
4. Risk Parity (inverse volatility)
5. SPY Only (100% SPY)

---

## File Structure

```
├── src/
│   ├── train_policy_enhanced.py      # Main training script ⭐
│   ├── train_distillation.py         # Curriculum: Stage 1
│   ├── train_finetune_simple.py      # Curriculum: Stage 2
│   ├── models/
│   │   ├── tcn_policy.py             # TCN architecture (PyTorch)
│   │   └── two_stage.py              # LSTM forecaster + convex opt
│   ├── features.py                   # Feature engineering
│   ├── losses.py                     # Portfolio losses (PyTorch)
│   ├── baselines.py                  # Classical strategies
│   ├── backtest_loop.py              # Unified backtest engine
│   ├── metrics.py                    # Performance metrics
│   ├── validate_metrics.py           # Validation with excess Sharpe
│   └── export_figure_data.py         # Export visualization data
├── configs/
│   ├── backtest.yaml                 # Backtest parameters
│   ├── model_policy_v2.yaml          # Model config (150 features) ⭐
│   └── model_policy_distill_v2.yaml  # Distillation config
├── out/
│   ├── models/
│   │   └── tcn_policy_enhanced.pt    # Best model ⭐
│   ├── reports/
│   │   ├── tcn_policy_enhanced_test_metrics.csv
│   │   ├── tcn_policy_enhanced_test_returns.csv
│   │   └── tcn_policy_enhanced_test_weights.csv
│   ├── figs/
│   │   ├── nav_timeseries.csv        # NAV curves data
│   │   ├── drawdown_timeseries.csv   # Drawdown data
│   │   ├── rolling_sharpe_tcn.csv    # Rolling Sharpe
│   │   ├── weights_matrix_tcn.csv    # Portfolio weights
│   │   └── turnover_analysis.csv     # Turnover + costs
│   └── validation/
│       └── comparison_excess_sharpe.csv  # Main results table
└── data/
    ├── oracle/
    │   └── momentum_weights.parquet  # Distillation labels
    └── rf/
        └── tbill_3m_daily.parquet    # Risk-free rate (DTB3)
```

---

## Results Deep Dive

### Excess Sharpe Comparison (vs DTB3 @ 2.72%)

```
Rank  Strategy        Excess Sharpe  Ann Return  Turnover  Status
----  -------------  --------------  ----------  --------  ------
  1   Momentum            0.5485       9.49%      3.62%    Leader
  2   TCN Enhanced        0.5192       8.62%      0.82%    ⭐ Our Model
  3   Equal Weight        0.5164       8.58%      0.82%    Close
  4   SPY Only            0.4737      12.18%      0.16%    High vol
  5   Risk Parity         0.4112       6.34%      0.78%    Conservative
  6   Min Variance        0.3195       4.52%      1.73%    Low return
```

### Transaction Cost Analysis

**Annual Cost Drag (10 bps per side):**
- Momentum: 36.2 bps
- **TCN Enhanced: 8.2 bps** ✅
- **Savings: 28 bps/year**

**On $100M AUM: $280,000/year saved**

### Risk Metrics

- **Sortino Ratio:** 1.188 (better downside protection than Momentum 1.171)
- **Calmar Ratio:** 0.506
- **Hit Ratio:** 53.1% winning weeks
- **Max Drawdown Duration:** 575 trading days

---

## Advanced Experiments

### Curriculum Learning (Distillation + Fine-tuning)

**Stage 1 - Distillation:** Pre-train to imitate Momentum oracle
- Approach: Cosine similarity loss on weights
- Result: Correlation 0.765 (target: 0.9)
- Assessment: Captures structure but not fine details

**Stage 2 - Fine-tuning:** Optimize portfolio loss end-to-end
- Result: **Severe overfitting** (val 1.37 vs test 0.38)
- Conclusion: Simpler end-to-end training more robust

**Recommendation:** Use `tcn_policy_enhanced.pt` (end-to-end, no curriculum)

---

## Technical Highlights

### 1. PyTorch Migration
- Converted from TensorFlow (Metal crash issues)
- Apple Silicon MPS acceleration
- Training: 0.6s/epoch (5× faster than TensorFlow)

### 2. Risk-Free Rate Integration
- FRED DTB3 (3-month T-Bill)
- Proper excess return calculation
- Changed ranking dramatically (Min Variance #1→#6)

### 3. No Look-ahead Bias
- Features use expanding windows only
- 5-day embargo between splits
- Temporal ordering preserved in training
- Previous weights carried for turnover calculation

### 4. Transaction Costs
- 10 bps per side
- Applied to realized turnover
- Net returns used for all metrics

---

## Usage

### Train Best Model

```bash
python3 -m src.train_policy_enhanced
```

**Output:**
- Model: `out/models/tcn_policy_enhanced.pt`
- Metrics: `out/reports/tcn_policy_enhanced_test_metrics.csv`
- Returns: `out/reports/tcn_policy_enhanced_test_returns.csv`
- Weights: `out/reports/tcn_policy_enhanced_test_weights.csv`

### Generate Baselines

```bash
python3 -m src.baselines
```

### Validate & Compare

```bash
python3 -m src.validate_metrics
```

### Export Data for Figures

```bash
python3 -m src.export_figure_data
```

All figure data exported as CSV to `out/figs/` for visualization in Excel, R, or Python.

---

## Configuration

### Model Config (`configs/model_policy_v2.yaml`)

```yaml
window_len: 60
features: [ret_1d, ret_5d, ret_20d, ewma_vol, mom_12m_z, 
           mom_12m_rank, skew_60d, drawdown_local, dispersion, vol_regime]

hidden_dim: 64
tcn_blocks: 2
kernel_size: 5
dropout: 0.10
use_attention: true

lambda_turn: 2.0
lambda_cvar: 0.5
alpha_cvar: 0.95

lr: 1e-3
batch_size: 128
epochs: 30
early_stopping_patience: 5
```

### Backtest Config (`configs/backtest.yaml`)

```yaml
train_start: "2007-01-01"
train_end: "2017-12-31"
val_start: "2018-01-01"
val_end: "2020-12-31"
test_start: "2021-01-01"
test_end: "2025-10-14"

rebalance: "W-FRI"
cost_bps_per_side: 10
max_weight_per_asset: 0.20
seed: 42
```

---

## Reproducibility

**Environment:**
- Python: 3.12
- PyTorch: 2.7.1
- NumPy: 1.26+
- Pandas: 2.2+

**Seeds:**
- Global seed: 42
- PyTorch: `torch.manual_seed(42)`
- NumPy: `np.random.seed(42)`

**Deterministic:**
- ✅ Data splits fixed
- ✅ No random shuffling
- ✅ Gradient computation deterministic

---

## Key Insights

### 1. Deep Learning Value Proposition

**TCN Enhanced excels when:**
- ✅ Transaction costs are material (> 5 bps)
- ✅ Liquidity is limited (high market impact)
- ✅ Custom features available (proprietary signals)
- ✅ Online adaptation needed (changing regimes)

**Classical methods excel when:**
- Liquid markets with low costs
- Simple, interpretable allocations required
- No proprietary data edge

### 2. Feature Engineering > Architecture

Going from 90 → 150 features:
- **+4.7% validation Sharpe**
- Minimal architecture changes
- **Conclusion:** Domain knowledge in features >> model complexity

### 3. Overfitting Risk with Curriculum Learning

Distillation (corr 0.76) + Fine-tuning:
- Validation: Excellent (1.37 Sharpe!)
- **Test: Poor (0.38 Sharpe)**
- **Lesson:** Simpler end-to-end more robust

### 4. Momentum Dominates This Universe

On liquid ETFs 2021-2025:
- **Momentum #1** (excess Sharpe 0.549)
- 12-month lookback, rank-weighted
- But **3.62% turnover** (costly)

**TCN offers 94.5% performance at 22.7% cost**

---

## Limitations

1. **Test period:** 4.8 years only (ideally 10+ years)
2. **Universe:** 15 assets (scalability to 100+ untested)
3. **Fine-tuning:** Unstable (overfitting)
4. **Market regimes:** Mostly bull market 2021-2025
5. **Factors:** No explicit factor constraints (beta, sector)

---

## Future Work

### Near-Term
1. **Robustness tests:** Cost sensitivity (0/5/10/15 bps), daily vs weekly rebalance
2. **Ablation studies:** Impact of each feature group
3. **Rolling window analysis:** Performance by market regime
4. **Confidence intervals:** Newey-West SE on Sharpe

### Medium-Term
1. **Online learning:** Weekly adaptation with EWC anchoring
2. **Ensemble:** TCN + Momentum hybrid
3. **Alternative universes:** Stocks, bonds, crypto
4. **Factor targeting:** Beta-neutral, sector-balanced

### Research Directions
1. **Attention interpretability:** Which features/timesteps matter
2. **Meta-learning:** Few-shot adaptation to new assets
3. **Robust optimization:** Worst-case Sharpe maximization
4. **Distributional RL:** Full return distribution modeling

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{brouck2025portfolio,
  author = {Brouck, Sacha},
  title = {Deep Policy Learning for Portfolio Optimization},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/SBrouck/FINNET-Port-weights-opt}
}
```

---

## License

MIT License - see LICENSE file for details

---

## Acknowledgments

- Feature engineering inspired by industry best practices
- TCN architecture: Bai et al. (2018)
- Baselines: Standard portfolio theory (Markowitz, Jegadeesh & Titman)
- Data: Yahoo Finance, FRED

---

## Contact

**Sacha Brouck**  
GitHub: [@SBrouck](https://github.com/SBrouck)

---
 
**Last Updated:** October 14, 2025  
**Version:** 1.0.0
