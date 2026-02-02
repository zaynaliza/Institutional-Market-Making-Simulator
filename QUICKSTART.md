# Quick Start Guide

## Installation & Setup

```bash
# Install dependencies
pip install numpy pandas matplotlib seaborn --break-system-packages

```

---

## What Happens When You Run It

### **Stage 1: Simulation**
```
======================================================================
INSTITUTIONAL MARKET MAKING BACKTEST
======================================================================
Steps: 10,000
Annual Volatility: 25.0%
Max Inventory: 100
Adverse Selection Threshold: 0.6
======================================================================
Progress: 100.0%
Backtest complete!
```

### **Stage 2: Performance Summary**
```
======================================================================
PERFORMANCE SUMMARY
======================================================================

📊 RETURNS
  Total PnL:              $29.66
  Sharpe Ratio:           4.32
  Max Drawdown:           -1.82%

💰 PNL ATTRIBUTION
  Spread Capture:         $30.45 (102.7%)
  Inventory P&L:          $0.04 (0.1%)
  Rebates:                $0.43

📈 TRADING ACTIVITY
  Total Trades:           2174
  Avg Edge per Trade:     $0.0140
  Win Rate:               100.0%
  Edge on Informed Flow:  $0.0143

⚠️  RISK METRICS
  Max Inventory:          100 shares
  Avg VPIN:               0.110
  Risk-Off Triggered:     No

🎯 MARKET CONDITIONS
  Normal Regime:          96.7% of time
  High_vol Regime:        2.7% of time
  Trending Regime:        0.6% of time

======================================================================

🔍 STRATEGY ASSESSMENT:
  ✓ Excellent risk-adjusted returns
  ✓ Drawdown well controlled
  ✓ High win rate - good trade selection
  ✓ Positive edge even on informed flow
======================================================================
```

### **Stage 3: Visual Dashboard**
A professional 10-panel dashboard will pop up showing:
- PnL attribution
- Inventory dynamics
- Adverse selection detection
- Volatility clustering
- Drawdown analysis
- Spread management
- Return distribution
- Regime performance
- Order book imbalance
- Trade quality

---

## Understanding Your Results

### **Good Strategy Indicators:**
✅ Sharpe Ratio > 2.0
✅ Max Drawdown < 5%
✅ Win Rate > 55%
✅ Spread Capture > 80% of PnL
✅ Adverse Selection loss < 5%

### **Red Flags:**
❌ Sharpe Ratio < 1.0
❌ Max Drawdown > 10%
❌ Win Rate < 50%
❌ Large negative adverse selection PnL
❌ Risk-off triggered frequently

---

## Customizing Parameters

Edit these in the `Config` class:

```python
# Simulation length
STEPS = 20000  # Double the simulation

# Risk limits
MAX_INVENTORY = 50  # More conservative

# Strategy aggressiveness
INVENTORY_SKEW_STRENGTH = 0.30  # More aggressive mean reversion
ADVERSE_SELECTION_THRESHOLD = 0.5  # More conservative

# Market conditions
ANNUAL_VOL = 0.40  # Test in high volatility
```

---

## Output Files

After running, you'll get:

1. **mm_backtest_results.csv**
   - Complete time series of all metrics
   - Can be loaded into Excel/Python for further analysis

2. **Dashboard Window**
   - Interactive matplotlib plots
   - Can save as PNG/PDF

---

## Typical Use Cases

### **1. Strategy Testing**
```python
# Test different inventory skew strengths
for skew in [0.10, 0.15, 0.20, 0.25, 0.30]:
    Config.INVENTORY_SKEW_STRENGTH = skew
    strategy, df = run_backtest()
    print(f"Skew {skew}: Sharpe = {sharpe:.2f}")
```

### **2. Regime Analysis**
```python
# How does strategy perform in high vol?
high_vol_periods = df[df['regime'] == 'high_vol']
print(f"High Vol Sharpe: {high_vol_periods['total_pnl'].std()}")
```

### **3. Adverse Selection Study**
```python
# Analyze trades by flow type
trades_df = pd.DataFrame(strategy.trade_history)
informed_pnl = trades_df[trades_df['flow_type'] == 'informed']['edge'].sum()
print(f"Lost to informed traders: ${informed_pnl:.2f}")
```

---

## Troubleshooting

**Issue: ImportError**
```bash
# Install missing packages
pip install numpy pandas matplotlib seaborn --break-system-packages
```

**Issue: Plots not showing**
```python
# Add at end of script:
plt.show()
```

**Issue: "Risk-off triggered" immediately**
```python
# Increase max drawdown tolerance
Config.MAX_DRAWDOWN_PCT = 0.10  # 10%
```
