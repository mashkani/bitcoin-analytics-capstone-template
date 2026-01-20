# Bitcoin DCA Weight Computation Model - Example 2

This document explains the technical indicator model that computes dynamic DCA (Dollar Cost Averaging) weights for Bitcoin investment strategies based on the 200-day moving average and a redistribution logic.

## Overview

The model computes daily investment weights for a given investment window (typically 12 months). It uses the 200-day simple moving average (SMA) as a primary signal to identify periods where Bitcoin is potentially undervalued.

**Key Properties:**

- **Signal**: Distance from the 200-day Moving Average (MA200).
- **Redistribution**: When a day's weight is boosted due to its distance below the MA200, the "excess" weight is taken from the future days in the last half of the window.
- **Constraints**: Weights sum to exactly 1.0, and no weight falls below `MIN_WEIGHT` (1e-6).

## Model Architecture

The model follows a straightforward process:

1. **Feature Construction**: Calculate `ma200` and `std200` based on past prices.
2. **Initial Allocation**: Start with a uniform distribution (equal weights).
3. **Signal Evaluation**: For each day, check if price is below `ma200`.
4. **Weight Boosting**: If below MA200, calculate a Z-score and boost the weight.
5. **Redistribution**: Reduce the weights of days in the second half of the 12-month window to compensate for the boost, ensuring the total sum remains 1.0.

### Mathematical Formulation

Given a window of length $N$:

1. $base\_weight = 1/N$
2. For each day $t$:
   If $price_t < MA200_t$:
   - $Z_t = \frac{MA200_t - price_t}{std200_t}$
   - $boosted\_weight_t = weight_t \times (1 + \alpha \times Z_t)$
   - $excess = boosted\_weight_t - weight_t$
   - Redistribute $excess$ uniformly over indices in the range $[\max(N/2, t+1), N)$.

## Implementation Details

### Feature Engineering

```python
def construct_features(df):
    past_price = df['PriceUSD_coinmetrics'].shift(1)
    df['ma200'] = past_price.rolling(window=200, min_periods=1).mean()
    df['std200'] = past_price.rolling(window=200, min_periods=1).std()
    return df
```

### Weight Computation Logic

The core of the strategy is the `compute_weights` function which iterates through each day and applies the boost/redistribution logic.

| Parameter | Value | Description |
|-----------|-------|-------------|
| `boost_alpha` | 1.25 | Sensitivity multiplier for the Z-score boost |
| `MA_WINDOW` | 200 | Moving average lookback window |
| `MIN_WEIGHT` | 1e-6 | Minimum weight floor to ensure daily participation |

## Comparison with Example 1

While Example 1 uses a complex combination of MVRV, Momentum, and Sentiment, Example 2 focuses on a pure technical indicator (MA200) with a unique budget redistribution mechanism. This approach is more conservative and relies on mean reversion to the long-term trend.
