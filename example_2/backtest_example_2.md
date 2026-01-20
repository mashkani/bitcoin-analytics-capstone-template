# Bitcoin DCA Backtest System - Example 2

This document explains the backtesting framework for the Example 2 strategy.

## Overview

The backtest system validates the Example 2 DCA strategy by comparing its performance against uniform DCA across rolling 1-year investment windows.

**Key Metrics:**

- **Win Rate**: Percentage of windows where the strategy outperforms uniform DCA.
- **SPD Percentile**: Performance normalized within each window's price range.
- **Model Score**: A balanced metric of win rate and recent performance.

## Core Process

1. **Load Data**: BTC price data is loaded from CoinMetrics.
2. **Precompute Features**: Calculate the 200-day Moving Average and Standard Deviation.
3. **Rolling Windows**: Iterate through 1-year windows, starting from 2018-01-01.
4. **Compute SPD**: Calculate Sats-per-Dollar for both the dynamic strategy and uniform DCA.
5. **Validation**: Check for forward-leakage and ensure all weights sum to 1.0.

## Differences from Example 1

- **Feature Set**: Uses only price-based technical indicators (MA200).
- **Weight Strategy**: Uses a redistribution mechanism instead of an exponential multiplier.
- **Validation**: Includes same rigorous forward-leakage and weight sum checks.

## Interpreting Outputs

The backtest generates several visualizations in the `output/` directory:

- `performance_comparison.svg`: Visualizes the percentile performance over time.
- `win_loss_comparison.svg`: Shows the ratio of windows where the strategy beat the baseline.
- `metrics_example_2.json`: Contains raw performance data for further analysis.
