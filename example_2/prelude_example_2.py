import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from example_2.model_development_example_2 import precompute_features
except ImportError:
    from model_development_example_2 import precompute_features

try:
    from IPython.display import display
except ImportError:
    display = print

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Configuration
BACKTEST_START = "2018-01-01"
BACKTEST_END = None  # Will use latest available date from data
INVESTMENT_WINDOW = 12  # months (deprecated: use WINDOW_OFFSET for consistency)
PURCHASE_FREQ = "Daily"  # Daily frequency for DCA purchases
# Standard 1-year window used across all modules
WINDOW_OFFSET = pd.DateOffset(years=1)

PURCHASE_FREQ_TO_OFFSET = {"Daily": "1D"}

# Tolerance for weight sum validation (small leniency for floating-point precision)
WEIGHT_SUM_TOLERANCE = 1e-5


def load_data():
    """Load BTC data from CoinMetrics CSV.

    Loads from local file: data/Coin Metrics/coinmetrics_btc.csv
    """
    # Try project root relative to this file
    base_dir = Path(__file__).parent.parent
    local_path = base_dir / "data" / "Coin Metrics" / "coinmetrics_btc.csv"

    # Fallback to CWD relative (for back-compat)
    if not local_path.exists():
        local_path = Path("data/Coin Metrics/coinmetrics_btc.csv")

    if not local_path.exists():
        raise FileNotFoundError(
            f"CoinMetrics BTC data file not found at {local_path}. "
            "Please ensure the file exists in the data/Coin Metrics/ directory."
        )

    logging.info(f"Loading CoinMetrics BTC data from local file: {local_path}")
    df = pd.read_csv(local_path)

    # Set time as index
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    df.index = df.index.normalize().tz_localize(None)

    # Remove duplicates and sort
    df = df.loc[~df.index.duplicated(keep="last")].sort_index()

    # Use PriceUSD column from CoinMetrics (complete 2025 data)
    if "PriceUSD" not in df.columns:
        raise ValueError("PriceUSD column not found in CoinMetrics data")

    # Rename PriceUSD to PriceUSD_coinmetrics for compatibility
    df["PriceUSD_coinmetrics"] = df["PriceUSD"]

    # Assert all dates from BACKTEST_START to the latest available date have BTC-USD prices
    latest_date = df.index.max()
    backtest_dates = df.index[
        (df.index >= pd.to_datetime(BACKTEST_START)) & (df.index <= latest_date)
    ]
    missing_prices = df.loc[backtest_dates, "PriceUSD_coinmetrics"].isnull()
    if missing_prices.any():
        first_missing = df.loc[backtest_dates, "PriceUSD_coinmetrics"][
            missing_prices
        ].index.min()
        logging.warning(
            f"Missing BTC-USD prices for {missing_prices.sum()} dates from {BACKTEST_START} to {latest_date.date()}. "
            f"First missing date: {first_missing.date()}"
        )

    logging.info(
        f"Loaded CoinMetrics data: {len(df)} rows, {df.index.min().date()} to {df.index.max().date()}"
    )
    return df


def _make_window_label(start: pd.Timestamp, end: pd.Timestamp) -> str:
    """Format rolling window label as 'YYYY-MM-DD → YYYY-MM-DD'."""
    return f"{start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')}"


def parse_window_dates(window_label: str) -> pd.Timestamp:
    """Extract start date from window label like '2016-01-01 → 2017-01-01'.

    Args:
        window_label: Window label in format 'YYYY-MM-DD → YYYY-MM-DD'

    Returns:
        Start date as pandas Timestamp
    """
    return pd.to_datetime(window_label.split(" → ")[0])


def generate_date_ranges(
    range_start: str, range_end: str, min_length_days: int = 120
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Generate date ranges where each start_date has an end_date exactly 1 year later.

    Uses DATE_FREQ (daily) for start date generation.
    Each start_date is paired with exactly one end_date that is 1 year later.
    # Uses WINDOW_OFFSET from prelude_template.py for consistency across modules.

    Args:
        range_start: Start of the date range (YYYY-MM-DD format)
        range_end: End of the date range (YYYY-MM-DD format)
        min_length_days: Minimum range length in days (default 120)

    Returns:
        List of (start_date, end_date) tuples
    """
    start, end = pd.to_datetime(range_start), pd.to_datetime(range_end)

    # Generate all possible start dates from range_start to range_end - 1 year
    max_start_date = end - WINDOW_OFFSET
    start_dates = pd.date_range(start=start, end=max_start_date, freq="D")

    # For each start date, set end_date to exactly 1 year later
    date_ranges = []
    for start_date in start_dates:
        end_date = start_date + WINDOW_OFFSET
        # Only include if end_date is within range_end
        if end_date <= end:
            date_ranges.append((start_date, end_date))

    return date_ranges


def group_ranges_by_start_date(
    date_ranges: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> dict[pd.Timestamp, list[pd.Timestamp]]:
    """Group list of (start, end) tuples by start_date.

    Args:
        date_ranges: List of (start_date, end_date) tuples

    Returns:
        Dictionary mapping start_date -> list of end_dates
    """
    grouped: dict[pd.Timestamp, list[pd.Timestamp]] = {}
    for start, end in date_ranges:
        if start not in grouped:
            grouped[start] = []
        grouped[start].append(end)
    return grouped


def compute_cycle_spd(
    dataframe: pd.DataFrame,
    strategy_function,
    features_df: pd.DataFrame | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    validate_weights: bool = True,
) -> pd.DataFrame:
    """Compute sats-per-dollar (SPD) statistics over rolling windows.

    Uses precomputed features for efficient computation. Uses 1-year windows for consistency.

    Args:
        dataframe: DataFrame containing price data with 'PriceUSD_coinmetrics' column
        strategy_function: Function that takes features DataFrame and returns weights
        features_df: Optional precomputed features. If None, computes them internally.
        start_date: Optional start date (default: BACKTEST_START)
        end_date: Optional end date (default: BACKTEST_END)
        validate_weights: Whether to validate that weights sum to 1.0 (default: True)

    Returns:
        DataFrame with SPD statistics indexed by window label
    """
    start = start_date or BACKTEST_START
    end = end_date or BACKTEST_END
    if end is None:
        end = dataframe.index.max().strftime("%Y-%m-%d")

    # Use provided features or compute them
    if features_df is None:
        full_feat = precompute_features(dataframe).loc[start:end]
    else:
        full_feat = features_df.loc[start:end]

    window_offset = WINDOW_OFFSET

    # Generate start dates daily
    max_start_date = pd.to_datetime(end) - window_offset
    start_dates = pd.date_range(
        start=pd.to_datetime(start),
        end=max_start_date,
        freq="D",  # Daily frequency for consistency
    )

    if len(start_dates) > 0:
        actual_end_date = (start_dates[-1] + window_offset).date()
        logging.info(
            f"Backtesting date range: {start_dates[0].date()} to {actual_end_date} "
            f"({len(start_dates)} total windows)"
        )

    results = []
    validated_windows = 0
    for window_start in start_dates:
        window_end = window_start + window_offset

        # Only include if end_date is within range
        if window_end > pd.to_datetime(end):
            continue

        price_slice = dataframe["PriceUSD_coinmetrics"].loc[window_start:window_end]
        if price_slice.empty:
            continue

        # Compute weights using strategy_function
        window_feat = full_feat.loc[window_start:window_end]
        weight_slice = strategy_function(window_feat)

        # Validate weights sum to 1.0 if requested
        if validate_weights:
            weight_sum = weight_slice.sum()
            assert np.isclose(weight_sum, 1.0, atol=WEIGHT_SUM_TOLERANCE), (
                f"Weights for range {window_start.date()} to {window_end.date()} "
                f"sum to {weight_sum:.10f}, expected 1.0 (tolerance: {WEIGHT_SUM_TOLERANCE})"
            )
            validated_windows += 1

        inv_price = 1e8 / price_slice  # sats per dollar
        min_spd, max_spd = inv_price.min(), inv_price.max()
        span = max_spd - min_spd
        uniform_spd = inv_price.mean()
        dynamic_spd = (weight_slice * inv_price).sum()

        # Handle edge case where span is zero (all prices identical)
        if span > 0:
            uniform_pct = (uniform_spd - min_spd) / span * 100
            dynamic_pct = (dynamic_spd - min_spd) / span * 100
        else:
            # When all prices are identical, percentile is undefined
            uniform_pct = float("nan")
            dynamic_pct = float("nan")

        results.append(
            {
                "window": _make_window_label(window_start, window_end),
                "min_sats_per_dollar": min_spd,
                "max_sats_per_dollar": max_spd,
                "uniform_sats_per_dollar": uniform_spd,
                "dynamic_sats_per_dollar": dynamic_spd,
                "uniform_percentile": uniform_pct,
                "dynamic_percentile": dynamic_pct,
                "excess_percentile": dynamic_pct - uniform_pct,
            }
        )

    if validate_weights and validated_windows > 0:
        logging.info(
            f"✓ Validated weight sums for {validated_windows} windows (all sum to 1.0)"
        )

    return pd.DataFrame(results).set_index("window")


def backtest_dynamic_dca(
    dataframe: pd.DataFrame,
    strategy_function,
    features_df: pd.DataFrame | None = None,
    *,
    strategy_label: str = "strategy",
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, float]:
    """Run rolling-window SPD backtest and log aggregated performance metrics.

    Uses precomputed features for efficient computation.

    Args:
        dataframe: DataFrame containing price data with 'PriceUSD_coinmetrics' column
        strategy_function: Function that takes features DataFrame and returns weights
        features_df: Optional precomputed features. If None, computes them internally.
        strategy_label: Label for logging (default: "strategy")
        start_date: Optional start date (default: BACKTEST_START)
        end_date: Optional end date (default: BACKTEST_END)

    Returns:
        Tuple of (SPD table DataFrame, exponential-decay average percentile)
    """
    spd_table = compute_cycle_spd(
        dataframe,
        strategy_function,
        features_df=features_df,
        start_date=start_date,
        end_date=end_date,
    )
    dynamic_spd = spd_table["dynamic_sats_per_dollar"]
    dynamic_pct = spd_table["dynamic_percentile"]

    # Exponential decay weighting (recent windows weighted more)
    N = len(dynamic_spd)
    exp_weights = 0.9 ** np.arange(N - 1, -1, -1)
    exp_weights /= exp_weights.sum()
    exp_avg_pct = (dynamic_pct.values * exp_weights).sum()

    logging.info(f"Aggregated Metrics for {strategy_label}:")
    logging.info(
        f"  SPD: min={dynamic_spd.min():.2f}, max={dynamic_spd.max():.2f}, "
        f"mean={dynamic_spd.mean():.2f}, median={dynamic_spd.median():.2f}"
    )
    logging.info(
        f"  Percentile: min={dynamic_pct.min():.2f}%, max={dynamic_pct.max():.2f}%, "
        f"mean={dynamic_pct.mean():.2f}%, median={dynamic_pct.median():.2f}%"
    )
    logging.info(f"  Exp-decay avg SPD percentile: {exp_avg_pct:.2f}%")

    return spd_table, exp_avg_pct


def check_strategy_submission_ready(dataframe: pd.DataFrame, strategy_function) -> None:
    """Validate strategy: no future data, valid weights, ≥50% win rate vs uniform DCA."""
    print("Validating strategy submission readiness...")
    passed = True

    # Forward-leakage test
    backtest_end = BACKTEST_END or dataframe.index.max().strftime("%Y-%m-%d")
    backtest_df = dataframe.loc[BACKTEST_START:backtest_end]
    full_weights = strategy_function(dataframe).reindex(backtest_df.index).fillna(0.0)

    for probe in backtest_df.index[:: max(len(backtest_df) // 50, 1)]:
        masked = dataframe.copy()
        masked.loc[masked.index > probe, :] = np.nan
        masked_wt = strategy_function(masked).reindex(full_weights.index).fillna(0.0)

        if not np.isclose(
            masked_wt.loc[probe], full_weights.loc[probe], rtol=1e-9, atol=1e-12
        ):
            print(
                f"[{probe.date()}] ❌ Forward-leakage detected (Δ={abs(masked_wt.loc[probe] - full_weights.loc[probe]):.2e})"
            )
            passed = False
            break

    # Weight validation per rolling window (using 1-year windows)
    window_offset = WINDOW_OFFSET
    backtest_end = BACKTEST_END or dataframe.index.max().strftime("%Y-%m-%d")
    for start in pd.date_range(
        pd.to_datetime(BACKTEST_START),
        pd.to_datetime(backtest_end) - window_offset,
        freq=PURCHASE_FREQ_TO_OFFSET[PURCHASE_FREQ],
    ):
        end = start + window_offset
        label = _make_window_label(start, end)
        w_slice = strategy_function(dataframe.loc[start:end])

        if (w_slice < 0).any():
            print(f"[{label}] ❌ Negative weights detected.")
            passed = False

        total = w_slice.sum()
        if not np.isclose(total, 1.0, rtol=1e-5, atol=1e-8):
            print(
                f"[{label}] ❌ Sum-to-1 check failed: {total:.4f} (weights must sum to 1.0)"
            )
            passed = False

    # Performance vs uniform DCA
    spd_table = compute_cycle_spd(dataframe, strategy_function)
    underperf = spd_table[
        spd_table["dynamic_percentile"] < spd_table["uniform_percentile"]
    ]

    if not underperf.empty:
        print("\n⚠️ Windows where strategy underperformed Uniform DCA:")
        display(
            underperf[["dynamic_percentile", "uniform_percentile"]].assign(
                Delta=lambda x: x["dynamic_percentile"] - x["uniform_percentile"]
            )
        )

    win_rate = 1 - len(underperf) / len(spd_table)
    print(
        f"\nSummary: {len(underperf)}/{len(spd_table)} underperformed ({100 * win_rate:.2f}% win rate)"
    )

    if win_rate >= 0.5:
        print(
            "✅ Strategy meets performance requirement (≥ 50% win rate vs. uniform DCA)."
        )
    else:
        print(
            "❌ Strategy failed performance requirement (< 50% win rate vs. uniform DCA)."
        )
        passed = False

    print()
    print(
        "✅ Strategy is ready for submission."
        if passed
        else "⚠️ Please address the above issues before submitting."
    )
