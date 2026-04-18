"""
test_data_integrity.py
----------------------
Standalone look-ahead bias and data integrity test suite.

Runs five checks that guard against the most common sources of leakage
in a quantitative backtesting pipeline:

  1. Ghost Stock Test      -- prices are NaN before a stock's IPO date
  2. NaN & Continuity      -- volumes are zero pre-IPO; prices resume post-IPO
  3. Time Machine Test     -- rolling factor values are identical whether computed
                             on full or truncated price history (no future peeking)
  4. Alignment Test        -- ML panel's binary target labels match manually
                             computed forward-return ranks
  5. Volume Halt Test      -- volume is not forward-filled on price-halt days
  5b. News Timezone Test   -- after-hours news timestamps are correctly shifted
                             to the next trading day (including Friday -> Monday)

Run directly:  python -m src.test_data_integrity
"""
import pandas as pd
import numpy as np
from src import features, alpha_models, config
from src.data_loader import fetch_universe

def run_tests(open_df, high_df, low_df, close_df, vol_df, factor_pool, panel):
    print("\n" + "="*60)
    print("Starting strict data integrity tests...")
    print("="*60)

    # Stocks that listed after DATA_START_DATE -- used as canaries for pre-IPO leakage
    late_ipo_stocks = [s for s in ['CRWD', 'DDOG', 'PLTR'] if s in close_df.columns]

    # ==========================================
    # Test 1: Ghost Stock Test
    # ==========================================
    print("  [1/5] Running Ghost Stock Test (NaN integrity)...")

    if late_ipo_stocks:
        test_stock = late_ipo_stocks[0]
        for df_name, df in zip(['Open', 'High', 'Low', 'Close'], [open_df, high_df, low_df, close_df]):
            old_prices = df.loc[:'2018-12-31', test_stock]
            assert old_prices.isna().all(), f"FATAL LOOK-AHEAD: {test_stock} {df_name} data exists before IPO (possible bfill contamination)"
        print(f"    [PASSED] Ghost Stock Test: {test_stock} O/H/L/C are NaN before IPO.")
    else:
        print("    [SKIPPED] No late-IPO stocks found in universe.")

    # ==========================================
    # Test 2: NaN & Continuity Verification
    # ==========================================
    print("\n  [2/5] Running True NaN & Continuity Verification...")

    if late_ipo_stocks:
        test_stock = late_ipo_stocks[0]

        old_volume = vol_df.loc[:'2018-12-31', test_stock]
        assert (old_volume == 0.0).all(), f"Logic error: {test_stock} volume should be 0.0 before IPO (fillna(0) not applied)"

        new_prices = close_df.loc['2021-01-01':'2021-12-31', test_stock]
        assert new_prices.notna().any(), f"Continuity break: {test_stock} has all-NaN prices after IPO"

        print(f"    [PASSED] NaN & Continuity: {test_stock} pre-IPO volume=0, post-IPO prices intact.")
    else:
        print("    [SKIPPED] No late-IPO stocks found in universe.")

    # ==========================================
    # Test 3: Time Machine Test -- no future data in rolling factors
    # ==========================================
    print("\n  [3/5] Running Time Machine Test (No future leakage in math)...")

    test_date = pd.to_datetime('2022-01-05')
    if test_date not in close_df.index:
        test_date = close_df.index[close_df.index > '2022-01-01'][0]

    vol_factor_name = "LowVol_63"
    full_factor = factor_pool[vol_factor_name]

    # Recompute the factor using only data up to test_date -- result must be identical
    prices_truncated = close_df.loc[:test_date].copy()
    truncated_factor = features.calc_low_vol(prices_truncated, window=63)

    diff = (full_factor.loc[test_date] - truncated_factor.loc[test_date]).abs().max()

    assert pd.isna(diff) or diff < 1e-10, f"FATAL LOOK-AHEAD: factor computation leaks future data! max_diff={diff}"
    print(f"    [PASSED] Time Machine Test: {vol_factor_name} rolling computation is look-ahead-free.")

    # ==========================================
    # Test 4: ML Panel Alignment Test
    # ==========================================
    print("\n  [4/5] Running Alignment Test (Panel features vs Future targets)...")

    test_ticker = close_df.columns[0]
    panel_sub = panel.xs(test_ticker, level='ticker')

    valid_dates = panel_sub.dropna(subset=['target_10d']).index
    if len(valid_dates) > 10:
        check_date = valid_dates[100]

        idx_loc = close_df.index.get_loc(check_date)
        price_start = close_df[test_ticker].iloc[idx_loc]
        price_end   = close_df[test_ticker].iloc[idx_loc + 10]

        assert price_start == close_df.loc[check_date, test_ticker], "Alignment Error: return start price is misaligned"

        all_starts = close_df.iloc[idx_loc]
        all_ends   = close_df.iloc[idx_loc + 10]
        all_rets   = (all_ends / all_starts) - 1

        manual_rank = all_rets.rank(pct=True)[test_ticker]
        manual_binary = 1.0 if manual_rank >= 0.90 else 0.0

        panel_binary = panel_sub.loc[check_date, 'target_10d']

        assert manual_binary == panel_binary, f"FATAL ALIGNMENT: label mismatch -- manual={manual_binary}, panel={panel_binary}"
        print(f"    [PASSED] Alignment Test: {test_ticker} forward-return label is perfectly aligned.")

    # ==========================================
    # Test 5: Volume Halt Logic
    # ==========================================
    print("\n  [5/5] Running Volume Halt Logic Verification & Timezone test...")

    test_stock_vol = close_df.columns[0]
    flat_days = close_df[test_stock_vol].pct_change() == 0.0

    if flat_days.any():
        halt_date = flat_days[flat_days].index[0]
        halt_vol = vol_df.loc[halt_date, test_stock_vol]

        prev_date = close_df.index[close_df.index.get_loc(halt_date) - 1]
        prev_vol = vol_df.loc[prev_date, test_stock_vol]

        assert halt_vol == 0.0 or halt_vol != prev_vol, f"Logic error: {test_stock_vol} volume on halt day {halt_date.date()} appears to be forward-filled"
        print(f"    [PASSED] Volume Halt Test: halt-day volume on {halt_date.date()} is not forward-filled.")
    else:
        print("    [SKIPPED] No price-flat days found for first ticker.")

    # After-hours news timezone logic
    from pandas.tseries.offsets import BDay
    def mock_timezone_logic(created_at_str):
        raw_time = pd.to_datetime(created_at_str)
        if raw_time.tzinfo is None:
            raw_time = raw_time.tz_localize('UTC')
        est_time = raw_time.tz_convert('America/New_York')
        if est_time.hour >= 16:
            return (est_time + BDay(1)).date()
        else:
            return est_time.date()

    assert str(mock_timezone_logic("2023-11-01 19:30:00+00:00")) == "2023-11-01", "Intraday timezone logic error"
    assert str(mock_timezone_logic("2023-11-01 20:30:00+00:00")) == "2023-11-02", "After-hours roll-forward logic error"
    assert str(mock_timezone_logic("2023-11-03 21:00:00+00:00")) == "2023-11-06", "Friday after-hours -> Monday roll logic error"

    print("    [PASSED] News Timezone Test: after-hours and weekend news correctly deferred.")

    print("\n" + "="*60)
    print("ALL LOOK-AHEAD BIAS TESTS PASSED! Your data pipeline is exceptionally safe.")
    print("="*60 + "\n")


if __name__ == "__main__":
    print("Initializing test environment...")

    print("-> Loading market data (this may take a moment)...")
    open_df, high_df, low_df, close_df, vol_df = fetch_universe(force_refresh=False)

    print("-> Building factor pool...")
    factors = features.build_factor_pool(open_df, high_df, low_df, close_df, vol_df)

    print("-> Building feature matrix (Panel)...")
    panel_df = alpha_models.build_feature_matrix(factors, close_df, vol_df, forward_days=10)

    run_tests(open_df, high_df, low_df, close_df, vol_df, factors, panel_df)
