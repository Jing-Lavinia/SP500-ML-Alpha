"""
features.py
-----------
Factor library: classic price/volume factors plus HLOC-specific signals.

Factor construction follows a standard pipeline:
  1. Raw signal computation (e.g. rolling std, ratio)
  2. Cross-sectional winsorisation and z-scoring via _cs_clean()
  3. Optional EWMA smoothing to reduce daily noise

HLOC factors exploit the Open, High, and Low prices that pure close-price
models ignore -- overnight gaps, intraday range, and close location within
the day's range all carry incremental information.

Dynamic ICIR synthesis (synthesize_dynamic) blends factor scores using
information-ratio weights computed over three rolling windows, with an
IC sign-consistency guard to suppress factors whose direction has flipped.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .utils import logger

def _cs_clean(raw: pd.DataFrame, ewm_span: int = 1) -> pd.DataFrame:
    """Cross-sectional winsorise (5xMAD), z-score, and optionally EWMA-smooth."""
    median = raw.median(axis=1)
    mad    = raw.sub(median, axis=0).abs().median(axis=1)
    winsor = raw.clip(lower=median - 5 * mad, upper=median + 5 * mad, axis=0)
    mu  = winsor.mean(axis=1)
    std = winsor.std(axis=1).replace(0, np.nan)
    zsc = winsor.sub(mu, axis=0).div(std, axis=0)
    if ewm_span > 1: return zsc.ewm(span=ewm_span, adjust=False).mean()
    return zsc

# --- Classic Price/Vol Factors ---
def calc_momentum(prices, window=252, skip=21, span=5):
    raw = prices.shift(skip).pct_change(window - skip, fill_method=None)
    return _cs_clean(raw, ewm_span=span)

def calc_mean_revert(prices, window=5):
    return _cs_clean(-prices.pct_change(window, fill_method=None), ewm_span=1)

def calc_low_vol(prices, window=63):
    rets = prices.pct_change(fill_method=None)
    return _cs_clean(-rets.rolling(window, min_periods=window//2).std(), ewm_span=5)

def calc_amihud_illiq(prices, volumes, window=21):
    dollar_vol = prices * volumes
    rets = prices.pct_change(fill_method=None).abs()
    ratio = rets / dollar_vol.replace(0, np.nan)
    return _cs_clean(-np.log1p(ratio.rolling(window, min_periods=window//2).mean()), ewm_span=3)

def calc_turnover_rate(volumes, window=20):
    raw = -np.log1p(volumes).rolling(window, min_periods=window//2).mean()
    return _cs_clean(raw, ewm_span=3)

# ==============================================================================
# HLOC Factors -- exploit Open, High, Low in addition to Close
# ==============================================================================

def calc_overnight_gap_reversal(open_df, close_df):
    """Overnight gap Reversal: Fades extreme overnight news reactions."""
    logger.info("  Factor: OvernightGapRev")
    # (Open today / Close yesterday) - 1
    gap = (open_df / close_df.shift(1)) - 1
    return _cs_clean(-gap, ewm_span=1)

def calc_intraday_momentum(open_df, close_df):
    """Intraday Momentum: Captures true institutional buying during market hours."""
    logger.info("  Factor: IntradayMom")
    # (Close today / Open today) - 1
    intraday = (close_df / open_df.replace(0, np.nan)) - 1
    return _cs_clean(intraday, ewm_span=3)

def calc_parkinson_vol(high_df, low_df, window=21):
    """Parkinson Volatility: Much more accurate risk estimator using H/L range."""
    logger.info(f"  Factor: ParkinsonVol_{window}")
    # sqrt( 1 / (4*ln(2)) * ln(High/Low)^2 )
    hl_ratio = np.log(high_df / low_df.replace(0, np.nan)) ** 2
    park_vol = np.sqrt((1.0 / (4.0 * np.log(2.0))) * hl_ratio.rolling(window, min_periods=window//2).mean())
    return _cs_clean(-park_vol, ewm_span=5)

def calc_close_location_value(high_df, low_df, close_df, window=21):
    """CLV (Stochastic): High CLV means closing near the high (buying pressure)."""
    logger.info(f"  Factor: CLV_{window}")
    # ((C - L) - (H - C)) / (H - L)
    num = (close_df - low_df) - (high_df - close_df)
    den = high_df - low_df
    clv = num / den.replace(0, np.nan)
    return _cs_clean(clv.rolling(window, min_periods=window//2).mean(), ewm_span=3)

# ---------------------------------------------------------------------------

def build_factor_pool(open_df, high_df, low_df, close_df, vol_df):
    logger.info("Building HLOC-enhanced factor pool...")
    pool = {}

    # Momentum & Trend
    pool["Momentum_252"]    = calc_momentum(close_df, window=252, skip=21, span=5)
    pool["Momentum_63"]     = calc_momentum(close_df, window=63, skip=5, span=3)
    pool["MeanRevert_5"]    = calc_mean_revert(close_df, window=5)

    # HLOC Specific
    pool["OvernightGapRev"] = calc_overnight_gap_reversal(open_df, close_df)
    pool["IntradayMom"]     = calc_intraday_momentum(open_df, close_df)
    pool["ParkinsonVol_21"] = calc_parkinson_vol(high_df, low_df, window=21)
    pool["ParkinsonVol_63"] = calc_parkinson_vol(high_df, low_df, window=63)
    pool["CLV_21"]          = calc_close_location_value(high_df, low_df, close_df, window=21)

    # Standard Volatility & Liquidity
    pool["LowVol_63"]       = calc_low_vol(close_df, window=63)
    pool["AmihudIlliq"]     = calc_amihud_illiq(close_df, vol_df, window=21)
    pool["TurnoverRate"]    = calc_turnover_rate(vol_df, window=20)

    logger.info(f"Factor pool built: {len(pool)} HLOC-powered factors")
    return pool

def _compute_icir_single_window(ic_s, lookback, min_periods, smooth_span):
    """Compute rolling ICIR for a single factor IC series and apply EWMA smoothing."""
    rolling_mean = ic_s.rolling(lookback, min_periods=min_periods).mean()
    rolling_std  = ic_s.rolling(lookback, min_periods=min_periods).std().replace(0, np.nan)
    icir_raw = rolling_mean / rolling_std
    return icir_raw.ewm(span=smooth_span, adjust=False).mean()


def _compute_monthly_ic_sign(ic_daily_series):
    """Aggregate daily IC to monthly mean -- used for sign-consistency protection."""
    return ic_daily_series.resample("ME").mean()


def synthesize_dynamic(
    factor_pool, prices,
    forward_days=None,
    lookback_days=None,
    icir_smooth_span=None,
    min_ic_periods=None,
    max_icir_clip=None,
    rebal_freq=5,
):
    """
    Dynamic ICIR composite synthesis engine.

    Combines all factors in `factor_pool` into a single composite score by
    weighting each factor proportionally to its |ICIR|.  Two guard mechanisms
    prevent overfitting:
      1. Multi-window ICIR blend (short/med/long) smooths out noisy weight flips.
      2. IC sign-consistency protection zeroes the weight of any factor whose
         monthly IC has been negative for N consecutive months.

    The daily IC series is shifted forward by `forward_days` before weight
    computation so that rebalancing weights only use fully-realised IC observations
    (no look-ahead).
    """
    from . import config as _cfg

    forward_days    = forward_days    or _cfg.ICIR_FORWARD_DAYS
    lookback_days   = lookback_days   or _cfg.ICIR_LOOKBACK_DAYS
    icir_smooth_span= icir_smooth_span or _cfg.ICIR_SMOOTH_SPAN
    min_ic_periods  = min_ic_periods  or _cfg.ICIR_MIN_PERIODS
    max_icir_clip   = max_icir_clip   or _cfg.ICIR_MAX_CLIP

    lookback_short  = _cfg.ICIR_LOOKBACK_SHORT
    lookback_med    = _cfg.ICIR_LOOKBACK_MED
    w_s, w_m, w_l   = _cfg.ICIR_WINDOW_WEIGHTS

    sign_window   = _cfg.IC_SIGN_PROTECTION_WINDOW
    sign_factors  = set(_cfg.IC_SIGN_PROTECTION_FACTORS)

    logger.info("=== Dynamic ICIR synthesis engine (multi-window + sign protection) ===")
    fwd_ret = prices.pct_change(forward_days, fill_method=None).shift(-forward_days)
    idx, cols = prices.index, prices.columns

    all_icir      = {}
    all_ic_daily  = {}
    all_monthly_ic= {}

    for name, factor_df in factor_pool.items():
        aligned = factor_df.reindex(index=idx, columns=cols).shift(1)
        daily_ic = []
        for i in range(len(aligned)):
            f_row = aligned.iloc[i].dropna()
            r_row = fwd_ret.iloc[i].dropna()
            common = f_row.index.intersection(r_row.index)
            daily_ic.append(
                spearmanr(f_row[common].values, r_row[common].values)[0]
                if len(common) >= 20 else np.nan
            )

    ic_s = pd.Series(daily_ic, index=aligned.index)

    # Shift IC forward by forward_days so weights at time T only see IC realised before T
    ic_s_safe = ic_s.shift(forward_days)

    all_ic_daily[name] = ic_s_safe

    # Blend three ICIR windows: short (reactive) + medium + long (stable)
    icir_short = _compute_icir_single_window(ic_s_safe, lookback_short, min_ic_periods // 2, icir_smooth_span)
    icir_med = _compute_icir_single_window(ic_s_safe, lookback_med, min_ic_periods, icir_smooth_span)
    icir_long = _compute_icir_single_window(ic_s_safe, lookback_days, min_ic_periods, icir_smooth_span)

    blended = w_s * icir_short.fillna(0) + w_m * icir_med.fillna(0) + w_l * icir_long.fillna(0)
    all_icir[name] = blended

    if name in sign_factors:
        all_monthly_ic[name] = _compute_monthly_ic_sign(ic_s_safe)

    icir_history = pd.DataFrame(all_icir)
    rebal_dates  = set(idx[::rebal_freq])
    composite    = pd.DataFrame(0.0, index=idx, columns=cols)
    prev_weights = {}

    for date in idx:
        if date in rebal_dates or not prev_weights:
            icir_row = icir_history.loc[date].clip(-max_icir_clip, max_icir_clip).dropna()

            # IC sign-consistency guard: zero out factors with N consecutive negative monthly ICs
            for fname in sign_factors:
                if fname not in icir_row.index:
                    continue
                monthly_ic = all_monthly_ic.get(fname)
                if monthly_ic is None:
                    continue

                # Only use months strictly before current month to avoid look-ahead
                current_month_start = date.replace(day=1)
                past_months = monthly_ic[monthly_ic.index < current_month_start].tail(sign_window)

                if len(past_months) >= sign_window and (past_months < 0).all():
                    logger.debug(f"  [sign-protect] {fname} paused at {date.date()} (IC negative {sign_window} consecutive months)")
                    icir_row[fname] = 0.0

            # Normalise ICIR weights to sum to 1 in absolute value
            total_abs = icir_row.abs().sum()
            if total_abs < 1e-8:
                prev_weights = {n: 1.0 / len(factor_pool) for n in factor_pool}
            else:
                prev_weights = {n: 0.0 for n in factor_pool}
                prev_weights.update((icir_row / total_abs).to_dict())

        daily = pd.Series(0.0, index=cols)
        for name, factor_df in factor_pool.items():
            w = prev_weights.get(name, 0.0)
            if abs(w) >= 1e-8:
                daily += factor_df.reindex(index=idx, columns=cols).loc[date].fillna(0) * w
        composite.loc[date] = daily

    return _cs_clean(composite, ewm_span=1), icir_history
