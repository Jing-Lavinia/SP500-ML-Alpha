"""
simulator.py
------------
Vectorised daily-frequency backtest engine.

Strategy: Variable Beta long/short equity (130/30 style) with:
  - Sector-neutral stock selection (top/bottom decile per sector)
  - Inverse-volatility position sizing within each leg
  - VIX-triggered deleveraging: ramps down gross exposure as VIX rises above threshold
  - Macro-regime overlay: reduces long exposure during bear-trend + rate-shock environments
  - Bull/bear regime switch based on price vs 120-day SMA

Portfolio mechanics:
  - Weights are smoothed over the holding period (rolling mean) to reduce rebalancing churn
  - A 0.5% deviation band suppresses tiny trades (turnover bands)
  - Per-leg position cap: +/-10% of NAV per stock

Metrics computed: CAGR, Sharpe, Sortino, Calmar, Omega, VaR/CVaR, drawdown statistics,
win rate, turnover, and exposure breakdown.
"""

import numpy as np
import pandas as pd
from .utils import logger
from . import config
import alphalens



def custom_forward_returns(prices: pd.DataFrame,
                           periods: list[int]) -> pd.DataFrame:
    """
    Manually compute forward returns for each period.
    Bypasses Alphalens' frequency-inference bug on intraday data.
    """
    returns_list = []
    for p in periods:
        ret = prices.pct_change(p, fill_method=None).shift(-p)
        stacked = ret.stack()
        stacked.name = f"{p}D"
        returns_list.append(stacked)
    fwd = pd.concat(returns_list, axis=1)
    fwd.index.names = ["date", "asset"]
    return fwd


def prepare_alphalens_data(prices: pd.DataFrame,
                           factor: pd.DataFrame,
                           sector_map: dict) -> pd.DataFrame:
    """
    Format data for Alphalens compatibility.

    Parameters
    ----------
    prices     : daily close price DataFrame
    factor     : composite factor score DataFrame
    sector_map : {ticker: sector_name}

    Returns
    -------
    factor_data : Alphalens-compatible clean factor DataFrame
    """
    logger.info("Preparing Alphalens data...")

    factor_stacked = factor.stack()
    factor_stacked.index.names = ["date", "asset"]

    tickers  = factor_stacked.index.get_level_values("asset")
    sectors  = tickers.map(sector_map).fillna("Unknown")
    sec_codes, sec_names = pd.factorize(pd.Series(sectors, index=factor_stacked.index))
    sector_series = pd.Series(sec_codes, index=factor_stacked.index)
    sector_labels = dict(enumerate(sec_names))

    fwd_returns = custom_forward_returns(prices, config.FORWARD_PERIODS)

    factor_data = alphalens.utils.get_clean_factor(
        factor          = factor_stacked,
        forward_returns = fwd_returns,
        groupby         = sector_series,
        groupby_labels  = sector_labels,
        quantiles       = 5,
        max_loss        = 0.35,
    )

    logger.info(f"Alphalens data ready: {len(factor_data):,} rows")
    return factor_data



def _build_dynamic_hedged_weights(
    factor_scores: pd.DataFrame,
    prices_df: pd.DataFrame,
    vix_series: pd.Series | None,
    sector_map: dict,
    macro_features: pd.DataFrame | None = None,
    quantile: float = 0.10,
) -> pd.DataFrame:
    """
    Build daily target weight matrix for a sector-neutral long/short portfolio.

    Selection: top/bottom `quantile` fraction within each sector.
    Sizing: inverse-volatility weighting within each leg.
    Leverage: dynamically scaled by VIX level, bull/bear regime, and macro winter flag.
    """
    sector_series = pd.Series(sector_map)
    ranks = factor_scores.copy()

    # Compute within-sector percentile ranks for each date
    for date in ranks.index:
        row = factor_scores.loc[date].dropna()
        if row.empty: continue
        sec = sector_series.reindex(row.index).fillna("Unknown")
        ranks.loc[date, row.index] = row.groupby(sec).rank(pct=True)

    long_sel = ranks >= (1.0 - quantile)
    short_sel = ranks <= quantile

    is_prob = (factor_scores.max().max() <= 1.0) and (factor_scores.min().min() >= 0.0)

    if is_prob:
        long_sig = factor_scores
        short_sig = 1.0 - factor_scores
    else:
        long_sig = 1.0
        short_sig = 1.0

    clean_prices = prices_df.replace(0.0, np.nan)
    daily_rets = clean_prices.pct_change(fill_method=None)
    daily_rets = daily_rets.clip(lower=-0.50, upper=0.50).fillna(0)
    rolling_vol = daily_rets.rolling(21, min_periods=10).std().shift(1)

    inv_vol = 1.0 / rolling_vol.replace(0, np.nan)

    # Inverse-vol weighted, normalised within each leg to sum to 1
    long_w = long_sel.astype(float) * long_sig * inv_vol
    long_w = long_w.div(long_w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    short_w = short_sel.astype(float) * short_sig * inv_vol
    short_w = short_w.div(short_w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    # Bull/bear regime: price above/below 120-day SMA
    mkt_idx = prices_df.mean(axis=1)
    sma_120 = mkt_idx.rolling(120).mean()
    is_bull_regime = (mkt_idx > sma_120).shift(1).fillna(True)

    # Macro winter: bear trend AND a recent large rate shock co-occurring
    is_macro_winter = pd.Series(False, index=prices_df.index)
    if macro_features is not None and "rate_shock_5d" in macro_features.columns:
        sma_200 = mkt_idx.rolling(200).mean()
        is_bear_trend = (mkt_idx < sma_200).shift(1).fillna(False)

        rate_shock = macro_features["rate_shock_5d"].reindex(prices_df.index).ffill()
        recent_rate_shock = (rate_shock >= 0.35).rolling(window=42, min_periods=1).max().shift(1).fillna(0).astype(bool)

        is_macro_winter = is_bear_trend & recent_rate_shock

    mkt_rets = daily_rets.mean(axis=1)
    mkt_vol_ann = mkt_rets.rolling(21).std() * np.sqrt(252)

    if vix_series is not None:
        aligned_vix = vix_series.reindex(factor_scores.index).ffill()
    else:
        aligned_vix = mkt_vol_ann * 100

    vix_t_minus_1 = aligned_vix.shift(1).fillna(20.0)

    # Deleverage ramp: linearly reduce exposure as VIX exceeds threshold
    vix_trigger = config.VIX_DELEVERAGE_LEVEL
    panic_ratio = ((vix_t_minus_1 - vix_trigger) / 10.0).clip(lower=0.0, upper=1.0)
    short_panic_ratio = ((vix_t_minus_1 - vix_trigger) / 8.0).clip(lower=0.0, upper=1.0)

    target_volatility = 0.15
    current_mkt_vol = mkt_vol_ann.shift(1).fillna(0.15)
    vol_scalar = (target_volatility / current_mkt_vol.replace(0, np.nan)).clip(lower=0.5, upper=1.2)

    base_long_tgt  = np.where(is_bull_regime, 1.30, 0.80)
    base_short_tgt = np.where(is_bull_regime, 0.30, 0.80)

    leverage_mult       = 1.0 - (panic_ratio * 0.4)
    short_leverage_mult = 1.0 - (short_panic_ratio * 0.6)

    final_long_tgt  = pd.Series(base_long_tgt  * leverage_mult,       index=prices_df.index) * vol_scalar
    final_short_tgt = pd.Series(base_short_tgt * short_leverage_mult, index=prices_df.index) * vol_scalar

    final_long_tgt[is_macro_winter] = 0.50
    final_short_tgt[is_macro_winter] = 0.50

    final_w = long_w.multiply(final_long_tgt, axis=0) - short_w.multiply(final_short_tgt, axis=0)

    # Hard cap: no single position exceeds +/-10% of NAV
    final_w[final_w > 0] = final_w[final_w > 0].clip(upper=0.10)
    final_w[final_w < 0] = final_w[final_w < 0].clip(lower=-0.10)

    return final_w


def run_realistic_backtest(
    prices_df: pd.DataFrame,
    factor_df: pd.DataFrame,
    macro_features: pd.DataFrame | None = None,
    holding_period: int | None = None,
    base_transaction_cost: float | None = None,
    stop_loss_pct: float | None = None,
    vix_series: pd.Series | None = None,
    sector_map: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:

    holding_period        = holding_period or config.HOLDING_PERIOD
    base_transaction_cost = base_transaction_cost or config.TRANSACTION_COST
    sector_map            = sector_map or config.SECTOR_MAP

    logger.info("=== Backtest engine: Variable Beta (130/30) with Vol-Targeting & Regime Switching ===")
    logger.info(f"  Holding period: {holding_period}d | Cost: {base_transaction_cost*1e4:.1f}bps")

    logger.info("Constructing Dynamic Target Weights...")
    raw_weights = _build_dynamic_hedged_weights(
        factor_df, prices_df, vix_series, sector_map=sector_map, macro_features=macro_features, quantile=0.10
    )

    logger.info(f"Applying {holding_period}-day rolling smooth rebalancing and TURNOVER BANDS...")
    # Rolling mean smooths target weights over the holding period -> fewer large rebalances
    target_weights = raw_weights.rolling(window=holding_period, min_periods=1).mean()

    final_weights = target_weights.copy()
    current_w = pd.Series(0.0, index=prices_df.columns)

    # Turnover band: only trade if deviation from current weight exceeds 0.5%
    for i in range(len(final_weights)):
        target_w = target_weights.iloc[i].fillna(0)
        deviation = (target_w - current_w).abs()
        trade_mask = deviation > 0.005
        current_w[trade_mask] = target_w[trade_mask]
        final_weights.iloc[i] = current_w

    avg_gross = final_weights.abs().sum(axis=1).mean()
    avg_long  = final_weights[final_weights > 0].sum(axis=1).mean()
    avg_short = final_weights[final_weights < 0].abs().sum(axis=1).mean()
    logger.info(f"  Leverage -- Gross: {avg_gross:.3f}x | Long: {avg_long:.3f}x | Short: {avg_short:.3f}x")

    logger.info("Computing returns and transaction costs...")
    daily_rets       = prices_df.pct_change(fill_method=None)
    current_bar_rets = daily_rets.fillna(0)

    gross_returns    = (final_weights.fillna(0) * current_bar_rets).sum(axis=1)
    turnover_daily   = final_weights.diff().abs().sum(axis=1) / 2.0
    cost_series      = turnover_daily * base_transaction_cost
    net_returns      = gross_returns - cost_series

    cum_returns  = (1 + net_returns).cumprod()
    rolling_peak = cum_returns.cummax()
    drawdowns    = (cum_returns - rolling_peak) / rolling_peak

    n_bars = len(cum_returns)
    bpy    = config.BARS_PER_YEAR

    cagr = (cum_returns.iloc[-1] ** (bpy / n_bars) - 1) if cum_returns.iloc[-1] > 0 else net_returns.mean() * bpy

    ann_vol_ret  = net_returns.std() * np.sqrt(bpy)
    sharpe       = cagr / ann_vol_ret if ann_vol_ret > 0 else 0.0
    max_dd       = drawdowns.min()

    neg_rets     = net_returns[net_returns < 0]
    downside_vol = neg_rets.std() * np.sqrt(bpy) if len(neg_rets) > 0 else 1e-6
    sortino      = cagr / downside_vol if downside_vol > 0 else 0.0

    non_zero = net_returns[net_returns != 0]
    win_rate = (non_zero > 0).sum() / len(non_zero) if len(non_zero) > 0 else 0.0

    avg_turnover = turnover_daily.mean()

    cur_len, dd_lens, dd_depths = 0, [], []
    in_dd = False
    peak_eq = 1.0
    for eq_val, dd_val in zip(cum_returns, drawdowns):
        if dd_val < 0:
            if not in_dd:
                in_dd = True
                peak_eq = eq_val / (1 + dd_val)
            cur_len += 1
            dd_depths.append(dd_val)
        else:
            if in_dd:
                dd_lens.append(cur_len)
            cur_len = 0
            in_dd = False
    if in_dd and cur_len > 0:
        dd_lens.append(cur_len)

    calmar      = cagr / abs(max_dd) if max_dd < 0 else 0.0
    total_ret   = cum_returns.iloc[-1] - 1

    pos_rets = net_returns[net_returns > 0]
    neg_rets_abs = net_returns[net_returns < 0].abs()
    profit_factor = (pos_rets.sum() / neg_rets_abs.sum()
                     if len(neg_rets_abs) > 0 and neg_rets_abs.sum() > 0 else np.nan)

    threshold = 0.0
    gains  = (net_returns - threshold).clip(lower=0).sum()
    losses = (threshold - net_returns).clip(lower=0).sum()
    omega  = gains / losses if losses > 0 else np.nan

    sorted_r = net_returns.sort_values()
    var_95   = float(sorted_r.quantile(0.05))
    cvar_95  = float(sorted_r[sorted_r <= var_95].mean()) if (sorted_r <= var_95).any() else var_95

    tail_ratio = (abs(float(sorted_r.quantile(0.95))) / abs(float(sorted_r.quantile(0.05)))
                  if sorted_r.quantile(0.05) != 0 else np.nan)

    from scipy.stats import skew, kurtosis
    ret_skew = float(skew(net_returns.dropna()))
    ret_kurt = float(kurtosis(net_returns.dropna()))

    monthly_ret = net_returns.resample("ME").apply(lambda r: (1 + r).prod() - 1)

    avg_dd       = float(np.mean(dd_depths)) if dd_depths else 0.0
    avg_dd_dur   = float(np.mean(dd_lens)) if dd_lens else 0.0
    max_dd_dur   = max(dd_lens) if dd_lens else 0

    metrics_dict = {
        "Total Return":           f"{total_ret:.4%}",
        "Geometric CAGR":         f"{cagr:.4%}",
        "Annualised Volatility":  f"{ann_vol_ret:.4%}",
        "Sharpe Ratio":           f"{sharpe:.4f}",
        "Sortino Ratio":          f"{sortino:.4f}",
        "Calmar Ratio":           f"{calmar:.4f}",
        "Omega Ratio":            f"{omega:.4f}" if not np.isnan(omega) else "--",
        "Tail Ratio":             f"{tail_ratio:.4f}" if not np.isnan(tail_ratio) else "--",
        "Profit Factor":          f"{profit_factor:.4f}" if not np.isnan(profit_factor) else "--",
        "Max Drawdown":           f"{max_dd:.4%}",
        "Avg Drawdown":           f"{avg_dd:.4%}",
        "Max DD Duration (days)": f"{max_dd_dur}",
        "Avg DD Duration (days)": f"{avg_dd_dur:.1f}",
        "VaR 95% (Daily)":        f"{var_95:.4%}",
        "CVaR 95% (Daily)":       f"{cvar_95:.4%}",
        "Skewness":               f"{ret_skew:.4f}",
        "Excess Kurtosis":        f"{ret_kurt:.4f}",
        "Win Rate (Daily)":       f"{win_rate:.4%}",
        "Best Day":               f"{net_returns.max():.4%}",
        "Worst Day":              f"{net_returns.min():.4%}",
        "Best Month":             f"{monthly_ret.max():.4%}",
        "Worst Month":            f"{monthly_ret.min():.4%}",
        "Avg Daily Turnover":     f"{avg_turnover:.4%}",
        "Annual Turnover":        f"{avg_turnover * bpy:.2f}x",
        "Avg Gross Exposure":     f"{avg_gross:.3f}x",
        "Avg Long Exposure":      f"{avg_long:.3f}x",
        "Avg Short Exposure":     f"{avg_short:.3f}x",
    }

    logger.info("=" * 52)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("=" * 52)
    for k, v in metrics_dict.items():
        logger.info(f"  {k:<28} {v}")
    logger.info("=" * 52)

    results_df = pd.DataFrame({
        "Gross Return": gross_returns, "Cost": cost_series, "Net Return": net_returns,
        "Equity Curve": cum_returns, "Drawdown": drawdowns, "Turnover": turnover_daily,
    })

    return results_df, final_weights, metrics_dict
