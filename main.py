"""
main.py
-------
Quantitative Factor Pipeline -- S&P 500 Daily
Architecture 3.0: Alt Data + Signal Weighting + Macro Hedging
"""
import argparse
import pandas as pd
from src.utils import logger
from src import config
from src.data_loader import (
    fetch_universe,
    fetch_vix,
    fetch_benchmark,
    fetch_macro_and_flows,
    fetch_alpaca_news_sentiment
)
from src import features, simulator, visualization, alpha_models

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--no-alphalens", action="store_true")
    p.add_argument("--force-refresh", action="store_true")
    p.add_argument("--linear-only", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()

    TARGET_HORIZON = 10

    logger.info("=" * 64)
    logger.info("  Quantitative Factor Pipeline -- S&P 500 Daily  ")
    logger.info(f"  Signal Weighting + Macro Hedge + {TARGET_HORIZON}d Horizon")
    logger.info("=" * 64)

    logger.info("STAGE 1: Loading core market data...")
    open_full, high_full, low_full, prices_full, volumes_full = fetch_universe(force_refresh=args.force_refresh)
    vix_full  = fetch_vix(force_refresh=args.force_refresh)
    spy_full  = fetch_benchmark(force_refresh=args.force_refresh)

    logger.info("STAGE 1B: Loading alternative data (Macro & Sentiment)...")
    fetch_macro_and_flows(force_refresh=args.force_refresh)
    fetch_alpaca_news_sentiment(config.SP500_TICKERS, config.DATA_START_DATE, config.END_DATE)

    logger.info("STAGE 2: Computing HLOC-enhanced factor pool...")
    factor_pool = features.build_factor_pool(open_full, high_full, low_full, prices_full, volumes_full)

    logger.info("STAGE 3: Building ML feature matrix...")
    panel = alpha_models.build_feature_matrix(factor_pool, prices_full, volumes_full, forward_days=TARGET_HORIZON)
    factor_ic_df = alpha_models.compute_factor_ic(
        panel,
        alpha_models._get_feature_cols(panel, factor_pool),
        f"target_{TARGET_HORIZON}d"
    )

    logger.info("STAGE 4: Dynamic ICIR composite synthesis...")
    composite_full, icir_history = features.synthesize_dynamic(factor_pool, prices_full)

    logger.info(f"Slicing to backtest window: {config.BACKTEST_START_DATE} -> {config.END_DATE}")
    prices = prices_full.loc[config.BACKTEST_START_DATE:config.END_DATE]
    vix_data = vix_full.loc[config.BACKTEST_START_DATE:config.END_DATE]
    spy_data = spy_full.loc[config.BACKTEST_START_DATE:config.END_DATE]
    linear_f = composite_full.loc[config.BACKTEST_START_DATE:config.END_DATE]

    macro_df = pd.read_csv(config.DATA_DIR / "macro_alt_features.csv", index_col=0, parse_dates=True)

    if not args.no_alphalens:
        logger.info("STAGE 5: Alphalens factor evaluation...")
        try:
            visualization.generate_alphalens_reports(
                simulator.prepare_alphalens_data(prices, linear_f, config.SECTOR_MAP))
        except Exception as exc:
            logger.warning(f"Alphalens failed: {exc}")

    logger.info("STAGE 6: Linear ICIR backtest...")
    res_linear, wts_linear, met_linear = simulator.run_realistic_backtest(
        prices, linear_f, macro_features=macro_df, holding_period=TARGET_HORIZON, vix_series=vix_data
    )

    res_ml, met_ml = None, None
    if not args.linear_only:
        logger.info("STAGE 7: Walk-forward ML scoring (LightGBM + Ridge)...")
        ml_scores, diagnostics = alpha_models.run_ml_scoring(panel, factor_pool, prices_full, forward_days=TARGET_HORIZON)
        ml_scores_bt = ml_scores.loc[config.BACKTEST_START_DATE:config.END_DATE]

        coverage = ml_scores_bt.notna().any(axis=1).mean()
        logger.info(f"ML score coverage in backtest window: {coverage:.1%}")

        if coverage > 0.3:
            logger.info("STAGE 8: ML backtest...")
            res_ml, wts_ml, met_ml = simulator.run_realistic_backtest(
                prices, ml_scores_bt, macro_features=macro_df, holding_period=TARGET_HORIZON, vix_series=vix_data
            )

    logger.info("STAGE 9: Generating visualisation reports...")
    visualization.render_performance_dashboard(res_linear, met_linear, spy_data, factor_ic_df, title_suffix=" -- Linear")
    visualization.render_factor_ic_summary(factor_ic_df)
    visualization.render_icir_weight_history(icir_history.loc[config.BACKTEST_START_DATE:config.END_DATE])
    visualization.render_monthly_returns_heatmap(res_linear, title_suffix=" -- Linear")
    visualization.render_cost_turnover_analysis(res_linear, title_suffix=" -- Linear")
    visualization.render_drawdown_analysis(res_linear, title_suffix=" -- Linear")
    visualization.render_regime_conditional_returns(
        res_linear, benchmark_series=spy_data, vix_series=vix_data, title_suffix=" -- Linear"
    )
    visualization.render_long_short_attribution(
        res_linear, wts_linear, prices, title_suffix=" -- Linear"
    )

    if not args.linear_only and 'diagnostics' in locals() and not diagnostics.empty:
        visualization.render_ml_diagnostics(diagnostics)
        if res_ml is not None:
            visualization.render_performance_dashboard(res_ml, met_ml, spy_data, factor_ic_df, title_suffix=" -- ML Ensemble")
            visualization.render_strategy_comparison(res_linear, res_ml, met_linear, met_ml, spy_data)
            visualization.render_monthly_returns_heatmap(res_ml, title_suffix=" -- ML Ensemble")
            visualization.render_cost_turnover_analysis(res_ml, title_suffix=" -- ML Ensemble")
            visualization.render_drawdown_analysis(res_ml, title_suffix=" -- ML Ensemble")
            visualization.render_regime_conditional_returns(
                res_ml, benchmark_series=spy_data, vix_series=vix_data, title_suffix=" -- ML Ensemble"
            )
            visualization.render_long_short_attribution(
                res_ml, wts_ml, prices, title_suffix=" -- ML Ensemble"
            )

    logger.info("STAGE 10: Exporting data artifacts...")
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 64)
    logger.info(f"  Pipeline complete -> {config.REPORTS_DIR}")
    logger.info("=" * 64)

if __name__ == "__main__":
    main()