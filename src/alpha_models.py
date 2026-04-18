"""
alpha_models.py
---------------
Walk-forward ML factor scoring engine.
Integrates HLOCV factors with alternative macro and sentiment data.

Key components:
  - build_feature_matrix: constructs the stockxdate panel with all features
    and a binary top-decile forward-return label for classification.
  - run_ml_scoring: walk-forward cross-validation using LightGBM + Ridge
    ensemble, producing out-of-sample probability scores for every date.
  - compute_factor_ic: monthly Spearman IC between each factor and the target.
  - _wf_splits: generates (train_start, train_end, test_start, test_end)
    tuples with an embargo gap to prevent leakage.
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from .utils import logger
from . import config


def build_feature_matrix(factor_pool, prices, volumes, forward_days=10):
    save_dir = config.REPORTS_DIR / "feature_store"
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Building ML feature matrix (Top 10% Targeting, {forward_days}d Horizon)...")
    idx, cols = prices.index, prices.columns

    # Binary label: 1 if a stock ranks in the top 10% of forward returns that day
    target_shift = -(forward_days - 1) if forward_days > 1 else 0
    fwd_ret_raw  = prices.pct_change(forward_days, fill_method=None).shift(target_shift)

    ranks = fwd_ret_raw.rank(axis=1, pct=True)
    target_binary = (ranks >= 0.90).astype(float)
    target_binary = target_binary.where(fwd_ret_raw.notna(), np.nan)

    target_binary.index.name, target_binary.columns.name = "date", "ticker"

    frames = []
    for name, df in factor_pool.items():
        s = df.reindex(index=idx, columns=cols).stack(dropna=False)
        s.name = name
        frames.append(s)

    panel = pd.concat(frames, axis=1)
    panel.index.names = ["date", "ticker"]

    # Market-level features -- same for all stocks on a given day
    mkt_idx = prices.mean(axis=1)
    panel['mkt_ret_5d'] = mkt_idx.pct_change(5).shift(1).reindex(panel.index.get_level_values('date')).values
    panel['mkt_ret_21d'] = mkt_idx.pct_change(21).shift(1).reindex(panel.index.get_level_values('date')).values
    panel['mkt_vol_21d'] = mkt_idx.pct_change().rolling(21).std().shift(1).reindex(panel.index.get_level_values('date')).values

    macro_file = config.DATA_DIR / "macro_alt_features.csv"
    if macro_file.exists():
        macro_df = pd.read_csv(macro_file, index_col=0, parse_dates=True)
        for col in macro_df.columns:
            panel[col] = macro_df[col].reindex(panel.index.get_level_values('date')).values

    news_file = config.DATA_DIR / "alpaca_news_sentiment.csv"
    if news_file.exists():
        news_df = pd.read_csv(news_file, index_col=["date", "ticker"], parse_dates=["date"])
        panel = panel.join(news_df, how='left')
        panel["news_sentiment_score"] = panel.get("news_sentiment_score", pd.Series(0, index=panel.index)).fillna(0)

    target_stacked = target_binary.stack(dropna=False)
    target_stacked.name = f"target_{forward_days}d"
    panel = panel.join(target_stacked)

    # Encode sector as integer for tree-based models
    tickers = panel.index.get_level_values("ticker")
    sector_raw = tickers.map(config.SECTOR_MAP).fillna("Unknown")
    sector_codes, _ = pd.factorize(sector_raw, sort=True)
    panel["sector_code"] = sector_codes

    # Interaction terms
    if "MomAccel" in panel.columns and "SkewFactor" in panel.columns:
        panel["ix_MomAccel_Skew"]    = panel["MomAccel"] * panel["SkewFactor"]
    if "Momentum_252" in panel.columns and "LowVol_63" in panel.columns:
        panel["ix_Mom252_LowVol"]    = panel["Momentum_252"] * panel["LowVol_63"]

    feature_cols = _get_feature_cols(panel, factor_pool)
    panel.dropna(subset=feature_cols, how="all", inplace=True)
    panel.to_parquet(save_dir / f"feature_matrix_fwd{forward_days}d.parquet")
    return panel

def _wf_splits(dates, train_days, retrain_freq, embargo_days):
    """
    Generate walk-forward (train, test) split indices with an embargo gap.
    The embargo ensures no overlap between training labels and test features,
    preventing forward-return leakage.
    """
    n = len(dates)
    i = train_days
    while i < n:
        train_start, train_end = dates[i - train_days], dates[i - 1]
        test_start_i = min(i + embargo_days, n - 1)
        test_end_i   = min(i + embargo_days + retrain_freq - 1, n - 1)
        yield train_start, train_end, dates[test_start_i], dates[test_end_i]
        i += retrain_freq


def _build_lgbm():
    import lightgbm as lgb
    return lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        scale_pos_weight=9.0,   # compensate for 9:1 class imbalance (top 10% label)
        n_estimators=300,
        learning_rate=0.02,
        num_leaves=31,
        max_depth=5,
        min_child_samples=30,
        feature_fraction=0.5,
        bagging_fraction=0.8,
        bagging_freq=3,
        lambda_l1=0.1,
        lambda_l2=0.5,
        verbose=-1,
        n_jobs=-1
    )

def _build_ridge():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    return Pipeline([
        ("scaler", StandardScaler()),
        ("logit",  LogisticRegression(class_weight={1: 9.0, 0: 1.0}, C=0.1, max_iter=1000, solver="liblinear")),
    ])


def run_ml_scoring(panel, factor_pool, prices, forward_days=10):
    target_col = f"target_{forward_days}d"
    feature_cols = _get_feature_cols(panel, factor_pool)

    all_dates = panel.index.get_level_values("date").unique().sort_values()
    tickers = prices.columns.tolist()
    train_days = int(config.ML_TRAIN_YEARS * config.BARS_PER_YEAR)

    logger.info("=== Walk-forward ML scoring engine (PROBABILITY RANKING) ===")
    score_wide = pd.DataFrame(np.nan, index=prices.index, columns=tickers)
    diag_records = []

    splits = list(_wf_splits(all_dates, train_days, config.ML_RETRAIN_FREQ, config.ML_EMBARGO_DAYS))

    if not splits: return score_wide.fillna(0), pd.DataFrame()

    for fold_idx, (t_start, t_end, test_start, test_end) in enumerate(splits, 1):
        train_mask = (panel.index.get_level_values("date") >= t_start) & (panel.index.get_level_values("date") <= t_end)
        train_df = panel.loc[train_mask, feature_cols + [target_col]].dropna()
        if len(train_df) < config.ML_MIN_TRAIN_ROWS: continue

        X_train, y_train = train_df[feature_cols].values, train_df[target_col].values
        test_mask = (panel.index.get_level_values("date") >= test_start) & (
                    panel.index.get_level_values("date") <= test_end)
        test_df = panel.loc[test_mask, feature_cols + [target_col]].dropna()
        if len(test_df) == 0: continue
        X_test, y_test = test_df[feature_cols].values, test_df[target_col].values

        lgbm_model, logit_model = _build_lgbm(), _build_ridge()
        lgbm_model.fit(X_train, y_train)
        logit_model.fit(X_train, y_train)

        pred_lgbm = lgbm_model.predict_proba(X_test)[:, 1]
        pred_logit = logit_model.predict_proba(X_test)[:, 1]

        # Ensemble: LightGBM captures non-linear patterns; Ridge adds regularised linearity
        pred_ens = 0.7 * pred_lgbm + 0.3 * pred_logit

        oos_ic_lgbm = oos_ic_ridge = oos_ic_ens = np.nan
        if len(test_df) > 20:
            oos_ic_lgbm = spearmanr(pred_lgbm, y_test)[0]
            oos_ic_ridge = spearmanr(pred_logit, y_test)[0]
            oos_ic_ens = spearmanr(pred_ens, y_test)[0]

        logger.info(
            f"  Fold {fold_idx:02d} | test: {test_start.date()} -> {test_end.date()} | IC(Prob): {oos_ic_ens:+.3f}")

        pred_series = pd.Series(pred_ens, index=test_df.index)
        for (date, ticker), score in pred_series.items():
            if date in score_wide.index and ticker in score_wide.columns:
                score_wide.loc[date, ticker] = score

        fi = dict(zip(feature_cols, lgbm_model.feature_importances_))

        diag_records.append({
            "fold": fold_idx, "train_end": t_end, "test_start": test_start,
            "test_end": test_end, "train_rows": len(train_df),
            "ic_lgbm": oos_ic_lgbm,
            "ic_ridge": oos_ic_ridge,
            "ic_ensemble": oos_ic_ens,
            **{f"fi_{k}": v for k, v in fi.items()}
        })

    diagnostics = pd.DataFrame(diag_records)
    if len(diagnostics) > 0:
        logger.info(f"Walk-forward complete | Mean OOS IC (ensemble): {diagnostics['ic_ensemble'].mean():+.4f}")

    # Fill gaps with the most recent score (staleness capped at one retrain period) and smooth
    score_clean = score_wide.ffill(limit=config.ML_RETRAIN_FREQ).ewm(span=5, adjust=False).mean()

    diag_path = config.REPORTS_DIR / "data_exports" / "ml_diagnostics.csv"
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(diag_path, index=False)

    return score_clean, diagnostics


def compute_factor_ic(panel, factor_cols, target_col):
    ic_rows = []
    dates   = panel.index.get_level_values("date").unique().sort_values()
    for period_end in pd.date_range(dates[0], dates[-1], freq="ME"):
        period_start = period_end - pd.DateOffset(months=1)
        mask = ((panel.index.get_level_values("date") >= period_start) & (panel.index.get_level_values("date") <= period_end))
        sub = panel.loc[mask].dropna(subset=factor_cols + [target_col])
        if len(sub) < 20: continue
        row = {"date": period_end}
        for fc in factor_cols:
            row[fc] = spearmanr(sub[fc].values, sub[target_col].values)[0]
        ic_rows.append(row)
    return pd.DataFrame(ic_rows).set_index("date")

def _get_feature_cols(panel: pd.DataFrame, factor_pool: dict[str, pd.DataFrame]) -> list[str]:
    base_cols = list(factor_pool.keys())
    alt_cols = [c for c in panel.columns if c in ['mkt_ret_5d', 'mkt_ret_21d', 'mkt_vol_21d',
                                                  'rate_shock_5d', 'credit_spread_trend',
                                                  'tech_relative_strength_21d', 'tech_volume_shock',
                                                  'news_sentiment_score']]
    ix_cols = [c for c in panel.columns if c.startswith("ix_")]
    return base_cols + alt_cols + ix_cols
