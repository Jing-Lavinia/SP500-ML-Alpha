"""
visualization.py
----------------
Publication-quality report generation.

Charts produced
---------------
  performance_dashboard.png       4-panel: equity curve / rolling Sharpe /
                                          drawdown / factor IC heatmap
  factor_ic_summary.png           Mean IC and ICIR bar charts per factor
  icir_weight_history.png         Dynamic ICIR weight time-series + stacked area
  ml_diagnostics.png              Walk-forward OOS IC per fold + feature importance
  monthly_returns_heatmap.png     Calendar heatmap of monthly net returns + annual bar chart
  cost_turnover_analysis.png      Gross/net return decomposition + turnover & cost time-series
  drawdown_analysis.png           Top-5 drawdown periods table + underwater equity curve overlay
  regime_conditional_returns.png  Return / Sharpe / hit-rate split by VIX regime & market trend
  long_short_attribution.png      Long-leg vs short-leg daily P&L attribution + rolling contribution
  (Alphalens plots)               Quantile returns / IC timeseries / turnover
"""

import io
from pathlib import Path
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import alphalens as al

from . import config
from .utils import logger

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------

_C = {
    "strategy":  "#C0392B",
    "benchmark": "#95A5A6",
    "positive":  "#27AE60",
    "negative":  "#E74C3C",
    "neutral":   "#2980B9",
    "drawdown":  "#E59866",
    "lgbm":      "#8E44AD",
    "ridge":     "#2980B9",
    "ensemble":  "#C0392B",
}

_GRID = dict(alpha=0.25, linestyle="--", linewidth=0.7, color="#999999")


def _style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.spines.top":  False,
        "axes.spines.right":False,
        "axes.grid":        True,
        "grid.alpha":       0.25,
        "grid.linestyle":   "--",
        "font.family":      "DejaVu Sans",
        "font.size":        10,
    })


def _pct(x, _):  return f"{x:.1%}"
def _num(x, _):  return f"{x:.1f}"
def _date_fmt(ax): ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))


# ---------------------------------------------------------------------------
# Alphalens tearsheet
# ---------------------------------------------------------------------------

def generate_alphalens_reports(factor_data: pd.DataFrame):
    """Render all Alphalens component plots and save to reports/plots/."""
    _style()
    plots_dir  = config.REPORTS_DIR / "plots"
    tables_dir = config.REPORTS_DIR / "tables"
    for sub in ["01_returns", "02_ic", "03_turnover"]:
        (plots_dir / sub).mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    mean_ret_q, _ = al.performance.mean_return_by_quantile(
        factor_data, by_date=False, by_group=True)
    mean_ret_q.to_csv(tables_dir / "01_returns_mean_by_quantile.csv")
    mean_ret_daily, _ = al.performance.mean_return_by_quantile(
        factor_data, by_date=True, by_group=False)

    al.plotting.plot_quantile_returns_bar(mean_ret_q, by_group=True)
    plt.gcf().savefig(plots_dir / "01_returns" / "01_quantile_bar.png",
                      bbox_inches="tight", dpi=150)
    plt.close("all")

    for p in config.FORWARD_PERIODS:
        al.plotting.plot_cumulative_returns_by_quantile(mean_ret_daily, period=f"{p}D")
        plt.gcf().savefig(plots_dir / "01_returns" / f"02_cumulative_{p}D.png",
                          bbox_inches="tight", dpi=150)
        plt.close("all")

    ic = al.performance.factor_information_coefficient(factor_data)
    ic.to_csv(tables_dir / "02_ic_timeseries.csv")
    for name, fn in [("ic_ts",   al.plotting.plot_ic_ts),
                     ("ic_hist", al.plotting.plot_ic_hist),
                     ("ic_qq",   al.plotting.plot_ic_qq)]:
        fn(ic)
        plt.gcf().savefig(plots_dir / "02_ic" / f"{name}.png",
                          bbox_inches="tight", dpi=150)
        plt.close("all")

    fra = al.performance.factor_rank_autocorrelation(factor_data)
    fra.to_csv(tables_dir / "03_turnover_fra.csv")
    al.plotting.plot_factor_rank_auto_correlation(fra)
    plt.gcf().savefig(plots_dir / "03_turnover" / "factor_rank_autocorrelation.png",
                      bbox_inches="tight", dpi=150)
    plt.close("all")

    buf = io.StringIO()
    orig = plt.show;  plt.show = lambda: None
    try:
        with redirect_stdout(buf):
            al.tears.create_returns_tear_sheet(factor_data, by_group=True)
            print("\n" + "=" * 60 + "\n")
            al.tears.create_information_tear_sheet(factor_data, by_group=True)
    finally:
        plt.show = orig;  plt.close("all")

    with open(config.REPORTS_DIR / "alphalens_summary.txt", "w") as f:
        f.write(buf.getvalue())
    logger.info(f"Alphalens reports → {plots_dir}")


# ---------------------------------------------------------------------------
# Main performance dashboard  (4 panels)
# ---------------------------------------------------------------------------

def render_performance_dashboard(
    results: pd.DataFrame,
    metrics: dict,
    benchmark_series: pd.Series | None = None,
    factor_ic: pd.DataFrame | None = None,
    output_dir: Path | None = None,
    title_suffix: str = "",
):
    """
    4-panel performance dashboard:
      [0] Equity curve vs SPY benchmark
      [1] Rolling 252-day Sharpe
      [2] Drawdown waterfall with annotation
      [3] Factor monthly IC heatmap
    """
    _style()
    output_dir = output_dir or config.REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    has_ic   = factor_ic is not None and not factor_ic.empty
    n_panels = 4 if has_ic else 3

    fig = plt.figure(figsize=(16, 4.2 * n_panels), facecolor="white")
    gs  = gridspec.GridSpec(n_panels, 1, hspace=0.48, figure=fig)

    eq   = results["Equity Curve"]
    nret = results["Net Return"]

    # --- Panel 0: Equity curve ---
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(eq.index, eq.values, label=f"Strategy{title_suffix}",
             color=_C["strategy"], linewidth=2.0, zorder=3)

    if benchmark_series is not None:
        bm  = benchmark_series.reindex(eq.index).ffill()
        bmc = (1 + bm.pct_change().fillna(0)).cumprod()
        ax0.plot(bmc.index, bmc.values, label="SPY (S&P 500)",
                 color=_C["benchmark"], linestyle="--", linewidth=1.4, alpha=0.9)

    ax0.set_title(f"Cumulative Returns{title_suffix}", fontsize=13, fontweight="bold", pad=10)
    ax0.set_ylabel("Cumulative Return")
    ax0.yaxis.set_major_formatter(FuncFormatter(_pct))
    ax0.legend(loc="upper left", fontsize=9, framealpha=0.7)
    _date_fmt(ax0)

    ann = (f"CAGR: {metrics.get('Geometric CAGR','—')}   "
           f"Sharpe: {metrics.get('Sharpe Ratio','—')}   "
           f"Sortino: {metrics.get('Sortino Ratio','—')}   "
           f"Max DD: {metrics.get('Max Drawdown','—')}")
    ax0.text(0.01, 0.04, ann, transform=ax0.transAxes,
             fontsize=8.5, color="#444", style="italic")

    # --- Panel 1: Rolling Sharpe ---
    ax1 = fig.add_subplot(gs[1])
    rw  = min(252, len(nret) // 3)
    rm  = nret.rolling(rw, min_periods=rw // 2).mean()
    rs  = nret.rolling(rw, min_periods=rw // 2).std()
    rsh = (rm / rs.replace(0, np.nan)) * np.sqrt(config.BARS_PER_YEAR)

    ax1.plot(rsh.index, rsh.values, color=_C["neutral"], linewidth=1.5,
             label=f"{rw}-day rolling Sharpe")
    ax1.axhline(0,   color="#888", linewidth=0.8, linestyle="--")
    ax1.axhline(1.0, color=_C["positive"], linewidth=0.7, linestyle=":", alpha=0.8,
                label="Sharpe = 1")
    ax1.axhline(2.0, color=_C["positive"], linewidth=0.7, linestyle=":", alpha=0.5,
                label="Sharpe = 2")
    ax1.fill_between(rsh.index, 0, rsh, where=(rsh >= 0),
                     alpha=0.12, color=_C["positive"])
    ax1.fill_between(rsh.index, 0, rsh, where=(rsh < 0),
                     alpha=0.12, color=_C["negative"])
    ax1.set_title(f"Rolling {rw}-Day Sharpe Ratio", fontsize=12, fontweight="bold", pad=8)
    ax1.set_ylabel("Sharpe Ratio")
    ax1.legend(fontsize=8.5, framealpha=0.7)
    _date_fmt(ax1)

    # --- Panel 2: Drawdown ---
    ax2 = fig.add_subplot(gs[2])
    dd  = results["Drawdown"]
    ax2.fill_between(dd.index, dd.values, 0,
                     color=_C["drawdown"], alpha=0.75, label="Drawdown")
    ax2.plot(dd.index, dd.values, color=_C["drawdown"], linewidth=0.9)
    ax2.set_title("Portfolio Drawdown", fontsize=12, fontweight="bold", pad=8)
    ax2.set_ylabel("Drawdown")
    ax2.yaxis.set_major_formatter(FuncFormatter(_pct))
    ax2.set_ylim(top=0.005)
    _date_fmt(ax2)

    idx_min = dd.idxmin()
    ax2.annotate(f"Max DD: {dd.min():.1%}",
                 xy=(idx_min, dd.min()), xytext=(10, 15),
                 textcoords="offset points", fontsize=8.5,
                 color=_C["negative"],
                 arrowprops=dict(arrowstyle="->", color=_C["negative"], lw=0.9))

    # --- Panel 3: Factor IC heatmap ---
    if has_ic:
        ax3 = fig.add_subplot(gs[3])
        ic_t = factor_ic.T
        ic_c = ic_t.clip(-0.05, 0.05)
        im   = ax3.imshow(ic_c.values, aspect="auto", cmap="RdYlGn",
                          vmin=-0.05, vmax=0.05, interpolation="nearest")
        ax3.set_yticks(range(len(ic_t.index)))
        ax3.set_yticklabels(ic_t.index, fontsize=7)
        step = max(1, len(ic_t.columns) // 12)
        ax3.set_xticks(range(0, len(ic_t.columns), step))
        ax3.set_xticklabels([str(d)[:7] for d in ic_t.columns[::step]],
                            rotation=45, ha="right", fontsize=7)
        ax3.set_title("Factor Monthly IC Heatmap", fontsize=12, fontweight="bold", pad=8)
        plt.colorbar(im, ax=ax3, label="Rank IC", fraction=0.015, pad=0.01)

    out = output_dir / "performance_dashboard.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    logger.info(f"Performance dashboard → {out}")


# ---------------------------------------------------------------------------
# Factor IC summary bar charts
# ---------------------------------------------------------------------------

def render_factor_ic_summary(
    factor_ic: pd.DataFrame,
    output_dir: Path | None = None,
):
    """Mean IC and ICIR horizontal bar charts, sorted by magnitude."""
    _style()
    output_dir = output_dir or config.REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    mean_ic = factor_ic.mean().sort_values()
    icir    = (mean_ic / factor_ic.std().replace(0, np.nan)).reindex(mean_ic.index)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor="white")
    fig.suptitle("Factor Information Coefficient Analysis", fontsize=13,
                 fontweight="bold", y=1.01)

    for ax, data, xlabel, title in [
        (axes[0], mean_ic, "Mean Rank IC",  "Mean Monthly IC by Factor"),
        (axes[1], icir,    "ICIR",          "IC Information Ratio (ICIR)"),
    ]:
        colors = [_C["positive"] if v > 0 else _C["negative"] for v in data.values]
        bars   = ax.barh(data.index, data.values, color=colors,
                         edgecolor="white", linewidth=0.5, height=0.7)
        ax.axvline(0, color="#333", linewidth=0.9)

        # Value labels on bars
        for bar, val in zip(bars, data.values):
            if not np.isnan(val):
                ha  = "left" if val >= 0 else "right"
                off = 0.0003 if val >= 0 else -0.0003
                ax.text(val + off, bar.get_y() + bar.get_height() / 2,
                        f"{val:+.3f}", va="center", ha=ha, fontsize=7, color="#333")

        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_xlabel(xlabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", **_GRID)

    fig.tight_layout()
    out = output_dir / "factor_ic_summary.png"
    fig.savefig(out, bbox_inches="tight", dpi=160)
    plt.close(fig)
    logger.info(f"Factor IC summary → {out}")


# ---------------------------------------------------------------------------
# Dynamic ICIR weight history
# ---------------------------------------------------------------------------

def render_icir_weight_history(
    icir_history: pd.DataFrame,
    output_dir: Path | None = None,
    top_n: int = 10,
):
    """
    Two-panel chart:
      [top]    Rolling ICIR time-series for top-N factors
      [bottom] Normalised ICIR weight allocation stacked area
    """
    _style()
    output_dir = output_dir or config.REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    top_factors = icir_history.abs().mean().nlargest(top_n).index.tolist()
    plot_data   = icir_history[top_factors].dropna(how="all")
    norm        = icir_history.abs().sum(axis=1).replace(0, np.nan)
    norm_w      = icir_history[top_factors].div(norm, axis=0).fillna(0)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), facecolor="white")
    fig.suptitle("Dynamic ICIR Factor Weight History", fontsize=14,
                 fontweight="bold", y=1.01)
    cmap = plt.cm.get_cmap("tab10", top_n)

    # Top panel — ICIR line series
    ax0 = axes[0]
    for i, col in enumerate(top_factors):
        ax0.plot(plot_data.index, plot_data[col].values,
                 label=col, color=cmap(i), linewidth=1.3, alpha=0.85)
    ax0.axhline(0, color="#555", linewidth=1.0, linestyle="--")
    ax0.fill_between(plot_data.index, plot_data.max(axis=1).clip(lower=0), 0,
                     alpha=0.04, color=_C["positive"])
    ax0.fill_between(plot_data.index, plot_data.min(axis=1).clip(upper=0), 0,
                     alpha=0.04, color=_C["negative"])
    ax0.set_title(f"Rolling ICIR per Factor — Top {top_n}", fontsize=11,
                  fontweight="bold")
    ax0.set_ylabel("ICIR  (weight direction × magnitude)")
    ax0.legend(fontsize=7.5, ncol=2, framealpha=0.6, loc="upper right")
    _date_fmt(ax0)

    # Bottom panel — stacked area
    ax1 = axes[1]
    pos_w = norm_w.clip(lower=0)
    neg_w = norm_w.clip(upper=0)
    ax1.stackplot(norm_w.index,
                  [pos_w[c].values for c in top_factors],
                  labels=top_factors,
                  colors=[cmap(i) for i in range(top_n)],
                  alpha=0.78)
    ax1.stackplot(norm_w.index,
                  [neg_w[c].values for c in top_factors],
                  colors=[cmap(i) for i in range(top_n)],
                  alpha=0.78)
    ax1.axhline(0, color="#333", linewidth=1.0)
    ax1.set_title("Normalised ICIR Weight Allocation (Stacked)", fontsize=11,
                  fontweight="bold")
    ax1.set_ylabel("Normalised weight")
    ax1.legend(fontsize=7.5, ncol=2, framealpha=0.6, loc="upper right")
    _date_fmt(ax1)

    fig.tight_layout()
    out = output_dir / "icir_weight_history.png"
    fig.savefig(out, bbox_inches="tight", dpi=180)
    plt.close(fig)
    logger.info(f"ICIR weight history → {out}")


# ---------------------------------------------------------------------------
# ML diagnostics chart
# ---------------------------------------------------------------------------

def render_ml_diagnostics(
    diagnostics: pd.DataFrame,
    output_dir: Path | None = None,
):
    """
    Three-panel ML walk-forward report:
      [0] OOS IC per fold (LGBM vs Ridge vs Ensemble)
      [1] Cumulative mean IC over folds
      [2] Top-15 feature importance (LightGBM, last fold)
    """
    if diagnostics is None or diagnostics.empty:
        logger.warning("ML diagnostics empty — skipping chart.")
        return

    _style()
    output_dir = output_dir or config.REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 13), facecolor="white")
    fig.suptitle("ML Walk-Forward Diagnostics", fontsize=14,
                 fontweight="bold", y=1.01)
    gs  = gridspec.GridSpec(3, 1, hspace=0.50, figure=fig)

    folds = diagnostics["fold"].values

    # Panel 0 — OOS IC per fold
    ax0 = fig.add_subplot(gs[0])
    w   = 0.25
    ax0.bar(folds - w, diagnostics["ic_lgbm"],     width=w, label="LightGBM",
            color=_C["lgbm"],  alpha=0.85)
    ax0.bar(folds,     diagnostics["ic_ridge"],    width=w, label="Ridge",
            color=_C["ridge"], alpha=0.85)
    ax0.bar(folds + w, diagnostics["ic_ensemble"], width=w, label="Ensemble (0.6/0.4)",
            color=_C["ensemble"], alpha=0.85)
    ax0.axhline(0, color="#555", linewidth=0.8)
    ax0.axhline(0.05,  color=_C["positive"], linewidth=0.7,
                linestyle=":", alpha=0.8, label="IC = 0.05")
    ax0.set_title("Out-of-Sample Rank IC per Fold", fontsize=11, fontweight="bold")
    ax0.set_xlabel("Fold")
    ax0.set_ylabel("Spearman Rank IC")
    ax0.legend(fontsize=8.5, framealpha=0.7)
    ax0.grid(axis="y", **_GRID)

    # Panel 1 — Cumulative mean IC
    ax1 = fig.add_subplot(gs[1])
    for col, label, color in [
        ("ic_lgbm",     "LightGBM",       _C["lgbm"]),
        ("ic_ridge",    "Ridge",          _C["ridge"]),
        ("ic_ensemble", "Ensemble",       _C["ensemble"]),
    ]:
        cumm = diagnostics[col].expanding().mean()
        ax1.plot(folds, cumm.values, label=label, color=color, linewidth=1.8)

    ax1.axhline(0, color="#888", linewidth=0.8, linestyle="--")
    ax1.set_title("Cumulative Mean OOS IC (Expanding)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Fold")
    ax1.set_ylabel("Cumulative Mean IC")
    ax1.legend(fontsize=8.5, framealpha=0.7)
    ax1.grid(axis="y", **_GRID)

    # Panel 2 — Feature importance (last fold, LightGBM)
    ax2 = fig.add_subplot(gs[2])
    fi_cols = [c for c in diagnostics.columns if c.startswith("fi_")]
    if fi_cols:
        last_fi = diagnostics[fi_cols].iloc[-1].sort_values(ascending=False).head(15)
        labels  = [c.replace("fi_", "") for c in last_fi.index]
        colors  = [_C["lgbm"] if not c.startswith("mkt_") else _C["neutral"]
                   for c in last_fi.index]
        ax2.barh(labels[::-1], last_fi.values[::-1],
                 color=colors[::-1], edgecolor="white", height=0.7)
        ax2.set_title("Top-15 Feature Importance — LightGBM (Last Fold)",
                      fontsize=11, fontweight="bold")
        ax2.set_xlabel("Feature Importance (split gain)")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.grid(axis="x", **_GRID)

    out = output_dir / "ml_diagnostics.png"
    fig.savefig(out, bbox_inches="tight", dpi=180)
    plt.close(fig)
    logger.info(f"ML diagnostics → {out}")


# ---------------------------------------------------------------------------
# Side-by-side comparison: linear ICIR vs ML
# ---------------------------------------------------------------------------

def _compute_benchmark_metrics(
    benchmark_series: pd.Series,
    ref_index: pd.Index,
) -> dict:
    """
    Compute the full performance metric set for the SPY benchmark using the
    same formulas applied in realistic_backtest, aligned to ref_index.
    Execution metrics (turnover, long/short exposure) are set to "—".
    """
    bm  = benchmark_series.reindex(ref_index).ffill()
    ret = bm.pct_change(fill_method=None).fillna(0)

    eq           = (1 + ret).cumprod()
    rolling_peak = eq.cummax()
    dd           = (eq - rolling_peak) / rolling_peak

    n_bars  = len(ret)
    bpy     = config.BARS_PER_YEAR
    cagr    = eq.iloc[-1] ** (bpy / n_bars) - 1
    total_r = eq.iloc[-1] - 1
    ann_vol = ret.std() * np.sqrt(bpy)
    sharpe  = cagr / ann_vol if ann_vol > 0 else 0.0
    max_dd  = dd.min()
    calmar  = cagr / abs(max_dd) if max_dd < 0 else 0.0

    neg     = ret[ret < 0]
    d_vol   = neg.std() * np.sqrt(bpy) if len(neg) > 0 else 1e-6
    sortino = cagr / d_vol if d_vol > 0 else 0.0

    pos_r   = ret[ret > 0]
    neg_abs = ret[ret < 0].abs()
    pf      = pos_r.sum() / neg_abs.sum() if neg_abs.sum() > 0 else np.nan

    gains  = ret.clip(lower=0).sum()
    losses = (-ret).clip(lower=0).sum()
    omega  = gains / losses if losses > 0 else np.nan

    sorted_r  = ret.sort_values()
    var_95    = float(sorted_r.quantile(0.05))
    cvar_95   = float(sorted_r[sorted_r <= var_95].mean()) if (sorted_r <= var_95).any() else var_95
    tail_r    = (abs(float(sorted_r.quantile(0.95))) / abs(float(sorted_r.quantile(0.05)))
                 if sorted_r.quantile(0.05) != 0 else np.nan)

    from scipy.stats import skew as _skew, kurtosis as _kurt
    ret_skew = float(_skew(ret.dropna()))
    ret_kurt = float(_kurt(ret.dropna()))

    non_zero  = ret[ret != 0]
    win_rate  = (non_zero > 0).sum() / len(non_zero) if len(non_zero) > 0 else 0.0
    monthly_r = ret.resample("ME").apply(lambda r: (1 + r).prod() - 1)

    cur_len, dd_lens, dd_depths = 0, [], []
    in_dd = False
    for dd_val in dd:
        if dd_val < 0:
            if not in_dd:
                in_dd = True
            cur_len += 1
            dd_depths.append(dd_val)
        else:
            if in_dd:
                dd_lens.append(cur_len)
            cur_len = 0
            in_dd = False
    if in_dd and cur_len > 0:
        dd_lens.append(cur_len)

    avg_dd     = float(np.mean(dd_depths)) if dd_depths else 0.0
    avg_dd_dur = float(np.mean(dd_lens))   if dd_lens   else 0.0
    max_dd_dur = max(dd_lens) if dd_lens else 0

    return {
        "Total Return":           f"{total_r:.4%}",
        "Geometric CAGR":         f"{cagr:.4%}",
        "Annualised Volatility":  f"{ann_vol:.4%}",
        "Sharpe Ratio":           f"{sharpe:.4f}",
        "Sortino Ratio":          f"{sortino:.4f}",
        "Calmar Ratio":           f"{calmar:.4f}",
        "Omega Ratio":            f"{omega:.4f}" if not np.isnan(omega) else "—",
        "Tail Ratio":             f"{tail_r:.4f}" if not np.isnan(tail_r) else "—",
        "Profit Factor":          f"{pf:.4f}"    if not np.isnan(pf)    else "—",
        "Max Drawdown":           f"{max_dd:.4%}",
        "Avg Drawdown":           f"{avg_dd:.4%}",
        "Max DD Duration (days)": f"{max_dd_dur}",
        "Avg DD Duration (days)": f"{avg_dd_dur:.1f}",
        "VaR 95% (Daily)":        f"{var_95:.4%}",
        "CVaR 95% (Daily)":       f"{cvar_95:.4%}",
        "Skewness":               f"{ret_skew:.4f}",
        "Excess Kurtosis":        f"{ret_kurt:.4f}",
        "Win Rate (Daily)":       f"{win_rate:.4%}",
        "Best Day":               f"{ret.max():.4%}",
        "Worst Day":              f"{ret.min():.4%}",
        "Best Month":             f"{monthly_r.max():.4%}",
        "Worst Month":            f"{monthly_r.min():.4%}",
        "Avg Daily Turnover":     "—",
        "Annual Turnover":        "—",
        "Avg Gross Exposure":     "1.000x",
        "Avg Long Exposure":      "1.000x",
        "Avg Short Exposure":     "0.000x",
    }


def render_strategy_comparison(
    results_linear: pd.DataFrame,
    results_ml: pd.DataFrame,
    metrics_linear: dict,
    metrics_ml: dict,
    benchmark_series: pd.Series | None = None,
    output_dir: Path | None = None,
):
    """
    Overlay equity curves and a four-column metrics table comparing
    the Linear ICIR composite, the ML Ensemble, and the SPY benchmark.

    SPY metrics are computed from benchmark_series over the same date
    range as the backtest results so the comparison is apples-to-apples.
    """
    _style()
    output_dir = output_dir or config.REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_index = results_ml["Equity Curve"].index

    metrics_spy: dict = {}
    bmc: pd.Series | None = None
    if benchmark_series is not None:
        bm      = benchmark_series.reindex(ref_index).ffill()
        bm_ret  = bm.pct_change(fill_method=None).fillna(0)
        bmc     = (1 + bm_ret).cumprod()
        metrics_spy = _compute_benchmark_metrics(benchmark_series, ref_index)

    fig, axes = plt.subplots(1, 2, figsize=(22, 9), facecolor="white")
    fig.suptitle("Strategy Comparison: Linear ICIR vs ML Ensemble vs SPY",
                 fontsize=13, fontweight="bold", y=1.01)

    ax = axes[0]
    ax.plot(results_linear["Equity Curve"].index,
            results_linear["Equity Curve"].values,
            label="Linear ICIR composite", color="#2980B9", linewidth=1.8)
    ax.plot(results_ml["Equity Curve"].index,
            results_ml["Equity Curve"].values,
            label="ML Ensemble (LGBM + Ridge)", color=_C["strategy"], linewidth=1.8)
    if bmc is not None:
        ax.plot(bmc.index, bmc.values, label="SPY (Buy & Hold)",
                color=_C["benchmark"], linestyle="--", linewidth=1.3, alpha=0.85)

    ax.set_title("Equity Curves", fontsize=11, fontweight="bold")
    ax.set_ylabel("Cumulative Return")
    ax.yaxis.set_major_formatter(FuncFormatter(_pct))
    ax.legend(fontsize=9, framealpha=0.7)
    _date_fmt(ax)

    ax2 = axes[1]
    ax2.axis("off")

    keys = [
        "Total Return",
        "Geometric CAGR",
        "Annualised Volatility",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Calmar Ratio",
        "Omega Ratio",
        "Tail Ratio",
        "Profit Factor",
        "Max Drawdown",
        "Avg Drawdown",
        "Max DD Duration (days)",
        "Avg DD Duration (days)",
        "VaR 95% (Daily)",
        "CVaR 95% (Daily)",
        "Skewness",
        "Excess Kurtosis",
        "Win Rate (Daily)",
        "Best Day",
        "Worst Day",
        "Best Month",
        "Worst Month",
        "Avg Daily Turnover",
        "Annual Turnover",
        "Avg Gross Exposure",
        "Avg Long Exposure",
        "Avg Short Exposure",
    ]

    show_keys = [k for k in keys if k in metrics_linear or k in metrics_ml]

    table_data = [
        [
            k,
            metrics_linear.get(k, "—"),
            metrics_ml.get(k, "—"),
            metrics_spy.get(k, "—") if metrics_spy else "—",
        ]
        for k in show_keys
    ]

    col_labels = ["Metric", "Linear ICIR", "ML Ensemble", "SPY (Benchmark)"]

    tbl = ax2.table(
        cellText  = table_data,
        colLabels = col_labels,
        cellLoc   = "center",
        loc       = "center",
        bbox      = [0.0, 0.0, 1.0, 0.97],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.8)

    header_colors = ["#2C3E50", "#1A5276", "#922B21", "#4A4A4A"]

    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor(header_colors[c] if c < len(header_colors) else "#2C3E50")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F2F3F4")
        else:
            cell.set_facecolor("white")
        cell.set_edgecolor("#D5D8DC")
        if r > 0 and c == 0:
            cell.set_text_props(fontweight="bold", color="#2C3E50")

    col_widths = [0.34, 0.22, 0.22, 0.22]
    for (r, c), cell in tbl.get_celld().items():
        cell.set_width(col_widths[c] if c < len(col_widths) else 0.22)

    ax2.set_title("Performance Metrics", fontsize=11, fontweight="bold", pad=12)

    fig.tight_layout()
    out = output_dir / "strategy_comparison.png"
    fig.savefig(out, bbox_inches="tight", dpi=180)
    plt.close(fig)
    logger.info(f"Strategy comparison → {out}")


# ---------------------------------------------------------------------------
# Monthly returns calendar heatmap
# ---------------------------------------------------------------------------

def render_monthly_returns_heatmap(
    results: pd.DataFrame,
    output_dir: Path | None = None,
    title_suffix: str = "",
):
    """
    Two-panel calendar analysis:
      [top]    12 x N-year heatmap of monthly net returns (RdYlGn)
      [bottom] Annual bar chart with win/loss colouring
    """
    _style()
    output_dir = output_dir or config.REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    net_ret = results["Net Return"].copy()
    net_ret.index = pd.to_datetime(net_ret.index)

    monthly = net_ret.resample("ME").apply(lambda r: (1 + r).prod() - 1)
    monthly_df = monthly.to_frame("ret")
    monthly_df["year"]  = monthly_df.index.year
    monthly_df["month"] = monthly_df.index.month

    pivot = monthly_df.pivot(index="year", columns="month", values="ret")
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"]

    annual = net_ret.resample("YE").apply(lambda r: (1 + r).prod() - 1)

    fig = plt.figure(figsize=(16, 8), facecolor="white")
    gs  = gridspec.GridSpec(2, 1, hspace=0.55, figure=fig,
                            height_ratios=[3, 1.2])
    fig.suptitle(f"Monthly & Annual Returns{title_suffix}", fontsize=13,
                 fontweight="bold", y=1.01)

    ax0 = fig.add_subplot(gs[0])
    vmax = max(0.08, pivot.abs().quantile(0.95).max())
    im   = ax0.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                      vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax0.set_yticks(range(len(pivot.index)))
    ax0.set_yticklabels(pivot.index.astype(str), fontsize=8)
    ax0.set_xticks(range(12))
    ax0.set_xticklabels(pivot.columns, fontsize=8)
    ax0.set_title("Monthly Net Returns", fontsize=11, fontweight="bold")

    for (r, c), val in np.ndenumerate(pivot.values):
        if not np.isnan(val):
            txt_color = "white" if abs(val) > vmax * 0.6 else "black"
            ax0.text(c, r, f"{val:.1%}", ha="center", va="center",
                     fontsize=6.5, color=txt_color, fontweight="bold")

    plt.colorbar(im, ax=ax0, label="Monthly Return", fraction=0.015, pad=0.01)

    ax1 = fig.add_subplot(gs[1])
    years  = annual.index.year
    values = annual.values
    colors = [_C["positive"] if v >= 0 else _C["negative"] for v in values]
    bars   = ax1.bar(years, values, color=colors, edgecolor="white",
                     linewidth=0.6, width=0.7)
    ax1.axhline(0, color="#333", linewidth=0.9)
    ax1.set_title("Annual Net Returns", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Annual Return")
    ax1.yaxis.set_major_formatter(FuncFormatter(_pct))
    ax1.set_xticks(years)
    ax1.set_xticklabels(years.astype(str), rotation=45, ha="right", fontsize=8)

    for bar, val in zip(bars, values):
        ha  = "center"
        ypos = val + 0.003 if val >= 0 else val - 0.003
        va  = "bottom" if val >= 0 else "top"
        ax1.text(bar.get_x() + bar.get_width() / 2, ypos, f"{val:.1%}",
                 ha=ha, va=va, fontsize=7.5, color="#333")

    fig.tight_layout()
    out = output_dir / "monthly_returns_heatmap.png"
    fig.savefig(out, bbox_inches="tight", dpi=180)
    plt.close(fig)
    logger.info(f"Monthly returns heatmap → {out}")


# ---------------------------------------------------------------------------
# Cost & turnover decomposition
# ---------------------------------------------------------------------------

def render_cost_turnover_analysis(
    results: pd.DataFrame,
    output_dir: Path | None = None,
    title_suffix: str = "",
):
    """
    Three-panel cost & turnover breakdown:
      [0] Cumulative gross vs net return overlay (cost drag highlighted)
      [1] Rolling 63-day annualised turnover (two-sided)
      [2] Rolling 63-day annualised transaction cost drag (bps)
    """
    _style()
    output_dir = output_dir or config.REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    gross_ret  = results["Gross Return"]
    net_ret    = results["Net Return"]
    cost       = results["Cost"]
    turnover   = results["Turnover"]

    gross_eq = (1 + gross_ret).cumprod()
    net_eq   = (1 + net_ret).cumprod()

    rw = 63
    roll_turnover_ann = turnover.rolling(rw, min_periods=rw // 2).mean() * config.BARS_PER_YEAR
    roll_cost_bps_ann = cost.rolling(rw, min_periods=rw // 2).mean() * config.BARS_PER_YEAR * 1e4

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), facecolor="white")
    fig.suptitle(f"Cost & Turnover Analysis{title_suffix}", fontsize=13,
                 fontweight="bold", y=1.01)

    ax0 = axes[0]
    ax0.plot(gross_eq.index, gross_eq.values, label="Gross Equity Curve",
             color=_C["neutral"], linewidth=1.8, linestyle="--")
    ax0.plot(net_eq.index, net_eq.values, label="Net Equity Curve",
             color=_C["strategy"], linewidth=2.0)
    ax0.fill_between(gross_eq.index, net_eq.values, gross_eq.values,
                     alpha=0.20, color=_C["negative"], label="Cost Drag")
    ax0.set_title("Gross vs Net Equity Curve (Cost Drag)", fontsize=11, fontweight="bold")
    ax0.set_ylabel("Cumulative Return")
    ax0.yaxis.set_major_formatter(FuncFormatter(_pct))
    ax0.legend(fontsize=8.5, framealpha=0.7)
    _date_fmt(ax0)

    total_cost_drag = gross_eq.iloc[-1] / net_eq.iloc[-1] - 1
    ax0.text(0.01, 0.05,
             f"Total cost drag: {total_cost_drag:.2%}  |  "
             f"Avg daily turnover: {turnover.mean():.2%}  |  "
             f"Annual turnover: {turnover.mean() * config.BARS_PER_YEAR:.2f}x",
             transform=ax0.transAxes, fontsize=8.5, color="#444", style="italic")

    ax1 = axes[1]
    ax1.plot(roll_turnover_ann.index, roll_turnover_ann.values,
             color=_C["neutral"], linewidth=1.5,
             label=f"{rw}-day rolling annualised turnover")
    ax1.axhline(roll_turnover_ann.mean(), color="#888", linewidth=0.8,
                linestyle="--", label=f"Mean: {roll_turnover_ann.mean():.2f}x")
    ax1.fill_between(roll_turnover_ann.index, roll_turnover_ann.values,
                     alpha=0.15, color=_C["neutral"])
    ax1.set_title(f"Rolling {rw}-Day Annualised Two-Sided Turnover", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Annualised Turnover (×)")
    ax1.yaxis.set_major_formatter(FuncFormatter(_num))
    ax1.legend(fontsize=8.5, framealpha=0.7)
    _date_fmt(ax1)

    ax2 = axes[2]
    ax2.plot(roll_cost_bps_ann.index, roll_cost_bps_ann.values,
             color=_C["negative"], linewidth=1.5,
             label=f"{rw}-day rolling annualised cost (bps)")
    ax2.fill_between(roll_cost_bps_ann.index, roll_cost_bps_ann.values,
                     alpha=0.20, color=_C["negative"])
    ax2.axhline(roll_cost_bps_ann.mean(), color="#888", linewidth=0.8,
                linestyle="--", label=f"Mean: {roll_cost_bps_ann.mean():.1f} bps/yr")
    ax2.set_title(f"Rolling {rw}-Day Annualised Transaction Cost Drag", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Cost Drag (bps / year)")
    ax2.legend(fontsize=8.5, framealpha=0.7)
    _date_fmt(ax2)

    fig.tight_layout()
    out = output_dir / "cost_turnover_analysis.png"
    fig.savefig(out, bbox_inches="tight", dpi=180)
    plt.close(fig)
    logger.info(f"Cost & turnover analysis → {out}")


# ---------------------------------------------------------------------------
# Drawdown period deep-dive
# ---------------------------------------------------------------------------

def render_drawdown_analysis(
    results: pd.DataFrame,
    output_dir: Path | None = None,
    title_suffix: str = "",
    top_n: int = 5,
):
    """
    Two-panel drawdown analysis:
      [top]    Underwater equity curve with top-N drawdown periods shaded
      [bottom] Summary table of top-N drawdowns (peak, trough, recovery, depth, duration)
    """
    _style()
    output_dir = output_dir or config.REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    eq = results["Equity Curve"]
    dd = results["Drawdown"]

    drawdown_periods = []
    in_dd = False
    peak_date = peak_val = trough_date = trough_val = None

    for date, d in dd.items():
        if d < 0 and not in_dd:
            in_dd    = True
            peak_date = date
            peak_val  = eq.loc[date]
            trough_date = trough_val = None
        if in_dd and d == 0:
            if trough_date is not None:
                drawdown_periods.append({
                    "Peak Date":     peak_date,
                    "Trough Date":   trough_date,
                    "Recovery Date": date,
                    "Depth":         trough_val / peak_val - 1,
                    "Duration (d)":  (trough_date - peak_date).days,
                    "Recovery (d)":  (date - trough_date).days,
                })
            in_dd = False
        if in_dd:
            if trough_date is None or eq.loc[date] < trough_val:
                trough_date = date
                trough_val  = eq.loc[date]

    dd_df = pd.DataFrame(drawdown_periods)
    if dd_df.empty:
        logger.warning("No completed drawdown periods found — skipping render_drawdown_analysis.")
        return

    dd_df = dd_df.nsmallest(top_n, "Depth").reset_index(drop=True)

    fig = plt.figure(figsize=(16, 10), facecolor="white")
    gs  = gridspec.GridSpec(2, 1, hspace=0.55, height_ratios=[2, 1.2], figure=fig)
    fig.suptitle(f"Drawdown Analysis{title_suffix}", fontsize=13,
                 fontweight="bold", y=1.01)

    ax0 = fig.add_subplot(gs[0])
    ax0.fill_between(dd.index, dd.values, 0,
                     color=_C["drawdown"], alpha=0.55, label="Drawdown")
    ax0.plot(dd.index, dd.values, color=_C["drawdown"], linewidth=0.9)

    palette = ["#C0392B", "#8E44AD", "#2980B9", "#27AE60", "#F39C12"]
    for i, row in dd_df.iterrows():
        ax0.axvspan(row["Peak Date"], row["Recovery Date"],
                    alpha=0.12, color=palette[i % len(palette)],
                    label=f"DD#{i+1}: {row['Depth']:.1%}")

    ax0.set_title(f"Portfolio Underwater Chart — Top {top_n} Drawdowns Highlighted",
                  fontsize=11, fontweight="bold")
    ax0.set_ylabel("Drawdown")
    ax0.yaxis.set_major_formatter(FuncFormatter(_pct))
    ax0.set_ylim(top=0.005)
    ax0.legend(fontsize=8, framealpha=0.7, ncol=3)
    _date_fmt(ax0)

    ax1 = fig.add_subplot(gs[1])
    ax1.axis("off")

    table_data = []
    for _, row in dd_df.iterrows():
        table_data.append([
            row["Peak Date"].strftime("%Y-%m-%d"),
            row["Trough Date"].strftime("%Y-%m-%d") if pd.notna(row["Trough Date"]) else "—",
            row["Recovery Date"].strftime("%Y-%m-%d"),
            f"{row['Depth']:.2%}",
            f"{int(row['Duration (d)'])}d",
            f"{int(row['Recovery (d)'])}d",
        ])

    tbl = ax1.table(
        cellText  = table_data,
        colLabels = ["Peak", "Trough", "Recovery", "Depth", "To Trough", "Recovery"],
        cellLoc   = "center",
        loc       = "center",
        bbox      = [0.0, 0.0, 1.0, 1.0],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2C3E50")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#FDFEFE")
        else:
            cell.set_facecolor("#F2F3F4")
        cell.set_edgecolor("#D5D8DC")
        if r > 0 and c == 3:
            cell.set_facecolor("#FADBD8")

    ax1.set_title(f"Top {top_n} Completed Drawdown Periods", fontsize=11,
                  fontweight="bold", pad=8)

    fig.tight_layout()
    out = output_dir / "drawdown_analysis.png"
    fig.savefig(out, bbox_inches="tight", dpi=180)
    plt.close(fig)
    logger.info(f"Drawdown analysis → {out}")


# ---------------------------------------------------------------------------
# Regime-conditional return analysis
# ---------------------------------------------------------------------------

def render_regime_conditional_returns(
    results: pd.DataFrame,
    benchmark_series: pd.Series | None = None,
    vix_series: pd.Series | None = None,
    output_dir: Path | None = None,
    title_suffix: str = "",
):
    """
    Two-panel regime analysis:
      [left]  Bar chart: annualised return / Sharpe by VIX regime
               (Low <20, Moderate 20-30, High >30)
      [right] Bar chart: annualised return by market trend
               (Bull: SPY > 200d SMA, Bear: SPY < 200d SMA)
    """
    _style()
    output_dir = output_dir or config.REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    net_ret = results["Net Return"].copy()
    idx     = net_ret.index

    regime_labels = pd.Series("Moderate VIX (20-30)", index=idx)
    if vix_series is not None:
        aligned_vix = vix_series.reindex(idx).ffill()
        regime_labels[aligned_vix < 20]  = "Low VIX (<20)"
        regime_labels[aligned_vix >= 30] = "High VIX (≥30)"

    vix_regimes    = ["Low VIX (<20)", "Moderate VIX (20-30)", "High VIX (≥30)"]
    vix_colors     = [_C["positive"], _C["neutral"], _C["negative"]]

    def _regime_stats(mask):
        r = net_ret[mask]
        if len(r) < 5:
            return dict(cagr=np.nan, sharpe=np.nan, hit=np.nan, n=0)
        ann   = r.mean() * config.BARS_PER_YEAR
        vol   = r.std() * np.sqrt(config.BARS_PER_YEAR)
        sh    = ann / vol if vol > 0 else np.nan
        hit   = (r > 0).sum() / (r != 0).sum() if (r != 0).sum() > 0 else np.nan
        return dict(cagr=ann, sharpe=sh, hit=hit, n=len(r))

    vix_stats = {lbl: _regime_stats(regime_labels == lbl) for lbl in vix_regimes}

    trend_labels = pd.Series("Bull", index=idx)
    if benchmark_series is not None:
        bm    = benchmark_series.reindex(idx).ffill()
        sma200 = bm.rolling(200, min_periods=100).mean()
        trend_labels[bm < sma200] = "Bear"
    trend_regimes = ["Bull", "Bear"]
    trend_colors  = [_C["positive"], _C["negative"]]
    trend_stats   = {lbl: _regime_stats(trend_labels == lbl) for lbl in trend_regimes}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="white")
    fig.suptitle(f"Regime-Conditional Performance{title_suffix}", fontsize=13,
                 fontweight="bold", y=1.02)

    def _draw_regime_bars(ax, regimes, stats_dict, colors, title):
        x      = np.arange(len(regimes))
        width  = 0.30
        cagrs  = [stats_dict[r]["cagr"]   for r in regimes]
        sharps = [stats_dict[r]["sharpe"] for r in regimes]
        hits   = [stats_dict[r]["hit"]    for r in regimes]

        ax2 = ax.twinx()
        bars1 = ax.bar(x - width, cagrs, width, label="Ann. Return",
                       color=colors, alpha=0.85, edgecolor="white")
        bars2 = ax.bar(x,         sharps, width, label="Sharpe",
                       color=colors, alpha=0.55, edgecolor="white", linewidth=0.8,
                       hatch="///")
        ax2.bar(x + width, hits, width, label="Hit Rate",
                color=colors, alpha=0.35, edgecolor="white", linewidth=0.8,
                hatch="...")

        ax.axhline(0, color="#555", linewidth=0.8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        xlabels = [f"{r}\n(n={stats_dict[r]['n']}d)" for r in regimes]
        ax.set_xticklabels(xlabels, fontsize=8.5)
        ax.set_ylabel("Annualised Return / Sharpe Ratio")
        ax.yaxis.set_major_formatter(FuncFormatter(_pct))
        ax2.set_ylabel("Daily Hit Rate")
        ax2.yaxis.set_major_formatter(FuncFormatter(_pct))
        ax2.set_ylim(0, 1)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  fontsize=8, framealpha=0.7, loc="upper right")

        for bar, val in zip(bars1, cagrs):
            if not np.isnan(val):
                va = "bottom" if val >= 0 else "top"
                ax.text(bar.get_x() + bar.get_width() / 2,
                        val + (0.002 if val >= 0 else -0.002),
                        f"{val:.1%}", ha="center", va=va, fontsize=7.5)

    _draw_regime_bars(axes[0], vix_regimes, vix_stats, vix_colors,
                      "VIX Regime Breakdown (Low / Moderate / High)")
    _draw_regime_bars(axes[1], trend_regimes, trend_stats, trend_colors,
                      "Market Trend Breakdown (Bull vs Bear)")

    fig.tight_layout()
    out = output_dir / "regime_conditional_returns.png"
    fig.savefig(out, bbox_inches="tight", dpi=180)
    plt.close(fig)
    logger.info(f"Regime conditional returns → {out}")


# ---------------------------------------------------------------------------
# Long / short leg P&L attribution
# ---------------------------------------------------------------------------

def render_long_short_attribution(
    results: pd.DataFrame,
    final_weights: pd.DataFrame,
    prices: pd.DataFrame,
    output_dir: Path | None = None,
    title_suffix: str = "",
):
    """
    Three-panel long/short attribution:
      [0] Cumulative long-leg vs short-leg vs combined P&L
      [1] Rolling 63-day contribution split (long / short / net)
      [2] Daily gross exposure: long side, short side, net
    """
    _style()
    output_dir = output_dir or config.REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    daily_rets = prices.pct_change(fill_method=None).fillna(0)
    aligned_w  = final_weights.reindex(index=daily_rets.index,
                                       columns=daily_rets.columns).fillna(0)

    long_w  = aligned_w.clip(lower=0)
    short_w = aligned_w.clip(upper=0)

    long_ret  = (long_w * daily_rets).sum(axis=1)
    short_ret = (short_w * daily_rets).sum(axis=1)
    net_ret   = results["Net Return"].reindex(long_ret.index).fillna(0)

    long_eq  = (1 + long_ret).cumprod()
    short_eq = (1 + short_ret).cumprod()
    net_eq   = (1 + net_ret).cumprod()

    rw = 63
    roll_long  = long_ret.rolling(rw,  min_periods=rw // 2).mean() * config.BARS_PER_YEAR
    roll_short = short_ret.rolling(rw, min_periods=rw // 2).mean() * config.BARS_PER_YEAR
    roll_net   = net_ret.rolling(rw,   min_periods=rw // 2).mean() * config.BARS_PER_YEAR

    long_exp  = long_w.sum(axis=1)
    short_exp = short_w.abs().sum(axis=1)
    net_exp   = long_exp - short_exp

    fig, axes = plt.subplots(3, 1, figsize=(16, 13), facecolor="white")
    fig.suptitle(f"Long / Short Leg Attribution{title_suffix}", fontsize=13,
                 fontweight="bold", y=1.01)

    ax0 = axes[0]
    ax0.plot(long_eq.index,  long_eq.values,  label="Long Leg",
             color=_C["positive"],  linewidth=1.8)
    ax0.plot(short_eq.index, short_eq.values, label="Short Leg",
             color=_C["negative"],  linewidth=1.8)
    ax0.plot(net_eq.index,   net_eq.values,   label="Net (after cost)",
             color=_C["strategy"],  linewidth=2.0, linestyle="--")
    ax0.set_title("Cumulative P&L: Long vs Short vs Net", fontsize=11, fontweight="bold")
    ax0.set_ylabel("Cumulative Return")
    ax0.yaxis.set_major_formatter(FuncFormatter(_pct))
    ax0.legend(fontsize=8.5, framealpha=0.7)
    _date_fmt(ax0)

    ax1 = axes[1]
    ax1.plot(roll_long.index,  roll_long.values,  label="Long contribution (ann.)",
             color=_C["positive"],  linewidth=1.5)
    ax1.plot(roll_short.index, roll_short.values, label="Short contribution (ann.)",
             color=_C["negative"],  linewidth=1.5)
    ax1.plot(roll_net.index,   roll_net.values,   label="Net return (ann.)",
             color=_C["strategy"],  linewidth=1.8, linestyle="--")
    ax1.axhline(0, color="#555", linewidth=0.8, linestyle="--")
    ax1.fill_between(roll_long.index,  roll_long.values,  0,
                     alpha=0.10, color=_C["positive"])
    ax1.fill_between(roll_short.index, roll_short.values, 0,
                     alpha=0.10, color=_C["negative"])
    ax1.set_title(f"Rolling {rw}-Day Annualised Contribution Split",
                  fontsize=11, fontweight="bold")
    ax1.set_ylabel("Ann. Contribution")
    ax1.yaxis.set_major_formatter(FuncFormatter(_pct))
    ax1.legend(fontsize=8.5, framealpha=0.7)
    _date_fmt(ax1)

    ax2 = axes[2]
    ax2.plot(long_exp.index,  long_exp.values,  label="Long exposure",
             color=_C["positive"],  linewidth=1.3)
    ax2.plot(short_exp.index, short_exp.values, label="Short exposure",
             color=_C["negative"],  linewidth=1.3)
    ax2.plot(net_exp.index,   net_exp.values,   label="Net exposure",
             color=_C["neutral"],   linewidth=1.5, linestyle="--")
    ax2.fill_between(long_exp.index,  0, long_exp.values,  alpha=0.10, color=_C["positive"])
    ax2.fill_between(short_exp.index, 0, short_exp.values, alpha=0.10, color=_C["negative"])
    ax2.axhline(0, color="#555", linewidth=0.8)
    ax2.set_title("Daily Gross Exposure: Long / Short / Net", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Exposure (×)")
    ax2.yaxis.set_major_formatter(FuncFormatter(_num))
    ax2.legend(fontsize=8.5, framealpha=0.7)
    _date_fmt(ax2)

    fig.tight_layout()
    out = output_dir / "long_short_attribution.png"
    fig.savefig(out, bbox_inches="tight", dpi=180)
    plt.close(fig)
    logger.info(f"Long/short attribution → {out}")
