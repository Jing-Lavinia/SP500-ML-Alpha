"""
config.py
---------
Central configuration for the SP500 quantitative factor pipeline.
Defines directory layout, universe tickers, sector mappings, backtest
parameters, ICIR synthesis settings, and ML pipeline hyperparameters.
All other modules import constants from here -- change values here to
reconfigure the entire pipeline.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE_DIR / "data" / "raw"
REPORTS_DIR = BASE_DIR / "reports"

PRICES_FILE    = DATA_DIR / "sp500_prices_daily.csv"
VOLUMES_FILE   = DATA_DIR / "sp500_volumes_daily.csv"
VIX_FILE       = DATA_DIR / "vix_daily.csv"
BENCHMARK_FILE = DATA_DIR / "spy_daily.csv"

# ---------------------------------------------------------------------------
# Time windows
# ---------------------------------------------------------------------------
DATA_START_DATE     = "2018-01-01"   # Full download start (warm-up for long-window factors)
BACKTEST_START_DATE = "2020-09-01"   # Evaluation window start (post warm-up)
END_DATE            = "2026-03-31"

# ---------------------------------------------------------------------------
# Universe  (~174 liquid S&P 500 names across 10 sectors)
# ---------------------------------------------------------------------------
SP500_TICKERS = [
    # Technology (38)
    "AAPL","MSFT","NVDA","AVGO","ORCL","ADBE","CSCO","ACN","INTC","TXN",
    "QCOM","AMD","AMAT","KLAC","LRCX","MU","ADI","MCHP","CDNS","SNPS",
    "CRM","NOW","INTU","PANW","FTNT","CRWD","WDAY","DDOG","ZS","TEAM",
    "SHOP","PLTR","APP","NFLX","META","GOOGL","AMZN","TSLA",
    # Financials (20)
    "JPM","BAC","WFC","GS","MS","BLK","C","AXP","SPGI","MCO",
    "ICE","CME","CB","PGR","TRV","MMC","AON","MET","PRU","AIG",
    # Healthcare (20)
    "UNH","JNJ","LLY","ABBV","MRK","TMO","ABT","DHR","ISRG","VRTX",
    "REGN","AMGN","GILD","BSX","EW","DXCM","IQV","A","ZBH","IDXX",
    # Consumer Discretionary (20)
    "BKNG","HD","MCD","SBUX","NKE","LOW","TJX","ORLY","AZO","TSCO",
    "ROST","YUM","DRI","CMG","ULTA","EBAY","ETSY","W","DHI","LEN",
    # Consumer Staples (20)
    "WMT","COST","PG","KO","PEP","PM","MO","MDLZ","CL","CHD",
    "KMB","SJM","CAG","K","CPB","GIS","HRL","TSN","KHC","MNST",
    # Industrials (20)
    "GE","HON","CAT","UPS","DE","RTX","LMT","BA","NOC","GD",
    "FDX","CSX","NSC","UNP","CTAS","FAST","ROK","EMR","ETN","ITW",
    # Energy (10)
    "XOM","CVX","COP","EOG","SLB","PXD","MPC","PSX","VLO","HES",
    # Utilities (10)
    "NEE","DUK","SO","AEP","EXC","D","PEG","XEL","PCG","SRE",
    # Materials (10)
    "LIN","APD","SHW","ECL","PPG","NEM","FCX","NUE","CF","MOS",
    # Communication (10)
    "TMUS","VZ","T","CMCSA","CHTR","DIS","PARA","WBD","OMC","IPG",
]

# ---------------------------------------------------------------------------
# Sector map  (used for neutralisation and Alphalens groupby)
# ---------------------------------------------------------------------------
SECTOR_MAP = {
    **{t: "Technology"    for t in ["AAPL","MSFT","NVDA","AVGO","ORCL","ADBE","CSCO","ACN","INTC","TXN","QCOM","AMD","AMAT","KLAC","LRCX","MU","ADI","MCHP","CDNS","SNPS","CRM","NOW","INTU","PANW","FTNT","CRWD","WDAY","DDOG","ZS","TEAM","SHOP","PLTR","APP","NFLX","META","GOOGL","AMZN","TSLA"]},
    **{t: "Financials"    for t in ["JPM","BAC","WFC","GS","MS","BLK","C","AXP","SPGI","MCO","ICE","CME","CB","PGR","TRV","MMC","AON","MET","PRU","AIG"]},
    **{t: "Healthcare"    for t in ["UNH","JNJ","LLY","ABBV","MRK","TMO","ABT","DHR","ISRG","VRTX","REGN","AMGN","GILD","BSX","EW","DXCM","IQV","A","ZBH","IDXX"]},
    **{t: "ConsumerDisc"  for t in ["BKNG","HD","MCD","SBUX","NKE","LOW","TJX","ORLY","AZO","TSCO","ROST","YUM","DRI","CMG","ULTA","EBAY","ETSY","W","DHI","LEN"]},
    **{t: "ConsumerStap"  for t in ["WMT","COST","PG","KO","PEP","PM","MO","MDLZ","CL","CHD","KMB","SJM","CAG","K","CPB","GIS","HRL","TSN","KHC","MNST"]},
    **{t: "Industrials"   for t in ["GE","HON","CAT","UPS","DE","RTX","LMT","BA","NOC","GD","FDX","CSX","NSC","UNP","CTAS","FAST","ROK","EMR","ETN","ITW"]},
    **{t: "Energy"        for t in ["XOM","CVX","COP","EOG","SLB","PXD","MPC","PSX","VLO","HES"]},
    **{t: "Utilities"     for t in ["NEE","DUK","SO","AEP","EXC","D","PEG","XEL","PCG","SRE"]},
    **{t: "Materials"     for t in ["LIN","APD","SHW","ECL","PPG","NEM","FCX","NUE","CF","MOS"]},
    **{t: "Communication" for t in ["TMUS","VZ","T","CMCSA","CHTR","DIS","PARA","WBD","OMC","IPG"]},
}

# ---------------------------------------------------------------------------
# Backtest parameters
# ---------------------------------------------------------------------------
HOLDING_PERIOD        = 5        # Rebalance every 5 trading days (weekly)
TRANSACTION_COST      = 0.0005   # 5 bps round-trip per unit turnover
STOP_LOSS_PCT         = 1.0      # Disabled -- sector-neutral L/S incompatible with single-leg stops
LONG_SHORT_QUANTILE   = 0.10     # Top/bottom 10% per sector -> long/short
TARGET_ANNUAL_VOL     = 0.12     # 12% annualised volatility target
VIX_DELEVERAGE_LEVEL  = 25.0     # VIX threshold to begin deleveraging

# ---------------------------------------------------------------------------
# Dynamic ICIR synthesis parameters  (linear fallback mode)
# ---------------------------------------------------------------------------
# Three lookback windows are blended to avoid weight flip-flopping from short-window noise
ICIR_LOOKBACK_DAYS  = 252   # Primary window: 1 full year, spans a complete bull/bear cycle
ICIR_LOOKBACK_SHORT = 63    # Short window: ~3 months, captures recent regime shifts
ICIR_LOOKBACK_MED   = 126   # Medium window: ~6 months
ICIR_WINDOW_WEIGHTS = (0.2, 0.3, 0.5)  # short/med/long blend weights; long dominates
ICIR_SMOOTH_SPAN    = 21    # EWMA span to smooth ICIR series
ICIR_MIN_PERIODS    = 42    # Minimum IC observations before computing ICIR
ICIR_MAX_CLIP       = 2.5   # Clip ICIR to prevent single-factor dominance
ICIR_FORWARD_DAYS   = 5     # Forward-return horizon for IC computation (align with holding period)

# IC sign-consistency protection: suspend a factor if its monthly IC is negative for N consecutive months
IC_SIGN_PROTECTION_WINDOW = 3   # N consecutive negative months triggers suspension
IC_SIGN_PROTECTION_FACTORS = [  # Only applied to directionally unstable factors
    "LowVol_63", "ParkinsonVol_21", "ParkinsonVol_63",
]

# ---------------------------------------------------------------------------
# ML pipeline parameters
# ---------------------------------------------------------------------------
ML_FORWARD_DAYS     = 5     # Prediction target: N-day forward return
ML_TRAIN_YEARS      = 1.5   # Training window length in years
ML_RETRAIN_FREQ     = 63    # Retrain every ~1 quarter (trading days)
ML_EMBARGO_DAYS     = 10    # Gap between train end and test start (leakage prevention)
ML_MIN_TRAIN_ROWS   = 5000  # Minimum training samples before activating ML

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
FORWARD_PERIODS = [1, 5, 10, 21]   # Alphalens IC analysis horizons
BARS_PER_YEAR   = 252
