"""
data_loader.py
--------------
Unified data fetching module.
Harvests S&P 500 daily OHLCV data, News Sentiment, VIX, Benchmarks,
and Macro-economic indicators via Alpaca and Yahoo Finance.

All fetch functions are cache-aware: they write to DATA_DIR on first run
and reload from disk on subsequent runs unless force_refresh=True.
"""

import os
import time
import pandas as pd
import yfinance as yf
from pathlib import Path
from pandas.tseries.offsets import BDay
from alpaca.data.historical import StockHistoricalDataClient, NewsClient
from alpaca.data.requests import StockBarsRequest, NewsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.requests import StockBarsRequest
from alpaca.data.enums import Adjustment
from .utils import logger
from . import config

PROXY = "http://127.0.0.1:7890"
os.environ['http_proxy'] = PROXY
os.environ['https_proxy'] = PROXY

ALPACA_API_KEY = 'YOUR_ALPACA_API_KEY'
ALPACA_SECRET_KEY = 'YOUR_ALPACA_SECRET_KEY'

alpaca_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

def fetch_alpaca_news_sentiment(tickers, start, end, force_refresh=False):
    out_file = config.DATA_DIR / "alpaca_news_sentiment.csv"

    if out_file.exists() and not force_refresh:
        logger.info("Loading cached Alpaca News Sentiment data...")
        return pd.read_csv(out_file, index_col=["date", "ticker"], parse_dates=["date"])

    # raw_data=True forces the SDK to return a plain dict, avoiding object attribute errors
    client = NewsClient(api_key=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY, raw_data=True)
    logger.info(f"Downloading Alpaca News Sentiment for {len(tickers)} tickers...")

    sentiment_data = []
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    for i in range(0, len(tickers), 10):
        batch = tickers[i:i+10]
        symbols_str = ",".join(batch)

        request_params = NewsRequest(
            symbols=symbols_str,
            start=start_dt,
            end=end_dt,
            limit=5000
        )

        try:
            response = client.get_news(request_params)
            articles_list = response.get("news", [])

            for article in articles_list:
                for symbol in article.get("symbols", []):
                    if symbol in batch:

                        headline = article.get("headline", "")
                        created_at = article.get("created_at")

                        raw_time = pd.to_datetime(created_at)
                        if raw_time.tzinfo is None:
                            raw_time = raw_time.tz_localize('UTC')
                        est_time = raw_time.tz_convert('America/New_York')

                        # After-hours news (>=16:00 ET) is attributed to the next trading day
                        if est_time.hour >= 16:
                            trade_date = (est_time + BDay(1)).date()
                        else:
                            trade_date = est_time.date()

                        sentiment_data.append({
                            "date": trade_date,
                            "ticker": symbol,
                            "impact": 1.0 if len(headline) > 50 else 0.5
                        })

            logger.info(f"  -> Success: Batch {batch[0]} to {batch[-1]} | Articles fetched: {len(articles_list)}")
            time.sleep(1)

        except Exception as e:
            logger.warning(f"Could not fetch news for batch {batch[:3]}...: {e}")

    df = pd.DataFrame(sentiment_data)
    if df.empty:
        logger.warning("Alpaca returned no news data.")
        return pd.DataFrame()

    df['date'] = pd.to_datetime(df['date'])
    sentiment_matrix = df.groupby(["date", "ticker"])["impact"].mean().unstack().fillna(0)
    sentiment_smoothed = sentiment_matrix.ewm(span=3).mean()

    final_df = sentiment_smoothed.stack().reset_index()
    final_df.columns = ["date", "ticker", "news_sentiment_score"]
    final_df.set_index(["date", "ticker"], inplace=True)

    final_df.to_csv(out_file)
    logger.info("Alpaca news data saved successfully.")

    return final_df

def _download_batch_alpaca(tickers: list[str], start: str, end: str, retries: int = 3):
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    for attempt in range(1, retries + 1):
        try:
            request_params = StockBarsRequest(
                symbol_or_symbols=tickers,
                timeframe=TimeFrame.Day,
                start=start_dt,
                end=end_dt,
                adjustment=Adjustment.ALL  # split- and dividend-adjusted prices
            )

            bars = alpaca_client.get_stock_bars(request_params)
            if bars.df.empty: raise ValueError("Alpaca returned empty DataFrame")

            df = bars.df.copy().reset_index()
            df['timestamp'] = df['timestamp'].dt.tz_convert('America/New_York').dt.tz_localize(None).dt.normalize()

            # Pivot long -> wide for each OHLCV dimension
            open_df   = df.pivot(index='timestamp', columns='symbol', values='open')
            high_df   = df.pivot(index='timestamp', columns='symbol', values='high')
            low_df    = df.pivot(index='timestamp', columns='symbol', values='low')
            close_df  = df.pivot(index='timestamp', columns='symbol', values='close')
            volume_df = df.pivot(index='timestamp', columns='symbol', values='volume')

            for d in [open_df, high_df, low_df, close_df, volume_df]:
                d.index.name = "Date"

            return open_df, high_df, low_df, close_df, volume_df

        except Exception as exc:
            logger.warning(f"Alpaca batch download attempt {attempt}/{retries} failed: {exc}")
            time.sleep(5 * attempt)

    raise RuntimeError(f"All {retries} Alpaca download attempts failed for batch {tickers[:3]}...")

def fetch_universe(tickers=None, start=None, end=None, batch_size=50, force_refresh=False):
    tickers = tickers or config.SP500_TICKERS
    start   = start   or config.DATA_START_DATE
    end     = end     or config.END_DATE

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    file_o = config.DATA_DIR / "sp500_open_daily.csv"
    file_h = config.DATA_DIR / "sp500_high_daily.csv"
    file_l = config.DATA_DIR / "sp500_low_daily.csv"
    file_c = config.PRICES_FILE
    file_v = config.VOLUMES_FILE

    if all(f.exists() for f in [file_o, file_h, file_l, file_c, file_v]) and not force_refresh:
        logger.info("Cached HLOCV files found -- loading from disk.")
        return (pd.read_csv(f, index_col=0, parse_dates=True) for f in [file_o, file_h, file_l, file_c, file_v])

    logger.info(f"Downloading HLOCV for {len(tickers)} tickers via Alpaca...")

    all_o, all_h, all_l, all_c, all_v = [], [], [], [], []
    batches = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]

    for idx, batch in enumerate(batches, 1):
        logger.info(f"  Batch {idx}/{len(batches)}: {batch[:5]}...")
        o, h, l, c, v = _download_batch_alpaca(batch, start, end)
        all_o.append(o); all_h.append(h); all_l.append(l); all_c.append(c); all_v.append(v)
        time.sleep(1)

    res = []
    for i, data_list in enumerate([all_o, all_h, all_l, all_c, all_v]):
        df = pd.concat(data_list, axis=1).sort_index()
        valid_tickers = [t for t in tickers if t in df.columns]
        df = df[valid_tickers]

        # Forward-fill price gaps up to 5 days (handles trading halts); volumes are NOT filled
        if i < 4:
            df = df.ffill(limit=5)

        res.append(df)

    o_df, h_df, l_df, c_df, v_df = res

    # Drop tickers with >30% missing close prices
    max_nans = int(0.30 * len(c_df))
    valid_cols = c_df.columns[c_df.isna().sum() <= max_nans]

    o_df = o_df[valid_cols]
    h_df = h_df[valid_cols]
    l_df = l_df[valid_cols]
    c_df = c_df[valid_cols]
    v_df = v_df[valid_cols].fillna(0)

    logger.info(f"Final universe: {c_df.shape[1]} tickers x {len(c_df)} dates")

    o_df.to_csv(file_o); h_df.to_csv(file_h); l_df.to_csv(file_l); c_df.to_csv(file_c); v_df.to_csv(file_v)
    return o_df, h_df, l_df, c_df, v_df

def fetch_vix(start=None, end=None, force_refresh=False):
    if config.VIX_FILE.exists() and not force_refresh:
        return pd.read_csv(config.VIX_FILE, index_col=0, parse_dates=True)["VIX"]
    start = start or config.DATA_START_DATE; end = end or config.END_DATE
    df = yf.download("^VIX", start=start, end=end, interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    vix_series = df['Close'].rename('VIX')
    vix_series.index.name = 'Date'
    vix_series.to_frame().to_csv(config.VIX_FILE)
    return vix_series

def fetch_benchmark(start=None, end=None, force_refresh=False):
    if config.BENCHMARK_FILE.exists() and not force_refresh:
        return pd.read_csv(config.BENCHMARK_FILE, index_col=0, parse_dates=True)["SPY"]
    start = start or config.DATA_START_DATE; end = end or config.END_DATE
    _, _, _, close, _ = _download_batch_alpaca(["SPY"], start, end)
    spy = close["SPY"].rename("SPY")
    spy.to_frame().to_csv(config.BENCHMARK_FILE)
    return spy

def fetch_macro_and_flows(start_date: str = None, end_date: str = None, force_refresh: bool = False) -> pd.DataFrame:
    start_date = start_date or config.DATA_START_DATE
    end_date = end_date or config.END_DATE

    out_file = config.DATA_DIR / "macro_alt_features.csv"

    if out_file.exists() and not force_refresh:
        logger.info("Loading cached macro alternative data...")
        return pd.read_csv(out_file, index_col=0, parse_dates=True)

    logger.info("Downloading macro and sector flow data from Yahoo Finance...")

    proxy_tickers = {
        "^TNX": "treasury_10y",
        "HYG":  "high_yield_credit",
        "XLK":  "tech_sector_flow",
        "SPY":  "market_baseline"
    }

    raw_data = yf.download(
        list(proxy_tickers.keys()),
        start=start_date,
        end=end_date,
        interval="1d",
        progress=False
    )["Close"]

    if isinstance(raw_data.columns, pd.MultiIndex):
        raw_data.columns = raw_data.columns.get_level_values(0)

    raw_data.rename(columns=proxy_tickers, inplace=True)
    raw_data.ffill(inplace=True)

    macro_features = pd.DataFrame(index=raw_data.index)

    # 5-day rate shock: captures sudden bond market moves
    macro_features["rate_shock_5d"] = raw_data["treasury_10y"].diff(5)
    # 21-day credit spread trend: proxy for risk appetite
    macro_features["credit_spread_trend"] = raw_data["high_yield_credit"].pct_change(21)

    tech_ret_21d = raw_data["tech_sector_flow"].pct_change(21)
    spy_ret_21d = raw_data["market_baseline"].pct_change(21)
    macro_features["tech_relative_strength_21d"] = tech_ret_21d - spy_ret_21d
    macro_features["tech_volume_shock"] = raw_data["tech_sector_flow"].pct_change(5)

    macro_features.fillna(0.0, inplace=True)

    macro_features.to_csv(out_file)
    logger.info(f"Macro alternative features saved: {macro_features.shape[1]} columns.")

    return macro_features
