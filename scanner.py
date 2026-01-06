import yfinance as yf
import pandas as pd
import numpy as np
from ta.volatility import AverageTrueRange
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from scipy.stats import linregress
from datetime import datetime

# ---------------- CONFIG ---------------- #
LOOKBACK = 90
TOP_N = 10
MARKET = "SPY"
MIN_DOLLAR_VOLUME = 50_000_000  # liquidity filter

UNIVERSE = [
    # Index ETFs
    "SPY", "QQQ", "DIA", "IWM",

    # Large-cap stocks
    "AAPL", "MSFT", "NVDA", "TSLA", "AMD",
    "META", "AMZN", "GOOGL", "NFLX",

    # Mid/Small-cap example
    "PLTR", "SOFI", "RIVN"
]

# ---------------- HELPERS ---------------- #
def load(symbol):
    df = yf.download(symbol, period=f"{LOOKBACK}d", progress=False)
    if df is None or df.empty or len(df) < 40:
        return None
    return df.dropna()

def slope(series):
    x = np.arange(len(series))
    return linregress(x, series).slope

def normalize(value, low, high):
    return max(0, min(1, (value - low) / (high - low)))

# ---------------- SCORING ---------------- #
def score_stock(symbol, market_df):
    df = load(symbol)
    if df is None or df.empty:
        return None

    # Safe retrieval of last close & volume
    try:
        last_close = float(df["Close"].iloc[-1])
        last_volume = float(df["Volume"].iloc[-1])
    except:
        return None

    dollar_volume = last_close * last_volume
    if dollar_volume < MIN_DOLLAR_VOLUME:
        return None  # liquidity filter

    # Intraday return
    ir = (df["Close"] - df["Open"]) / df["Open"]

    # Indicators
    atr = AverageTrueRange(df["High"], df["Low"], df["Close"], 14).average_true_range()
    rsi = RSIIndicator(df["Close"], 14).rsi()
    ema9 = EMAIndicator(df["Close"], 9).ema_indicator()
    ema21 = EMAIndicator(df["Close"], 21).ema_indicator()
    ema50 = EMAIndicator(df["Close"], 50).ema_indicator()

    rel_vol = last_volume / df["Volume"].rolling(20).mean().iloc[-1]

    # Beta vs market
    try:
        beta = df["Close"].pct_change().cov(
            market_df["Close"].pct_change()
        ) / market_df["Close"].pct_change().var()
    except:
        beta = 1  # default if calculation fails

    # Scoring
    score = 0

    # Intraday behavior
    score += normalize(ir.mean(), -0.002, 0.004) * 3
    score += normalize((ir > 0).mean(), 0.4, 0.65) * 2
    score += normalize(slope(ir.tail(20)), -0.0005, 0.0005) * 2

    # Volatility
    score += normalize(atr.iloc[-1] / last_close, 0.01, 0.05) * 2

    # Trend
    score += (ema9.iloc[-1] > ema21.iloc[-1]) * 1
    score += (ema21.iloc[-1] > ema50.iloc[-1]) * 1

    # Momentum sanity (avoid overbought)
    if rsi.iloc[-1] < 70:
        score += normalize(rsi.iloc[-1], 40, 65) * 1

    # Volume confirmation
    score += normalize(rel_vol, 0.8, 2.5) * 2

    # Market alignment
    score += (beta > 1) * 1
    score += (df["Close"].pct_change().mean() >
              market_df["Close"].pct_change().mean()) * 1

    return {
        "Symbol": symbol,
        "Score": round(score * 10, 2),
        "RSI": round(rsi.iloc[-1], 1),
        "ATR%": round(atr.iloc[-1] / last_close, 3),
        "RelVolume": round(rel_vol, 2),
        "DollarVolume(M)": round(dollar_volume / 1e6, 1)
    }

# ---------------- RUN SCAN ---------------- #
def run():
    market_df = load(MARKET)
    if market_df is None:
        print("Error: Market data could not be loaded")
        return pd.DataFrame()

    results = []
    for sym in UNIVERSE:
        r = score_stock(sym, market_df)
        if r:
            results.append(r)

    df = pd.DataFrame(results).sort_values("Score", ascending=False).head(TOP_N)

    # Save CSV with today's date
    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"intraday_picks_{today}.csv"
    df.to_csv(filename, index=False)
    print(f"\n✅ CSV saved: {filename}")
    return df

# ---------------- MAIN ---------------- #
if __name__ == "__main__":
    df = run()
    if not df.empty:
        print(df.to_string(index=False))
    else:
        print("No valid picks today.")

    
  
