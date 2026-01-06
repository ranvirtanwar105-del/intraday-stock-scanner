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
    "SPY", "QQQ", "DIA", "IWM",
    "AAPL", "MSFT", "NVDA", "TSLA", "AMD",
    "META", "AMZN", "GOOGL", "NFLX",
    "PLTR", "SOFI", "RIVN"
]

# ---------------- HELPERS ---------------- #
def load(symbol):
    df = yf.download(symbol, period=f"{LOOKBACK}d", progress=False)
    if df.empty or len(df) < 40:
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
    if df is None:
        return None

    dollar_volume = df["Close"].iloc[-1] * df["Volume"].iloc[-1]
    if dollar_volume < MIN_DOLLAR_VOLUME:
        return None  # hard liquidity filter

    ir = (df["Close"] - df["Open"]) / df["Open"]

    atr = AverageTrueRange(df["High"], df["Low"], df["Close"], 14).average_true_range()
    rsi = RSIIndicator(df["Close"], 14).rsi()
    ema9 = EMAIndicator(df["Close"], 9).ema_indicator()
    ema21 = EMAIndicator(df["Close"], 21).ema_indicator()
    ema50 = EMAIndicator(df["Close"], 50).ema_indicator()

    rel_vol = df["Volume"].iloc[-1] / df["Volume"].rolling(20).mean().iloc[-1]
    beta = df["Close"].pct_change().cov(
        market_df["Close"].pct_change()
    ) / market_df["Close"].pct_change().var()

    score = 0

    # Intraday behavior
    score += normalize(ir.mean(), -0.002, 0.004) * 3
    score += normalize((ir > 0).mean(), 0.4, 0.65) * 2
    score += normalize(slope(ir.tail(20)), -0.0005, 0.0005) * 2

    # Volatility (needed, but not extreme)
    score += normalize(atr.iloc[-1] / df["Close"].iloc[-1], 0.01, 0.05) * 2

    # Trend
    score += (ema9.iloc[-1] > ema21.iloc[-1]) * 1
    score += (ema21.iloc[-1] > ema50.iloc[-1]) * 1

    # Momentum sanity check (penalize extremes)
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
        "ATR%": round(atr.iloc[-1] / df["Close"].iloc[-1], 3),
        "RelVolume": round(rel_vol, 2),
        "DollarVolume(M)": round(dollar_volume / 1e6, 1)
    }

# ---------------- RUN ---------------- #
def run():
    market_df = load(MARKET)
    results = []

    for sym in UNIVERSE:
        r = score_stock(sym, market_df)
        if r:
            results.append(r)

    df = pd.DataFrame(results).sort_values("Score", ascending=False).head(TOP_N)

    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"intraday_picks_{today}.csv"
    df.to_csv(filename, index=False)

    print(f"\nSaved results to {filename}\n")
    return df

if __name__ == "__main__":
    print(run().to_string(index=False))
