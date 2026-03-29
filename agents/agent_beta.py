import yfinance as yf
import pandas as pd
import numpy as np

# ============================================================
# AGENT BETA — Technical Analyzer 
#   - 7 indicators instead of 2 (RSI, MACD, BB, SMA Cross,
#     Stochastic, ATR, Volume)
#   - Weighted confluence scoring (continuous, not binary)
#   - Fixed fillna(0) bug — warmup NaNs handled correctly
#   - Multi-timeframe context (short + long term)
#   - Volume confirmation layer
#   - Richer key_driver explanation
# ============================================================


# ---------- Indicator Calculators ----------

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_bollinger(series: pd.Series, period=20, std_dev=2):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    # %B: where price is within the band (0 = lower, 1 = upper)
    pct_b = (series - lower) / (upper - lower).replace(0, np.nan)
    return upper, lower, pct_b


def calc_stochastic(df: pd.DataFrame, k_period=14, d_period=3):
    low_min = df['Low'].rolling(k_period).min()
    high_max = df['High'].rolling(k_period).max()
    k = 100 * (df['Close'] - low_min) / (high_max - low_min).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k, d


def calc_atr(df: pd.DataFrame, period=14) -> pd.Series:
    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Close'].shift()).abs()
    lc = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


# ---------- Scoring Helpers ----------

def score_rsi(rsi: float) -> tuple[float, str]:
    """Returns (contribution -1..+1, signal_text)"""
    if rsi < 25:
        return +1.0, f"Deeply oversold (RSI {rsi:.1f}) — high bounce potential"
    elif rsi < 35:
        return +0.6, f"Oversold (RSI {rsi:.1f}) — mild buy signal"
    elif rsi > 75:
        return -1.0, f"Deeply overbought (RSI {rsi:.1f}) — pullback risk"
    elif rsi > 65:
        return -0.6, f"Overbought (RSI {rsi:.1f}) — caution"
    elif 45 <= rsi <= 55:
        return 0.0, f"Neutral RSI ({rsi:.1f})"
    elif rsi > 55:
        return +0.3, f"RSI bullish zone ({rsi:.1f})"
    else:
        return -0.3, f"RSI bearish zone ({rsi:.1f})"


def score_macd(macd_val: float, histogram: float) -> tuple[float, str]:
    if macd_val > 0 and histogram > 0:
        return +0.8, "MACD bullish: line positive & histogram expanding"
    elif macd_val > 0 and histogram < 0:
        return +0.2, "MACD mildly bullish: positive but momentum fading"
    elif macd_val < 0 and histogram < 0:
        return -0.8, "MACD bearish: line negative & histogram falling"
    elif macd_val < 0 and histogram > 0:
        return -0.2, "MACD recovering but still in negative territory"
    return 0.0, "MACD neutral"


def score_bollinger(pct_b: float) -> tuple[float, str]:
    if pct_b < 0:
        return +0.9, f"Price below lower Bollinger Band (%B={pct_b:.2f}) — mean reversion likely"
    elif pct_b < 0.2:
        return +0.5, f"Price near lower band (%B={pct_b:.2f}) — support zone"
    elif pct_b > 1.0:
        return -0.9, f"Price above upper Bollinger Band (%B={pct_b:.2f}) — overextended"
    elif pct_b > 0.8:
        return -0.5, f"Price near upper band (%B={pct_b:.2f}) — resistance zone"
    return 0.0, f"Price mid-band (%B={pct_b:.2f}) — neutral"


def score_sma_cross(price: float, sma50: float, sma200: float) -> tuple[float, str]:
    above50 = price > sma50
    above200 = price > sma200
    golden = sma50 > sma200  # Golden cross in effect
    if above50 and above200 and golden:
        return +0.8, "Golden cross active: price above both SMAs — strong uptrend"
    elif above50 and above200:
        return +0.4, "Price above both SMAs — medium-term bullish"
    elif not above50 and not above200 and not golden:
        return -0.8, "Death cross active: price below both SMAs — downtrend"
    elif not above50 and not above200:
        return -0.4, "Price below both SMAs — bearish"
    return 0.0, "Mixed SMA signals — consolidation"


def score_stochastic(k: float, d: float) -> tuple[float, str]:
    if k < 20 and d < 20:
        return +0.7, f"Stochastic oversold (K={k:.1f}, D={d:.1f}) — potential reversal"
    elif k > 80 and d > 80:
        return -0.7, f"Stochastic overbought (K={k:.1f}, D={d:.1f}) — potential reversal"
    elif k > d and k < 50:
        return +0.2, "Stochastic crossing up from low — early bullish"
    elif k < d and k > 50:
        return -0.2, "Stochastic crossing down from high — early bearish"
    return 0.0, "Stochastic neutral"


def score_volume(latest_vol: float, avg_vol: float, price_change: float) -> tuple[float, str]:
    ratio = latest_vol / avg_vol if avg_vol > 0 else 1.0
    if ratio > 1.5 and price_change > 0:
        return +0.6, f"High volume ({ratio:.1f}x avg) on up move — confirmed bullish"
    elif ratio > 1.5 and price_change < 0:
        return -0.6, f"High volume ({ratio:.1f}x avg) on down move — confirmed bearish"
    elif ratio < 0.7:
        return 0.0, f"Low volume ({ratio:.1f}x avg) — weak conviction"
    return 0.0, f"Average volume ({ratio:.1f}x avg) — neutral"


# ---------- Main Agent Function ----------

def analyze_technicals(ticker: str) -> dict:
    """
    Fetches historical data and calculates 7 technical indicators.
    Returns a JSON contract compatible with the Master Node.
    """
    print(f"[Agent Beta v2] Fetching market data for {ticker}...")

    try:
        stock = yf.Ticker(ticker)

        # Fetch enough data for 200-day SMA
        df = stock.history(period="1y")
        if df.empty or len(df) < 30:
            raise ValueError("Insufficient price data. Ticker might be invalid.")

        print("[Agent Beta v2] Computing 7 indicators with confluence scoring...")

        close = df['Close']
        price = float(close.iloc[-1])
        prev_price = float(close.iloc[-2])
        price_change = price - prev_price

        # --- Calculate all indicators ---
        rsi = calc_rsi(close)
        macd_line, signal_line, histogram = calc_macd(close)
        bb_upper, bb_lower, pct_b = calc_bollinger(close)
        stoch_k, stoch_d = calc_stochastic(df)
        atr = calc_atr(df)

        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()

        # --- Extract latest values (safe — no fillna(0) poisoning) ---
        rsi_val = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
        macd_val = float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0.0
        hist_val = float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else 0.0
        pct_b_val = float(pct_b.iloc[-1]) if not pd.isna(pct_b.iloc[-1]) else 0.5
        k_val = float(stoch_k.iloc[-1]) if not pd.isna(stoch_k.iloc[-1]) else 50.0
        d_val = float(stoch_d.iloc[-1]) if not pd.isna(stoch_d.iloc[-1]) else 50.0
        sma50_val = float(sma50.iloc[-1]) if not pd.isna(sma50.iloc[-1]) else price
        sma200_val = float(sma200.iloc[-1]) if not pd.isna(sma200.iloc[-1]) else price
        atr_val = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0

        latest_vol = float(df['Volume'].iloc[-1])
        avg_vol = float(df['Volume'].rolling(20).mean().iloc[-1])

        # --- Score each indicator (contribution in range -1..+1) ---
        # Weights must sum to 1.0
        WEIGHTS = {
            "rsi":        0.20,
            "macd":       0.20,
            "bollinger":  0.15,
            "sma_cross":  0.20,
            "stochastic": 0.10,
            "volume":     0.15,
        }

        rsi_score,   rsi_signal   = score_rsi(rsi_val)
        macd_score,  macd_signal  = score_macd(macd_val, hist_val)
        bb_score,    bb_signal    = score_bollinger(pct_b_val)
        sma_score,   sma_signal   = score_sma_cross(price, sma50_val, sma200_val)
        stoch_score, stoch_signal = score_stochastic(k_val, d_val)
        vol_score,   vol_signal   = score_volume(latest_vol, avg_vol, price_change)

        # --- Weighted confluence score → normalize to 1-10 ---
        raw = (
            rsi_score   * WEIGHTS["rsi"]        +
            macd_score  * WEIGHTS["macd"]       +
            bb_score    * WEIGHTS["bollinger"]  +
            sma_score   * WEIGHTS["sma_cross"]  +
            stoch_score * WEIGHTS["stochastic"] +
            vol_score   * WEIGHTS["volume"]
        )
        # raw is in [-1, +1]; map to [1, 10]
        normalized_score = round(5.0 + raw * 4.5, 2)
        normalized_score = max(1.0, min(10.0, normalized_score))

        # --- Build key driver summary ---
        signals = [
            (abs(rsi_score),   rsi_signal),
            (abs(macd_score),  macd_signal),
            (abs(bb_score),    bb_signal),
            (abs(sma_score),   sma_signal),
            (abs(stoch_score), stoch_signal),
            (abs(vol_score),   vol_signal),
        ]
        # Top 2 strongest signals become the key driver
        top_signals = sorted(signals, key=lambda x: x[0], reverse=True)[:2]
        key_driver = " | ".join(s[1] for s in top_signals)

        # ATR-based volatility context
        atr_pct = (atr_val / price) * 100
        volatility_context = (
            "High volatility" if atr_pct > 3
            else "Moderate volatility" if atr_pct > 1.5
            else "Low volatility"
        )

        return {
            "agent_id": "beta_technical",
            "status": "success",
            "normalized_score": normalized_score,
            "raw_metrics": {
                "rsi_14":            round(rsi_val, 2),
                "macd":              round(macd_val, 4),
                "macd_histogram":    round(hist_val, 4),
                "bollinger_pct_b":   round(pct_b_val, 3),
                "stochastic_k":      round(k_val, 2),
                "stochastic_d":      round(d_val, 2),
                "sma_50":            round(sma50_val, 2),
                "sma_200":           round(sma200_val, 2),
                "atr_14":            round(atr_val, 2),
                "atr_pct":           round(atr_pct, 2),
                "volume_vs_avg":     round(latest_vol / avg_vol, 2),
                "volatility":        volatility_context,
                "key_driver":        key_driver,
            },
            "signal_breakdown": {
                "rsi":        {"score": rsi_score,   "signal": rsi_signal,   "weight": WEIGHTS["rsi"]},
                "macd":       {"score": macd_score,  "signal": macd_signal,  "weight": WEIGHTS["macd"]},
                "bollinger":  {"score": bb_score,    "signal": bb_signal,    "weight": WEIGHTS["bollinger"]},
                "sma_cross":  {"score": sma_score,   "signal": sma_signal,   "weight": WEIGHTS["sma_cross"]},
                "stochastic": {"score": stoch_score, "signal": stoch_signal, "weight": WEIGHTS["stochastic"]},
                "volume":     {"score": vol_score,   "signal": vol_signal,   "weight": WEIGHTS["volume"]},
            }
        }

    except Exception as e:
        print(f"[Agent Beta v2] Technical Analysis Error: {e}")
        return {
            "agent_id": "beta_technical",
            "status": "fallback",
            "normalized_score": 5.0,
            "raw_metrics": {
                "rsi_14": 50.0,
                "macd": 0.0,
                "macd_histogram": 0.0,
                "bollinger_pct_b": 0.5,
                "stochastic_k": 50.0,
                "stochastic_d": 50.0,
                "sma_50": 0.0,
                "sma_200": 0.0,
                "atr_14": 0.0,
                "atr_pct": 0.0,
                "volume_vs_avg": 1.0,
                "volatility": "Unknown",
                "key_driver": "Market data unavailable. Defaulted to neutral.",
            },
            "signal_breakdown": {}
        }


# --- For testing the agent in isolation ---
if __name__ == "__main__":
    import json
    print("Testing Agent Beta v2...")
    result = analyze_technicals("HDFCBANK.NS")
    print(json.dumps(result, indent=2))