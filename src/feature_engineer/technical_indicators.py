"""
technical_indicators.py

This module computes technical indicators for the Korean stock market.
Indicators are grouped by category:
  - Trend: SMA, EMA, WMA, MACD, ADX, Parabolic SAR, Ichimoku, CCI
  - Momentum: RSI, Stochastic Oscillator, MFI, Williams %R
  - Volatility: Bollinger Bands, ATR, Donchian Channel, Keltner Channel, Standard Deviation
  - Volume: OBV, CMF, Accumulation/Distribution, VWAP
  - Support/Resistance Tools: Pivot Points, Fibonacci Retracement

Each function is vectorized for batch processing and designed to work with sliding-window data.
Missing values are backfilled as needed for transformer compatibility.
"""

import numpy as np
import pandas as pd


# ============================
# TREND INDICATORS
# ============================

def compute_SMA(series: pd.Series, window: int) -> pd.Series:
    """
    Calculate the Simple Moving Average (SMA).
    
    Parameters:
        series (pd.Series): Price series.
        window (int): Number of periods.
        
    Returns:
        pd.Series: SMA values.
    """
    return series.rolling(window=window, min_periods=window).mean()


def compute_EMA(series: pd.Series, window: int) -> pd.Series:
    """
    Calculate the Exponential Moving Average (EMA).
    
    Parameters:
        series (pd.Series): Price series.
        window (int): Number of periods.
        
    Returns:
        pd.Series: EMA values.
    """
    return series.ewm(span=window, adjust=False).mean()


def compute_WMA(series: pd.Series, window: int) -> pd.Series:
    """
    Calculate the Weighted Moving Average (WMA) using linearly decreasing weights.
    
    Parameters:
        series (pd.Series): Price series.
        window (int): Number of periods.
        
    Returns:
        pd.Series: WMA values.
    """
    weights = np.arange(1, window + 1)
    return series.rolling(window).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)


def compute_MACD(series: pd.Series, short_period: int = 12, long_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
    """
    Calculate the MACD indicator.
    
    Parameters:
        series (pd.Series): Price series.
        short_period (int): Short-term EMA period.
        long_period (int): Long-term EMA period.
        signal_period (int): Signal line EMA period.
        
    Returns:
        pd.DataFrame: MACD, Signal, and Histogram columns.
    """
    ema_short = compute_EMA(series, short_period)
    ema_long = compute_EMA(series, long_period)
    macd_line = ema_short - ema_long
    signal_line = compute_EMA(macd_line, signal_period)
    histogram = macd_line - signal_line
    return pd.DataFrame({
        'MACD': macd_line,
        'Signal': signal_line,
        'Histogram': histogram
    })


def compute_ADX(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate the Average Directional Index (ADX).
    
    Parameters:
        high (pd.Series): High price series.
        low (pd.Series): Low price series.
        close (pd.Series): Close price series.
        window (int): Lookback period.
        
    Returns:
        pd.Series: ADX values.
    """
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    up_move = high - high.shift()
    down_move = low.shift() - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    tr_smooth = tr.rolling(window).sum()
    plus_dm_smooth = pd.Series(plus_dm).rolling(window).sum()
    minus_dm_smooth = pd.Series(minus_dm).rolling(window).sum()

    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)) * 100
    adx = pd.Series(dx).rolling(window).mean()
    return adx


def compute_parabolic_SAR(high: pd.Series, low: pd.Series, acceleration: float = 0.02, max_acceleration: float = 0.2) -> pd.Series:
    """
    Calculate the Parabolic SAR.
    
    Parameters:
        high (pd.Series): High price series.
        low (pd.Series): Low price series.
        acceleration (float): Acceleration factor.
        max_acceleration (float): Maximum acceleration factor.
        
    Returns:
        pd.Series: Parabolic SAR values.
    """
    sar = [low.iloc[0]]
    trend = 1  # 1 for uptrend, -1 for downtrend
    af = acceleration
    ep = high.iloc[0]
    for i in range(1, len(high)):
        prev_sar = sar[-1]
        if trend == 1:
            sar_new = prev_sar + af * (ep - prev_sar)
            sar_new = min(sar_new, low.iloc[i-1], low.iloc[i])
            if high.iloc[i] > ep:
                ep = high.iloc[i]
                af = min(af + acceleration, max_acceleration)
            if low.iloc[i] < sar_new:
                trend = -1
                sar_new = ep
                ep = low.iloc[i]
                af = acceleration
        else:
            sar_new = prev_sar + af * (ep - prev_sar)
            sar_new = max(sar_new, high.iloc[i-1], high.iloc[i])
            if low.iloc[i] < ep:
                ep = low.iloc[i]
                af = min(af + acceleration, max_acceleration)
            if high.iloc[i] > sar_new:
                trend = 1
                sar_new = ep
                ep = high.iloc[i]
                af = acceleration
        sar.append(sar_new)
    return pd.Series(sar, index=high.index)


def compute_ichimoku(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
    """
    Calculate Ichimoku Cloud components.
    
    Parameters:
        high (pd.Series): High price series.
        low (pd.Series): Low price series.
        close (pd.Series): Close price series.
        
    Returns:
        pd.DataFrame: Columns for Tenkan-sen, Kijun-sen, Senkou Span A, Senkou Span B, Chikou Span.
    """
    period_tenkan = 9
    period_kijun = 26
    period_senkou_b = 52
    
    tenkan_sen = (high.rolling(window=period_tenkan).max() + low.rolling(window=period_tenkan).min()) / 2
    kijun_sen = (high.rolling(window=period_kijun).max() + low.rolling(window=period_kijun).min()) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(period_kijun)
    senkou_span_b = ((high.rolling(window=period_senkou_b).max() + low.rolling(window=period_senkou_b).min()) / 2).shift(period_kijun)
    chikou_span = close.shift(-period_kijun)
    
    return pd.DataFrame({
        'Tenkan_sen': tenkan_sen,
        'Kijun_sen': kijun_sen,
        'Senkou_span_A': senkou_span_a,
        'Senkou_span_B': senkou_span_b,
        'Chikou_span': chikou_span
    })


def compute_CCI(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate the Commodity Channel Index (CCI).
    
    Parameters:
        high (pd.Series): High price series.
        low (pd.Series): Low price series.
        close (pd.Series): Close price series.
        window (int): Lookback period.
        
    Returns:
        pd.Series: CCI values.
    """
    typical_price = (high + low + close) / 3
    sma = typical_price.rolling(window=window).mean()
    mad = typical_price.rolling(window=window).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (typical_price - sma) / (0.015 * mad + 1e-10)
    return cci


# ============================
# MOMENTUM INDICATORS
# ============================

def compute_RSI(close: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI).
    
    Parameters:
        close (pd.Series): Close price series.
        window (int): Lookback period.
        
    Returns:
        pd.Series: RSI values.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14, smoothK: int = 3) -> pd.DataFrame:
    """
    Calculate the Stochastic Oscillator (%K and %D).
    
    Parameters:
        high (pd.Series): High price series.
        low (pd.Series): Low price series.
        close (pd.Series): Close price series.
        window (int): Lookback period for %K.
        smoothK (int): Smoothing period for %D.
        
    Returns:
        pd.DataFrame: Columns '%K' and '%D'.
    """
    lowest_low = low.rolling(window=window, min_periods=window).min()
    highest_high = high.rolling(window=window, min_periods=window).max()
    percent_k = ((close - lowest_low) / (highest_high - lowest_low)) * 100
    percent_d = percent_k.rolling(window=smoothK, min_periods=smoothK).mean()
    return pd.DataFrame({'%K': percent_k, '%D': percent_d})


def compute_MFI(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate the Money Flow Index (MFI).
    
    Parameters:
        high (pd.Series): High price series.
        low (pd.Series): Low price series.
        close (pd.Series): Close price series.
        volume (pd.Series): Volume series.
        window (int): Lookback period.
        
    Returns:
        pd.Series: MFI values.
    """
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    pos_flow = np.where(typical_price > typical_price.shift(), money_flow, 0)
    neg_flow = np.where(typical_price < typical_price.shift(), money_flow, 0)
    pos_mf = pd.Series(pos_flow).rolling(window=window, min_periods=window).sum()
    neg_mf = pd.Series(neg_flow).rolling(window=window, min_periods=window).sum()
    mfi = 100 - (100 / (1 + pos_mf / (neg_mf + 1e-10)))
    return pd.Series(mfi, index=close.index)


def compute_williams_R(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate Williams %R.
    
    Parameters:
        high (pd.Series): High price series.
        low (pd.Series): Low price series.
        close (pd.Series): Close price series.
        window (int): Lookback period.
        
    Returns:
        pd.Series: Williams %R values.
    """
    highest_high = high.rolling(window=window, min_periods=window).max()
    lowest_low = low.rolling(window=window, min_periods=window).min()
    percent_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    return percent_r


# ============================
# VOLATILITY INDICATORS
# ============================

def compute_bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2) -> pd.DataFrame:
    """
    Calculate Bollinger Bands.
    
    Parameters:
        series (pd.Series): Price series.
        window (int): Lookback period.
        num_std (float): Number of standard deviations.
        
    Returns:
        pd.DataFrame: Columns for middle, upper, and lower bands.
    """
    middle_band = series.rolling(window=window).mean()
    std_dev = series.rolling(window=window).std()
    upper_band = middle_band + num_std * std_dev
    lower_band = middle_band - num_std * std_dev
    return pd.DataFrame({'middle_band': middle_band, 'upper_band': upper_band, 'lower_band': lower_band})


def compute_ATR(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate the Average True Range (ATR).
    
    Parameters:
        high (pd.Series): High price series.
        low (pd.Series): Low price series.
        close (pd.Series): Close price series.
        window (int): Lookback period.
        
    Returns:
        pd.Series: ATR values.
    """
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=window, min_periods=window).mean()
    return atr


def compute_donchian_channel(high: pd.Series, low: pd.Series, window: int = 20) -> pd.DataFrame:
    """
    Calculate the Donchian Channel.
    
    Parameters:
        high (pd.Series): High price series.
        low (pd.Series): Low price series.
        window (int): Lookback period.
        
    Returns:
        pd.DataFrame: Columns for upper band, lower band, and middle line.
    """
    upper_band = high.rolling(window=window, min_periods=window).max()
    lower_band = low.rolling(window=window, min_periods=window).min()
    middle_line = (upper_band + lower_band) / 2
    return pd.DataFrame({'upper_band': upper_band, 'lower_band': lower_band, 'middle_line': middle_line})


def compute_keltner_channel(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20, multiplier: float = 1.5) -> pd.DataFrame:
    """
    Calculate the Keltner Channel.
    
    Parameters:
        high (pd.Series): High price series.
        low (pd.Series): Low price series.
        close (pd.Series): Close price series.
        window (int): Lookback period for EMA.
        multiplier (float): Multiplier for ATR.
        
    Returns:
        pd.DataFrame: Columns for middle line, upper band, and lower band.
    """
    middle_line = close.ewm(span=window, adjust=False).mean()
    atr = compute_ATR(high, low, close, window=window)
    upper_band = middle_line + multiplier * atr
    lower_band = middle_line - multiplier * atr
    return pd.DataFrame({'middle_line': middle_line, 'upper_band': upper_band, 'lower_band': lower_band})


def compute_std_dev(series: pd.Series, window: int) -> pd.Series:
    """
    Calculate rolling standard deviation.
    
    Parameters:
        series (pd.Series): Price series.
        window (int): Lookback period.
        
    Returns:
        pd.Series: Standard deviation values.
    """
    return series.rolling(window=window).std()


# ============================
# VOLUME INDICATORS
# ============================

def compute_OBV(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate On-Balance Volume (OBV).
    
    Parameters:
        close (pd.Series): Close price series.
        volume (pd.Series): Volume series.
        
    Returns:
        pd.Series: OBV values.
    """
    obv = [0]
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.append(obv[-1] + volume.iloc[i])
        elif close.iloc[i] < close.iloc[i-1]:
            obv.append(obv[-1] - volume.iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=close.index)


def compute_CMF(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate Chaikin Money Flow (CMF).
    
    Parameters:
        high (pd.Series): High price series.
        low (pd.Series): Low price series.
        close (pd.Series): Close price series.
        volume (pd.Series): Volume series.
        window (int): Lookback period.
        
    Returns:
        pd.Series: CMF values.
    """
    mfm = ((close - low) - (high - close)) / (high - low + 1e-10)
    mfv = mfm * volume
    cmf = mfv.rolling(window=window).sum() / volume.rolling(window=window).sum()
    return cmf


def compute_accumulation_distribution(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate the Accumulation/Distribution (A/D) line.
    
    Parameters:
        high (pd.Series): High price series.
        low (pd.Series): Low price series.
        close (pd.Series): Close price series.
        volume (pd.Series): Volume series.
        
    Returns:
        pd.Series: A/D values.
    """
    clv = ((close - low) - (high - close)) / (high - low + 1e-10)
    ad = (clv * volume).cumsum()
    return ad


def compute_VWAP(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate the Volume-Weighted Average Price (VWAP).
    
    Parameters:
        close (pd.Series): Close price series.
        volume (pd.Series): Volume series.
        
    Returns:
        pd.Series: VWAP values.
    """
    return (close * volume).cumsum() / volume.cumsum()


# ============================
# SUPPORT & RESISTANCE TOOLS
# ============================

def compute_pivot_points(prev_high: float, prev_low: float, prev_close: float) -> dict:
    """
    Calculate Pivot Points and derived support/resistance levels.
    
    Parameters:
        prev_high (float): Previous period high.
        prev_low (float): Previous period low.
        prev_close (float): Previous period close.
        
    Returns:
        dict: Dictionary containing PP, R1, S1, R2, and S2.
    """
    PP = (prev_high + prev_low + prev_close) / 3
    R1 = 2 * PP - prev_low
    S1 = 2 * PP - prev_high
    R2 = PP + (prev_high - prev_low)
    S2 = PP - (prev_high - prev_low)
    return {'PP': PP, 'R1': R1, 'S1': S1, 'R2': R2, 'S2': S2}


def compute_fibonacci_retracement(high: float, low: float, ratios: list = [0.236, 0.382, 0.5, 0.618, 0.786]) -> dict:
    """
    Calculate Fibonacci retracement levels.
    
    Parameters:
        high (float): The high price of the trend.
        low (float): The low price of the trend.
        ratios (list): Fibonacci ratios.
        
    Returns:
        dict: Dictionary of retracement levels keyed by ratio (e.g., "23.6%").
    """
    diff = high - low
    levels = {}
    for ratio in ratios:
        level = high - diff * ratio
        levels[f"{ratio * 100:.1f}%"] = level
    return levels


# ============================
# Helper: Compute All Indicators
# ============================

def compute_all_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Compute technical indicators based on configuration.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing columns like 'close', 'high', 'low', 'volume'.
        config (dict): Configuration dictionary (e.g., loaded from config/indicators.yaml)
        
    Returns:
        pd.DataFrame: DataFrame with additional indicator columns.
    """
    # Create a copy so we don't modify the original DataFrame
    df = df.copy()
    
    # --- Trend Indicators ---
    if config.get("trend", {}).get("compute_SMA", True):
        window = config["trend"].get("SMA_window", 20)
        df[f"SMA_{window}"] = compute_SMA(df["close"], window)
    if config.get("trend", {}).get("compute_EMA", True):
        window = config["trend"].get("EMA_window", 20)
        df[f"EMA_{window}"] = compute_EMA(df["close"], window)
    if config.get("trend", {}).get("compute_MACD", True):
        macd_df = compute_MACD(df["close"])
        df = pd.concat([df, macd_df.add_prefix("MACD_")], axis=1)
    if config.get("trend", {}).get("compute_ADX", True):
        window = config["trend"].get("ADX_window", 14)
        df["ADX"] = compute_ADX(df["high"], df["low"], df["close"], window)
    if config.get("trend", {}).get("compute_parabolic_SAR", True):
        df["Parabolic_SAR"] = compute_parabolic_SAR(df["high"], df["low"])
    if config.get("trend", {}).get("compute_ichimoku", True):
        ichimoku_df = compute_ichimoku(df["high"], df["low"], df["close"])
        df = pd.concat([df, ichimoku_df.add_prefix("Ichimoku_")], axis=1)
    if config.get("trend", {}).get("compute_CCI", True):
        window = config["trend"].get("CCI_window", 20)
        df["CCI"] = compute_CCI(df["high"], df["low"], df["close"], window)
    
    # --- Momentum Indicators ---
    if config.get("momentum", {}).get("compute_RSI", True):
        window = config["momentum"].get("RSI_window", 14)
        df["RSI"] = compute_RSI(df["close"], window)
    if config.get("momentum", {}).get("compute_stochastic", True):
        df_stoch = compute_stochastic_oscillator(df["high"], df["low"], df["close"])
        df = pd.concat([df, df_stoch.add_prefix("Stoch_")], axis=1)
    if config.get("momentum", {}).get("compute_MFI", True):
        window = config["momentum"].get("MFI_window", 14)
        df["MFI"] = compute_MFI(df["high"], df["low"], df["close"], df["volume"], window)
    if config.get("momentum", {}).get("compute_williams_R", True):
        window = config["momentum"].get("WilliamsR_window", 14)
        df["Williams_R"] = compute_williams_R(df["high"], df["low"], df["close"], window)
    
    # --- Volatility Indicators ---
    if config.get("volatility", {}).get("compute_bollinger", True):
        window = config["volatility"].get("Bollinger_window", 20)
        num_std = config["volatility"].get("Bollinger_std", 2)
        bb_df = compute_bollinger_bands(df["close"], window, num_std)
        df = pd.concat([df, bb_df.add_prefix("BB_")], axis=1)
    if config.get("volatility", {}).get("compute_ATR", True):
        window = config["volatility"].get("ATR_window", 14)
        df["ATR"] = compute_ATR(df["high"], df["low"], df["close"], window)
    if config.get("volatility", {}).get("compute_donchian", True):
        window = config["volatility"].get("Donchian_window", 20)
        donchian_df = compute_donchian_channel(df["high"], df["low"], window)
        df = pd.concat([df, donchian_df.add_prefix("Donchian_")], axis=1)
    if config.get("volatility", {}).get("compute_keltner", True):
        window = config["volatility"].get("Keltner_window", 20)
        multiplier = config["volatility"].get("Keltner_multiplier", 1.5)
        keltner_df = compute_keltner_channel(df["high"], df["low"], df["close"], window, multiplier)
        df = pd.concat([df, keltner_df.add_prefix("Keltner_")], axis=1)
    if config.get("volatility", {}).get("compute_std_dev", True):
        window = config["volatility"].get("StdDev_window", 20)
        df[f"StdDev_{window}"] = compute_std_dev(df["close"], window)
    
    # --- Volume Indicators ---
    if config.get("volume", {}).get("compute_OBV", True):
        df["OBV"] = compute_OBV(df["close"], df["volume"])
    if config.get("volume", {}).get("compute_CMF", True):
        window = config["volume"].get("CMF_window", 20)
        df["CMF"] = compute_CMF(df["high"], df["low"], df["close"], df["volume"], window)
    if config.get("volume", {}).get("compute_A/D", True):
        df["A/D"] = compute_accumulation_distribution(df["high"], df["low"], df["close"], df["volume"])
    if config.get("volume", {}).get("compute_VWAP", True):
        df["VWAP"] = compute_VWAP(df["close"], df["volume"])
    
    # --- Support/Resistance Tools ---
    if config.get("support_resistance", {}).get("compute_pivot", True):
        # Assuming last row of previous period data is available as scalars (this may be computed separately)
        last_row = df.iloc[-1]
        pivots = compute_pivot_points(last_row["high"], last_row["low"], last_row["close"])
        for key, value in pivots.items():
            df[f"Pivot_{key}"] = value  # Constant across the dataset (or use a rolling window approach)
    if config.get("support_resistance", {}).get("compute_fibonacci", True):
        # Similarly, using last row high/low to compute retracement levels
        last_row = df.iloc[-1]
        fib_levels = compute_fibonacci_retracement(last_row["high"], last_row["low"])
        for key, value in fib_levels.items():
            df[f"Fib_{key}"] = value

    # Backfill any missing values for compatibility with sliding-window processing
    df.fillna(method='bfill', inplace=True)
    
    # Optionally, you can apply scaling here (or in a later pipeline step)
    return df


# ============================
# Example Testing Block
# ============================
if __name__ == "__main__":
    # For testing, load a sample cleaned dataset.
    # In practice, this will be your 'korean_stock_data_cleaned.csv'
    sample_data = {
        'date': pd.date_range(start='2022-01-01', periods=50, freq='D'),
        'close': np.random.uniform(100, 200, 50),
        'high': np.random.uniform(100, 200, 50),
        'low': np.random.uniform(100, 200, 50),
        'volume': np.random.uniform(1000, 5000, 50)
    }
    df_sample = pd.DataFrame(sample_data)
    
    # Example configuration dictionary (this would be loaded from config/indicators.yaml)
    config = {
        "trend": {
            "compute_SMA": True,
            "SMA_window": 20,
            "compute_EMA": True,
            "EMA_window": 20,
            "compute_MACD": True,
            "compute_ADX": True,
            "ADX_window": 14,
            "compute_parabolic_SAR": True,
            "compute_ichimoku": True,
            "compute_CCI": True,
            "CCI_window": 20
        },
        "momentum": {
            "compute_RSI": True,
            "RSI_window": 14,
            "compute_stochastic": True,
            "compute_MFI": True,
            "MFI_window": 14,
            "compute_williams_R": True,
            "WilliamsR_window": 14
        },
        "volatility": {
            "compute_bollinger": True,
            "Bollinger_window": 20,
            "Bollinger_std": 2,
            "compute_ATR": True,
            "ATR_window": 14,
            "compute_donchian": True,
            "Donchian_window": 20,
            "compute_keltner": True,
            "Keltner_window": 20,
            "Keltner_multiplier": 1.5,
            "compute_std_dev": True,
            "StdDev_window": 20
        },
        "volume": {
            "compute_OBV": True,
            "compute_CMF": True,
            "CMF_window": 20,
            "compute_A/D": True,
            "compute_VWAP": True
        },
        "support_resistance": {
            "compute_pivot": True,
            "compute_fibonacci": True
        }
    }
    
    df_with_indicators = compute_all_indicators(df_sample, config)
    print(df_with_indicators.head())
