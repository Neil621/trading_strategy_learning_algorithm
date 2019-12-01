from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def author():
    return "nwatt3"


############################
#
# Statistics
#
############################

def cumulative_return(port_val):
    cum_ret = (port_val / port_val.iloc[0,0]) - 1
    return cum_ret.iloc[-1,0]


def avg_daily_returns(port_val):
    daily_ret = (port_val / port_val.shift(1)) - 1
    return daily_ret.iloc[1:,0].mean()


def std_daily_returns(port_val):
    daily_ret = (port_val / port_val.shift(1)) - 1
    return daily_ret.iloc[1:,0].std()


def sharpe_ratio(port_val, k=np.sqrt(252)):
    return k * avg_daily_returns(port_val) / std_daily_returns(port_val)


############################
#
# Indicators
#
############################

def get_sma(prices, window=10):
    return prices.rolling(window=window, min_periods=window).mean()


def get_stdev(prices, window=10):
    return prices.rolling(window=window, min_periods=window).std()


def get_daily_returns(prices):
    return prices / prices.shift(1) - 1


def get_sma_ratio(prices, window=10):
    return prices / prices.rolling(window=window, min_periods=window).mean()


def get_bb_ratio(prices, sma, stdev):
    return (prices - sma) / (2 * stdev)


def get_momentum(prices, n=10):
    return prices / prices.shift(n-1) - 1


def get_MACD(prices, n_fast=12, n_slow=26, n_signal=9):
    """Moving Average Convergence Divergence
    :param prices: Historical prices
    :param n_fast: short period
    :param n_slow: long period
    :param n_signal: signal period
    :return: MACD, MACD_signal, divergence
    .. note:: SELL signal when divergence goes from + to -, vice versa
    """
    ewma_fast = prices.ewm(span=n_fast, min_periods=n_fast).mean()
    ewma_slow = prices.ewm(span=n_slow, min_periods=n_slow).mean()
    MACD = ewma_fast - ewma_slow
    MACD_signal = MACD.ewm(span=n_signal, min_periods=n_signal).mean()
    divergence = MACD - MACD_signal
    return MACD, MACD_signal, divergence


def get_indicators(prices):
    daily_returns = get_daily_returns(prices)
    stdev = get_stdev(prices)
    sma = get_sma(prices)
    sma_ratio = get_sma_ratio(prices)
    momentum = get_momentum(prices)
    bb_ratio = get_bb_ratio(prices, sma, stdev)
    MACD, MACD_signal, divergence = get_MACD(prices)
    df = pd.concat([
        prices, daily_returns, stdev,
        momentum, sma, sma_ratio, bb_ratio,
        MACD, MACD_signal, divergence
    ], axis=1)
    df.columns = prices.columns.tolist() + [
        "daily_returns", "stdev",
        "momentum", "SMA", "SMAr", "bbr",
        "MACD", "MACD_signal", "divergence"
    ]

    return df


if __name__ == "__main__":
    pass
