import datetime as dt
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from util import get_data
from indicators import get_indicators
from indicators import cumulative_return
from indicators import avg_daily_returns
from indicators import std_daily_returns
from indicators import sharpe_ratio
from TheoreticallyOptimalStrategy import TheoreticallyOptimalStrategy
from marketsimcode import compute_portvals


def author():
    return "nwatt3"


class ManualStrategy(TheoreticallyOptimalStrategy):

    def testPolicy(self, symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2010, 12, 31), sv=100000,
                   bbr_up=1.0, bbr_low=-1.0, SMAr_up=1.05, SMAr_low=0.95):
        # Grab historical data
        prices, index_prices, trading_days = self.get_historical_data(symbol, pd.date_range(sd, ed))

        # Indicators
        indicators = get_indicators(prices.to_frame(symbol))
        divergence = indicators["divergence"]
        bbr = indicators["bbr"]
        SMAr = indicators["SMAr"]

        # Trading positions (strategy)
        df_positions = pd.Series(index=trading_days)
        for day in df_positions.index:
            previous_day = df_positions.index.get_loc(day) - 1

            ############################
            # First days of the period
            ############################
            if previous_day < 0:
                # We need at least yesterday and before yesterday for MACD
                df_positions.loc[day] = 0
                continue

            ############################
            # Rest of the period
            ############################
            elif previous_day >= 0:
                previous_day = df_positions.index[previous_day]

                if (divergence.loc[previous_day] > 0 and divergence.loc[day] < 0) or (bbr.loc[day] > bbr_up and SMAr.loc[day] > SMAr_up):
                    # Stock may be overbought, SELL signal
                    df_positions.loc[day] = -1  # SHORT
                elif (divergence.loc[previous_day] < 0 and divergence.loc[day] > 0) or (bbr.loc[day] < bbr_low and SMAr.loc[day] < SMAr_low):
                    # Stock may be oversold, BUY signal
                    df_positions.loc[day] = 1  # LONG
                else:
                    df_positions.loc[day] = 0  # DO NOTHING

            else:
                raise Exception("Error logic")

        # Positions to orders
        df_trades = self.positionsToOrders(df_positions)

        return df_trades.to_frame(symbol)


def test_code():  
    ms = ManualStrategy()

    # In sample
    symbol = "JPM"
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    period = pd.date_range(sd, ed)
    trading_days = get_data(["SPY"], dates=period).index
    sv = 100000

    # Benchmark
    benchmark_trade = pd.DataFrame([1000] + [0] * (len(trading_days)-1), columns=[symbol], index=trading_days)
    benchmark = compute_portvals(benchmark_trade, start_val=sv, commission=9.95, impact=0.005)
    benchmark /= benchmark.iloc[0,:]

    ############################
    #
    # In-sample cases
    #
    ############################

    # Test case : classic, to compare with classic thresholds
    df_trades_classic = ms.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
    portvals_classic = compute_portvals(df_trades_classic, start_val=sv, commission=9.95, impact=0.005)
    portvals_classic /= portvals_classic.iloc[0, :]

    # Test case : grid search
    """
    best_cum_ret = None
    best_SMA = None
    best_bbr = None
    for SMA in [(1.01, 0.99), (1.05, 0.95), (1.10, 0.90), (1.20, 0.80)]:
        for bbr in [(0.85, -0.85), (0.90, -0.90), (0.95, -0.95), (1., -1.), (1.05, -1.05)]:
            df_trades = ms.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv,
                                       bbr_up=bbr[0], bbr_low=bbr[1],
                                       SMAr_up=SMA[0], SMAr_low=SMA[1])
            portvals = compute_portvals(df_trades, start_val=sv, commission=9.95, impact=0.005)
            portvals /= portvals.iloc[0, :]
            gs_cum_ret = cumulative_return(portvals)
            #print "SMA: ({}, {})".format(SMA[0], SMA[1])
            #print "bbr: ({}, {})".format(bbr[0], bbr[1])
            #print "Cumulative return: {}".format(gs_cum_ret)

            if (best_cum_ret is None) or (gs_cum_ret > best_cum_ret):
                best_cum_ret = gs_cum_ret
                best_SMA = SMA
                best_bbr = bbr

    print "Best cumulative return: {}".format(best_cum_ret)
    print "Best SMA: ({}, {})".format(best_SMA[0], best_SMA[1])
    print "Best bbr: ({}, {})".format(best_bbr[0], best_bbr[1])
    df_trades_gs = ms.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv,
                              bbr_up=best_bbr[0], bbr_low=best_bbr[1],
                              SMAr_up=best_SMA[0], SMAr_low=best_SMA[1])
    portvals_gs = compute_portvals(df_trades_gs, start_val=sv, commission=9.95, impact=0.005)
    portvals_gs /= portvals_gs.iloc[0, :]
    """

    # Test case : hard coded with found
    df_trades_hard = ms.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv,
                              bbr_up=0.9, bbr_low=-0.9,
                              SMAr_up=1.01, SMAr_low=0.99)
    portvals_hard = compute_portvals(df_trades_hard, start_val=sv, commission=9.95, impact=0.005)
    portvals_hard /= portvals_hard.iloc[0,:]

    ############################
    #
    # Plotting
    #
    ############################

    fig, ax = plt.subplots()
    benchmark.plot(ax=ax, color="g")
    portvals_classic.plot(ax=ax, color="r")
    plt.legend(["Benchmark", "Manual Strategy"])
    plt.title("Manual Strategy on JPM stock over in-sample period")
    plt.xlabel("Dates")
    plt.ylabel("Normalized value")
    for day, order in df_trades_classic[df_trades_classic[symbol] != 0].iterrows():
        if order[symbol] < 0:  # SHORT
            ax.axvline(day, color="k", alpha=0.5)
        elif order[symbol] > 0:  # LONG
            ax.axvline(day, color="b", alpha=0.5)
    plt.tight_layout()
    plt.savefig("figures/manual_classic.png")
    # plt.show()

    """
    fig, ax = plt.subplots()
    benchmark.plot(ax=ax, color="g")
    portvals_gs.plot(ax=ax, color="r")
    plt.legend(["Benchmark", "Manual Strategy"])
    plt.title("Manual Strategy on JPM stock over in-sample period")
    plt.xlabel("Dates")
    plt.ylabel("Normalized value")
    for day, order in df_trades_gs[df_trades_gs[symbol] != 0].iterrows():
        if order[symbol] < 0: # SHORT
            ax.axvline(day, color="k", alpha=0.5)
        elif order[symbol] > 0: # LONG
            ax.axvline(day, color="b", alpha=0.5)
    plt.tight_layout()
    plt.savefig("figures/manual_gridsearched.png")
    #plt.show()
    """

    fig, ax = plt.subplots()
    benchmark.plot(ax=ax, color="g")
    portvals_hard.plot(ax=ax, color="r")
    plt.legend(["Benchmark", "Manual Strategy"])
    plt.title("Manual Strategy on JPM stock over in-sample period")
    plt.xlabel("Dates")
    plt.ylabel("Normalized value")
    for day, order in df_trades_hard[df_trades_hard[symbol] != 0].iterrows():
        if order[symbol] < 0:  # SHORT
            ax.axvline(day, color="k", alpha=0.5)
        elif order[symbol] > 0:  # LONG
            ax.axvline(day, color="b", alpha=0.5)
    plt.tight_layout()
    plt.savefig("figures/manual_hard.png")
    # plt.show()

    ############################
    #
    # Summary statistics
    #
    ############################

    # Final portval
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio_ = [
        cumulative_return(portvals_hard),
        avg_daily_returns(portvals_hard),
        std_daily_returns(portvals_hard),
        sharpe_ratio(portvals_hard)
    ]
    # Benchmark
    cum_ret_benchmark, avg_daily_ret_benchmark, std_daily_ret_benchmark, sharpe_ratio_benchmark = [
        cumulative_return(benchmark),
        avg_daily_returns(benchmark),
        std_daily_returns(benchmark),
        sharpe_ratio(benchmark)
    ]

    print "Date Range: {} to {}".format(sd, ed)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio_)
    print "Sharpe Ratio of Benchmark : {}".format(sharpe_ratio_benchmark)
    print
    # Cumulative return of the benchmark and portfolio
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of Benchmark : {}".format(cum_ret_benchmark)
    print
    # Stdev of daily returns of benchmark and portfolio
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of Benchmark : {}".format(std_daily_ret_benchmark)
    print
    # Mean of daily returns of benchmark and portfolio
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of Benchmark : {}".format(avg_daily_ret_benchmark)
    print
    print "Final Portfolio Value: {}".format(portvals_hard.iloc[-1, 0])

    ############################
    #
    # Out-of-sample case
    #
    ############################

    # Out of sample
    symbol = "JPM"
    sd = dt.datetime(2010, 1, 1)
    ed = dt.datetime(2011, 12, 31)
    period = pd.date_range(sd, ed)
    trading_days = get_data(["SPY"], dates=period).index
    sv = 100000

    # Benchmark
    benchmark_trade = pd.DataFrame([1000] + [0] * (len(trading_days) - 1), columns=[symbol], index=trading_days)
    benchmark = compute_portvals(benchmark_trade, start_val=sv, commission=9.95, impact=0.005)
    benchmark /= benchmark.iloc[0, :]

    # Out case : hard coded with found
    df_trades_hard = ms.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv,
                                   bbr_up=0.9, bbr_low=-0.9,
                                   SMAr_up=1.01, SMAr_low=0.99)
    portvals_hard = compute_portvals(df_trades_hard, start_val=sv, commission=9.95, impact=0.005)
    portvals_hard /= portvals_hard.iloc[0, :]

    fig, ax = plt.subplots()
    benchmark.plot(ax=ax, color="g")
    portvals_hard.plot(ax=ax, color="r")
    plt.legend(["Benchmark", "Manual Strategy"])
    plt.title("Manual Strategy on JPM stock over out-of-sample period")
    plt.xlabel("Dates")
    plt.ylabel("Normalized value")
    plt.grid()
    plt.tight_layout()
    plt.savefig("figures/comparison_1.png")
    # plt.show()

    # Final portval
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio_ = [
        cumulative_return(portvals_hard),
        avg_daily_returns(portvals_hard),
        std_daily_returns(portvals_hard),
        sharpe_ratio(portvals_hard)
    ]
    # Benchmark
    cum_ret_benchmark, avg_daily_ret_benchmark, std_daily_ret_benchmark, sharpe_ratio_benchmark = [
        cumulative_return(benchmark),
        avg_daily_returns(benchmark),
        std_daily_returns(benchmark),
        sharpe_ratio(benchmark)
    ]

    print "Date Range: {} to {}".format(sd, ed)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio_)
    print "Sharpe Ratio of Benchmark : {}".format(sharpe_ratio_benchmark)
    print
    # Cumulative return of the benchmark and portfolio
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of Benchmark : {}".format(cum_ret_benchmark)
    print
    # Stdev of daily returns of benchmark and portfolio
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of Benchmark : {}".format(std_daily_ret_benchmark)
    print
    # Mean of daily returns of benchmark and portfolio
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of Benchmark : {}".format(avg_daily_ret_benchmark)
    print
    print "Final Portfolio Value: {}".format(portvals_hard.iloc[-1, 0])


if __name__ == "__main__":
    np.random.seed(903430342)
    test_code()
