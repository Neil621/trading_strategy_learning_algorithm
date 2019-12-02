


import datetime as dt
import matplotlib; matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from util import get_data
from ManualStrategy import ManualStrategy
import StrategyLearner as sl
from marketsimcode import compute_portvals

from indicators import cum_return
from indicators import sharpe_ratio


def author():
    return "nwatt3"


def test_code():
    # In-sample
    symbol = "JPM"
    sv = 100000
    commission = 0.0
    sd_in = dt.datetime(2008, 1, 1)
    ed_in = dt.datetime(2009, 12, 31)
    period = pd.date_range(sd_in, ed_in)
    trading_days = get_data(["SPY"], dates=period).index

    ############################
    #
    # Cumulative return
    #
    ############################

    df_cum_ret = pd.DataFrame(
        columns=["Benchmark", "Manual Strategy", "QLearning Strategy"],
        index=np.linspace(0.0, 0.01, num=10)
    )
    for impact, _ in df_cum_ret.iterrows():
        print("Compare cumulative return against impact={}".format(impact))

        # Benchmark
        benchmark_trade = pd.DataFrame([1000] + [0] * (len(trading_days) - 1), columns=[symbol], index=trading_days)
        benchmark = compute_portvals(benchmark_trade, start_val=sv, commission=commission, impact=impact)
        benchmark /= benchmark.iloc[0, :]
        df_cum_ret.loc[impact, "Benchmark"] = cum_return(benchmark)

        # Manual Strategy
        ms = ManualStrategy()
        #df_trades_manual = ms.testPolicy(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv,
         #                                bbr_up=0.9, bbr_low=-0.9,
          #                               SMAr_up=1.01, SMAr_low=0.99)
        
        df_trades_manual = ms.testPolicy(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv,
                                         boll_bandr_up=0.8, boll_bandr_low=0.2,
                                         simple_moving_averager_up=1.05, simple_moving_averager_low=0.95)
        
        
        
        portvals_manual = compute_portvals(df_trades_manual, start_val=sv, commission=commission, impact=impact)
        portvals_manual /= portvals_manual.iloc[0, :]
        df_cum_ret.loc[impact, "Manual Strategy"] = cum_return(portvals_manual)

        # QLearning Strategy
        learner = sl.StrategyLearner(verbose=False, seed=True, impact=impact, commission=commission)
        learner.addEvidence(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv)
        df_trades_qlearning = learner.testPolicy(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv)
        portvals_qlearning = compute_portvals(df_trades_qlearning, start_val=sv, commission=commission, impact=impact)
        portvals_qlearning /= portvals_qlearning.iloc[0, :]
        df_cum_ret.loc[impact, "QLearning Strategy"] = cum_return(portvals_qlearning)

    fig, ax = plt.subplots()
    df_cum_ret[["Benchmark"]].plot(ax=ax, color="g", marker="o")
    df_cum_ret[["Manual Strategy"]].plot(ax=ax, color="r", marker="o")
    df_cum_ret[["QLearning Strategy"]].plot(ax=ax, color="b", marker="o")
    plt.title("Cumulative return against impact on {} stock over in-sample period".format(symbol))
    plt.xlabel("Impact")
    plt.ylabel("Cumulative return")
    plt.grid()
    plt.tight_layout()
    plt.savefig("experiment2_{}_cr_in_sample.png".format(symbol))

    ############################
    #
    # Sharpe ratio
    #
    ############################

    df_sharpe = pd.DataFrame(
        columns=["Benchmark", "Manual Strategy", "QLearning Strategy"],
        index=np.linspace(0.0, 0.01, num=10)
    )
    for impact, _ in df_sharpe.iterrows():
        print("Compare Sharpe ratio against impact={}".format(impact))

        # Benchmark
        benchmark_trade = pd.DataFrame([1000] + [0] * (len(trading_days) - 1), columns=[symbol], index=trading_days)
        benchmark = compute_portvals(benchmark_trade, start_val=sv, commission=commission, impact=impact)
        benchmark /= benchmark.iloc[0, :]
        df_sharpe.loc[impact, "Benchmark"] = sharpe_ratio(benchmark)

        # Manual Strategy
        ms = ManualStrategy()
        #df_trades_manual = ms.testPolicy(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv,
         #                                bbr_up=0.9, bbr_low=-0.9,
          #                               SMAr_up=1.01, SMAr_low=0.99)
        
        df_trades_manual = ms.testPolicy(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv,
                                         boll_bandr_up=0.8, boll_bandr_low=0.2,
                                         simple_moving_averager_up=1.05, simple_moving_averager_low=0.95)
        
        
        portvals_manual = compute_portvals(df_trades_manual, start_val=sv, commission=commission, impact=impact)
        portvals_manual /= portvals_manual.iloc[0, :]
        df_sharpe.loc[impact, "Manual Strategy"] = sharpe_ratio(portvals_manual)

        # QLearning Strategy
        learner = sl.StrategyLearner(verbose=False, seed=True, impact=impact, commission=commission)
        learner.addEvidence(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv)
        df_trades_qlearning = learner.testPolicy(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv)
        portvals_qlearning = compute_portvals(df_trades_qlearning, start_val=sv, commission=commission, impact=impact)
        portvals_qlearning /= portvals_qlearning.iloc[0, :]
        df_sharpe.loc[impact, "QLearning Strategy"] = sharpe_ratio(portvals_qlearning)

    fig, ax = plt.subplots()
    df_sharpe[["Benchmark"]].plot(ax=ax, color="g", marker="o")
    df_sharpe[["Manual Strategy"]].plot(ax=ax, color="r", marker="o")
    df_sharpe[["QLearning Strategy"]].plot(ax=ax, color="b", marker="o")
    plt.title("Sharpe ratio against impact on {} stock over in-sample period".format(symbol))
    plt.xlabel("Impact")
    plt.ylabel("Sharpe ratio")
    plt.grid()
    plt.tight_layout()
    plt.savefig("experiment2_{}_sr_in_sample.png".format(symbol))

    ############################
    #
    # Number of orders
    #
    ############################

    nb_orders = pd.DataFrame(
        columns=["Benchmark", "Manual Strategy", "QLearning Strategy"],
        index=np.linspace(0.0, 0.01, num=5).tolist() + [0.2, 0.35, 0.5, 0.65, 0.8]
    )
    for impact, _ in nb_orders.iterrows():
        print("Compare number of orders against impact={}".format(impact))

        # Benchmark
        benchmark_trade = pd.DataFrame([1000] + [0] * (len(trading_days) - 1), columns=[symbol], index=trading_days)
        nb_orders.loc[impact, "Benchmark"] = (np.abs(benchmark_trade[symbol]) > 0).sum()

        # Manual Strategy
        ms = ManualStrategy()
        #df_trades_manual = ms.testPolicy(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv,
         #                                bbr_up=0.9, bbr_low=-0.9,
          #                               SMAr_up=1.01, SMAr_low=0.99)
        
        
        df_trades_manual = ms.testPolicy(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv,
                                         boll_bandr_up=0.8, boll_bandr_low=0.2,
                                         simple_moving_averager_up=1.05, simple_moving_averager_low=0.95)
        
        
        
         
        
        
        
        nb_orders.loc[impact, "Manual Strategy"] = (np.abs(df_trades_manual[symbol]) > 0).sum()

        # QLearning Strategy
        learner = sl.StrategyLearner(verbose=False, seed=True, impact=impact, commission=commission)
        learner.addEvidence(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv)
        df_trades_qlearning = learner.testPolicy(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv)
        nb_orders.loc[impact, "QLearning Strategy"] = (np.abs(df_trades_qlearning[symbol]) > 0).sum()

    fig, ax = plt.subplots()
    nb_orders[["Benchmark"]].plot(ax=ax, color="g", marker="o")
    nb_orders[["Manual Strategy"]].plot(ax=ax, color="r", marker="o")
    nb_orders[["QLearning Strategy"]].plot(ax=ax, color="b", marker="o")
    plt.title("Number of orders against impact on {} stock over in-sample period".format(symbol))
    plt.xlabel("Impact")
    plt.ylabel("Number of orders")
    plt.grid()
    plt.tight_layout()
    plt.savefig("experiment2_{}_norders_in_sample.png".format(symbol))


if __name__ == "__main__":
    test_code()
