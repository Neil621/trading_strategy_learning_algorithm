import datetime as dt
import matplotlib; matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pandas as pd
from util import get_data
from ManualStrategy import ManualStrategy
import StrategyLearner as sl
from marketsimcode import compute_portvals


def author():
    return "nwatt3"


def run_experiment():
    # In-sample
    symbol = "JPM"
    sv = 100000
    commission = 0.0
    impact = 0.005
    sd_in = dt.datetime(2008, 1, 1)
    ed_in = dt.datetime(2009, 12, 31)
    sd_out = dt.datetime(2010, 1, 1)
    ed_out = dt.datetime(2011, 12, 31)
    
    period = pd.date_range(sd_in, ed_in)
    trading_days = get_data(["SPY"], dates=period).index

    # Benchmark
    benchmark_trade = pd.DataFrame([1000] + [0] * (len(trading_days) - 1), columns=[symbol], index=trading_days)
    benchmark = compute_portvals(benchmark_trade, start_val=sv, commission=commission, impact=impact)
    benchmark /= benchmark.iloc[0, :]

    # Manual Strategy
    ms = ManualStrategy()
    df_trades_manual = ms.testPolicy(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv,
                                       boll_bandr_up=0.9, boll_bandr_low=-0.9,
                                       simple_moving_averager_up=1.01, simple_moving_averager_low=0.99)
    portvals_manual = compute_portvals(df_trades_manual, start_val=sv, commission=commission, impact=impact)
    portvals_manual /= portvals_manual.iloc[0, :]

    # QLearning Strategy in sample
    learner = sl.StrategyLearner(verbose=False, seed=True, impact=impact, commission=commission)
    learner.addEvidence(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv)
    df_trades = learner.testPolicy(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv)
    portvals_qlearning = compute_portvals(df_trades, start_val=sv, commission=commission, impact=impact)
    portvals_qlearning /= portvals_qlearning.iloc[0, :]

    fig, ax = plt.subplots()
    benchmark.plot(ax=ax, color="g")
    portvals_manual.plot(ax=ax, color="r")
    portvals_qlearning.plot(ax=ax, color="b")
    plt.legend(["Benchmark", "Manual Strategy", "QLearner Strategy"])
    plt.title("Strategy comparison on {} stock over in-sample period".format(symbol))
    plt.xlabel("Dates")
    plt.ylabel("Normalized value")
    plt.grid()
    plt.tight_layout()
    plt.savefig("experiment1_{}_in_sample.png".format(symbol))
    
    #out-of-sample
    periodOut = pd.date_range(sd_out, ed_out)
    trading_daysOut = get_data(["SPY"], dates=periodOut).index

    # Benchmark
    benchmark_tradeOut = pd.DataFrame([1000] + [0] * (len(trading_daysOut) - 1), columns=[symbol], index=trading_daysOut)
    benchmarkOut = compute_portvals(benchmark_tradeOut, start_val=sv, commission=commission, impact=impact)
    benchmarkOut /= benchmark.iloc[0, :]

    # Manual Strategy
    ms = ManualStrategy()
    df_trades_manualOut = ms.testPolicy(symbol=symbol, sd=sd_out, ed=ed_out, sv=sv,
                                       boll_bandr_up=0.9, boll_bandr_low=-0.9,
                                       simple_moving_averager_up=1.01, simple_moving_averager_low=0.99)
    portvals_manualOut = compute_portvals(df_trades_manualOut, start_val=sv, commission=commission, impact=impact)
    portvals_manualOut /= portvals_manual.iloc[0, :]

    # QLearning Strategy out of sample
    learner = sl.StrategyLearner(verbose=False, seed=True, impact=impact, commission=commission)
    learner.addEvidence(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv)
    df_tradesOut = learner.testPolicy(symbol=symbol, sd=sd_out, ed=ed_out, sv=sv)
    portvals_qlearningOut = compute_portvals(df_tradesOut, start_val=sv, commission=commission, impact=impact)
    portvals_qlearningOut /= portvals_qlearning.iloc[0, :]

    fig, ax = plt.subplots()
    benchmarkOut.plot(ax=ax, color="g")
    portvals_manualOut.plot(ax=ax, color="r")
    portvals_qlearningOut.plot(ax=ax, color="b")
    plt.legend(["Benchmark", "Manual Strategy", "QLearner Strategy"])
    plt.title("Strategy comparison on {} stock over Out-of-sample period".format(symbol))
    plt.xlabel("Dates")
    plt.ylabel("Normalized value")
    plt.grid()
    plt.tight_layout()
    plt.savefig("experiment1_{}_out_sample.png".format(symbol))


if __name__ == "__main__":
    run_experiment()