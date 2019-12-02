
from indicators import get_indicators
from indicators import cum_return
from indicators import average_daily_ret
from indicators import standard_dev_daily_ret
import datetime as dt
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from util import get_data

from indicators import sharpe_ratio
from TheoreticallyOptimalStrategy import theoreticallly_optimal_strategy
from marketsimcode import compute_portvals

np.random.seed(5875)


def author():
    return "nwatt3"


class ManualStrategy(theoreticallly_optimal_strategy):


    
    
    def testPolicy(self, symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2010, 12, 31), sv=100000,
                   boll_bandr_up=0.8, boll_bandr_low=0.2, simple_moving_averager_up=1.05, simple_moving_averager_low=0.95):
        # Grab historical data
        prices, index_prices, trading_days = self.retrieve_price_data(symbol, pd.date_range(sd, ed))

        # Indicators
        indicators = get_indicators(prices.to_frame(symbol))
        #divergence = indicators["divergence"]
        boll_bandr = indicators["boll_bandr"]
        simple_moving_averager = indicators["simple_moving_averager"]
        stdev_divergence=indicators["stdev_divergence"]
        

        # Trading positions (strategy)
        df_positions = pd.Series(index=trading_days)
        for day in df_positions.index:
            previous_day = df_positions.index.get_loc(day) - 1

           
            if previous_day < 0:
                
                df_positions.loc[day] = 0
                continue

           
            elif previous_day >= 0:
                previous_day = df_positions.index[previous_day]

                #volatility over threshold do NOTHING as indicators are meaningless
                
                if stdev_divergence.loc[day]>0.05:
                
                    df_positions.loc[day]= 0 # do nothing at too volatile!
                
                else:
                
                    if (boll_bandr.loc[day] > boll_bandr_up and simple_moving_averager.loc[day] > simple_moving_averager_up):
                    # two sell signals = > sell
                       df_positions.loc[day] = -1  
                    elif  (boll_bandr.loc[day] < boll_bandr_low and simple_moving_averager.loc[day] < simple_moving_averager_low):
                    # Stock may be oversold, BUY signal
                       df_positions.loc[day] = 1  # LONG
                    else:
                        df_positions.loc[day] = 0  # DO NOTHING

            else:
                raise Exception("Error logic")

        # Positions to orders
        df_trades = self.generate_orders(df_positions)

        return df_trades.to_frame(symbol)


def test_code():
    manual_strategy = ManualStrategy()

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

   
    
    df_trades_tweaked = manual_strategy.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
    portvals_tweaked = compute_portvals(df_trades_tweaked, start_val=sv, commission=9.95, impact=0.005)
    portvals_tweaked /= portvals_tweaked.iloc[0, :]


   

    fig, ax = plt.subplots()
    benchmark.plot(ax=ax, color="g")
    portvals_tweaked.plot(ax=ax, color="r")
    plt.legend(["Benchmark", "rule-based portfolio"])
    plt.title("Rule-based manual strategy applied to JPM stock (in-sample period)")
    plt.xlabel("Dates")
    plt.ylabel("Normalized value")
    for day, order in df_trades_tweaked[df_trades_tweaked[symbol] != 0].iterrows():
        if order[symbol] < 0:  # sell, short entry points
            ax.axvline(day, color="black", alpha=0.5)
        elif order[symbol] > 0:  # buy, long entry points
            ax.axvline(day, color="blue", alpha=0.5)
    plt.tight_layout()
    
    plt.savefig("manual_insample.png")
   

 
    

    
    
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio_ = [
        cum_return(portvals_tweaked),
        average_daily_ret(portvals_tweaked),
        standard_dev_daily_ret(portvals_tweaked),
        sharpe_ratio(portvals_tweaked)
    ]
    # Benchmark
    cum_ret_benchmark, avg_daily_ret_benchmark, std_daily_ret_benchmark, sharpe_ratio_benchmark = [
        cum_return(benchmark),
        average_daily_ret(benchmark),
        standard_dev_daily_ret(benchmark),
        sharpe_ratio(benchmark)
    ]

    print('Date Range: {} to {}'.format(sd, ed))
    print
    
    
   
    # Cumulative return of the benchmark and portfolio
    print("In Sample Cumulative Return of Strategy: {}".format(cum_ret))
    print("In Sample Cumulative Return of Benchmark : {}".format(cum_ret_benchmark))
    print
    # Stdev of daily returns of benchmark and portfolio
    print("In Sample Stdev of Strategy: {}".format(std_daily_ret))
    print("In Sample Stdev of Benchmark : {}".format(std_daily_ret_benchmark))
    print
    print("Sharpe Ratio of Strategy: {}".format(sharpe_ratio_))
    print("Sharpe Ratio of Benchmark : {}".format(sharpe_ratio_benchmark))
    print
    # Mean of daily returns of benchmark and portfolio
    print("In Sample Mean of daily returns  of Strategy: {}".format(avg_daily_ret))
    print("In Sample Mean of daily returns  of Benchmark : {}".format(avg_daily_ret_benchmark))
   
   
    # Out of sample
    symbol = "JPM"
    sd = dt.datetime(2010, 1, 1)
    ed = dt.datetime(2011, 12, 31)
    period = pd.date_range(sd, ed)
    trading_days = get_data(["SPY"], dates=period).index
    sv = 100000

   
    
    df_trades_tweaked = manual_strategy.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
    portvals_tweaked = compute_portvals(df_trades_tweaked, start_val=sv, commission=9.95, impact=0.005)
    portvals_tweaked /= portvals_tweaked.iloc[0, :]
    
     # Benchmark
    benchmark_trade = pd.DataFrame([1000] + [0] * (len(trading_days) - 1), columns=[symbol], index=trading_days)
    benchmark = compute_portvals(benchmark_trade, start_val=sv, commission=9.95, impact=0.005)
    benchmark /= benchmark.iloc[0, :]



    fig, ax = plt.subplots()
    benchmark.plot(ax=ax, color="g")
    portvals_tweaked.plot(ax=ax, color="r")
    plt.legend(["Benchmark", "rule-based portfolio"])
    plt.title("Rule-based manual strategy applied to JPM stock (out-of-sample period)")
    plt.xlabel("Dates")
    plt.ylabel("Normalized value")
    plt.grid()
    plt.tight_layout()
    
    plt.savefig("manual_out_of_sample_comparison.png")
    
    
    
    fig, ax = plt.subplots()
    benchmark.plot(ax=ax, color="g")
    portvals_tweaked.plot(ax=ax, color="r")
    plt.legend(["Benchmark", "rule-based portfolio"])
    plt.title("Rule-based manual strategy applied to JPM stock (in-sample period) trades")
    plt.xlabel("Dates")
    plt.ylabel("Normalized value")
    for day, order in df_trades_tweaked[df_trades_tweaked[symbol] != 0].iterrows():
        if order[symbol] < 0:  # sell, short entry points
            ax.axvline(day, color="black", alpha=0.5)
        elif order[symbol] > 0:  # buy, long entry points
            ax.axvline(day, color="blue", alpha=0.5)
    plt.tight_layout()
    
    plt.savefig("manual_out_sample_buysell.png")
    
    

    # Final portval
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio_ = [
        cum_return(portvals_tweaked),
        average_daily_ret(portvals_tweaked),
        standard_dev_daily_ret(portvals_tweaked),
        sharpe_ratio(portvals_tweaked)
    ]
    # Benchmark
    cum_ret_benchmark, avg_daily_ret_benchmark, std_daily_ret_benchmark, sharpe_ratio_benchmark = [
        cum_return(benchmark),
        average_daily_ret(benchmark),
        standard_dev_daily_ret(benchmark),
        sharpe_ratio(benchmark)
    ]

    print("Date Range: {} to {}".format(sd, ed))
    print
    
    print
    # Cumulative return of the benchmark and portfolio
    print("Out of Sample Cumulative Return of Strategy: {}".format(cum_ret))
    print("Out of Sample Cumulative Return of Benchmark : {}".format(cum_ret_benchmark))
    print
    # Stdev of daily returns of benchmark and portfolio
    print("Out of Sample Stdev  of Strategy: {}".format(std_daily_ret))
    print("Out of Sample Stdev  of Benchmark : {}".format(std_daily_ret_benchmark))
    print
    # Mean of daily returns of benchmark and portfolio
    print("Out of Sample Mean of daily returns of Strategy: {}".format(avg_daily_ret))
    print("Out of Sample Mean of daily returns of Benchmark : {}".format(avg_daily_ret_benchmark))
    


if __name__ == "__main__":
    test_code()
