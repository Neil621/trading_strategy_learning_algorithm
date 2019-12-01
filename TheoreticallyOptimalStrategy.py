import datetime as dt
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from util import get_data
from indicators import get_daily_returns
from indicators import cum_return
from indicators import average_daily_ret
from indicators import standard_dev_daily_ret
from marketsimcode import compute_portvals



np.random.seed(45892)


def author():
    return "nwatt3"


class theoreticallly_optimal_strategy(object):
    def __init__(self):
        pass

   
    def retrieve_price_data(self, symbol, dates):
         #from util, data file must be located in one directory level up from the active directory
        data = get_data([symbol], dates)
        prices = data[symbol]
        prices /= prices.iloc[0]
        prices_index = data["SPY"]
        prices_index /= prices_index.iloc[0]
        trading_days = prices_index.index
        return prices, prices_index, trading_days

    def generate_orders(self, df_positions):
        df_trades = pd.Series(index=df_positions.index)
        holding = 0
        for day in df_positions.index:
            position = df_positions.loc[day]
            if position == -1:  # sell 1000
                df_trades.loc[day] = {
                    -1000: 0,
                    0: -1000,
                    1000: -2000,
                }.get(holding)
            elif position == 1:  # buy 1000
                df_trades.loc[day] = {
                    -1000: 2000,
                    0: 1000,
                    1000: 0,
                }.get(holding)
            else:  # nothing to do
                df_trades.loc[day] = 0

            holding += df_trades.loc[day]

        return df_trades

    def testPolicy(self, symbol="JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
        # retrieve prices
        prices, prices_index, trading_days = self.retrieve_price_data(symbol, pd.date_range(sd, ed))
        daily_returns = get_daily_returns(prices)

        # trading positions for theoretical strategy
        df_positions = pd.Series(index=trading_days)
        previous_day = None
        for day in df_positions.index:
            if previous_day is None:
                df_positions.loc[day] = 0
                previous_day = day
                continue

            if daily_returns.loc[day] < 0:
                #sell
                df_positions.loc[previous_day] = -1 
            elif daily_returns.loc[day] > 0:
                df_positions.loc[previous_day] = 1 
                ## buy
            else:
                #do nothing
                df_positions.loc[previous_day] = 0 

            previous_day = day

        
        df_trades = self.generate_orders(df_positions)

        return df_trades.to_frame(symbol)


def test_code():
    theoretically_optimal_strategy = theoreticallly_optimal_strategy()

    # In sample
    symbol = "JPM"
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    period = pd.date_range(sd, ed)
    trading_days = get_data(["SPY"], dates=period).index
    sv = 100000

     #benchmark, if do no trading and hold only
    benchmark_trade = pd.DataFrame([1000] + [0] * (len(trading_days) - 1), columns=["JPM"], index=trading_days)
    benchmark = compute_portvals(benchmark_trade, start_val=sv, commission=0, impact=0)
    benchmark /= benchmark.iloc[0, :]

    # with the theoretical strategy
    df_trades = theoretically_optimal_strategy.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
    portvals = compute_portvals(df_trades, start_val=sv, commission=0, impact=0)
    portvals /= portvals.iloc[0, :]

    
    fig, ax = plt.subplots()
    benchmark.plot(ax=ax, color="g")
    portvals.plot(ax=ax, color="r")
    plt.legend(["Benchmark", "Theoretically Optimal Strategy "])
    plt.title("Theoretically Optimal Strategy, JPM (in sample)")
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.grid()
    
    plt.savefig("theoreticallly_optimal_strategy_chart.png")
    

    #stats
    cum_ret, avg_daily_ret, std_daily_ret,  = [
        cum_return(portvals),
        average_daily_ret(portvals),
        standard_dev_daily_ret(portvals),
        
    ]
    cum_ret_benchmark, avg_daily_ret_benchmark, std_daily_ret_benchmark,  = [
        cum_return(benchmark),
        average_daily_ret(benchmark),
        standard_dev_daily_ret(benchmark),
        
    ]

    print("Date Range: {} to {}".format(sd, ed))
    print
    print("Ending Portfolio Value: {}".format(portvals.iloc[-1,0]))
    print("Ending Benchmark Value: {}".format(benchmark.iloc[-1,0]))
   
    print
    # Cumulative return of the benchmark and portfolio
    print("Cumulative return of Strategy: {}".format(cum_ret))
    print("Cumulative Return of Benchmark : {}".format(cum_ret_benchmark))
    print
    # Stdev of daily returns of benchmark and portfolio
    print("Stdev  of Strategy: {}".format(std_daily_ret))
    print("Stdev  of Benchmark : {}".format(std_daily_ret_benchmark))
    print
    # Mean of daily returns of benchmark and portfolio
    print("Mean of daily returns  of Strategy: {}".format(avg_daily_ret))
    print("Mean of daily returns of Benchmark : {}".format(avg_daily_ret_benchmark))
    
    



if __name__ == "__main__":
    test_code()