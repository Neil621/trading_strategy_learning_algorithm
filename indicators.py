import datetime as dt
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from util import get_data



np.random.seed(5793)


def author():
    return "nwatt3"



#calulated values

def standard_dev_daily_ret(port_val):
    daily_ret = (port_val / port_val.shift(1)) - 1
    return daily_ret.iloc[1:,0].std()


def cum_return(port_val):
    cum_ret = (port_val / port_val.iloc[0,0]) - 1
    return cum_ret.iloc[-1,0]




def sharpe_ratio(port_val, n=np.sqrt(252)):
    return n * average_daily_ret(port_val) / standard_dev_daily_ret(port_val)

def average_daily_ret(port_val):
    daily_ret = (port_val / port_val.shift(1)) - 1
    return daily_ret.iloc[1:,0].mean()


#indicators (SMA, standard dev, bollinger band)


def get_stdev(prices, window=20, window_mean=40):
    stdev=prices.rolling(window=window, min_periods=window).std()
    sd_signal=prices.rolling(window=window_mean, min_periods=window_mean).std()
    stdev_divergence=stdev - sd_signal
    return stdev, sd_signal, stdev_divergence
    
def get_boll_band_ratio(prices, simple_moving_average, stdev):
    return (prices - (simple_moving_average- 2 * stdev)) / (4 * stdev)


def get_daily_returns(prices):
    return prices / prices.shift(1) - 1



def get_simple_moving_average(prices, window=20):
    return prices.rolling(window=window, min_periods=window).mean()


def get_simple_moving_average_ratio(prices, window=20):
    return prices / prices.rolling(window=window, min_periods=window).mean()




def get_momentum(prices, n=10):
    return prices / prices.shift(n-1) - 1




def get_indicators(prices):
    daily_returns = get_daily_returns(prices)
    
    
    
    
    #stdev = get_stdev(prices)
    stdev, stdev_signal, stdev_divergence=get_stdev(prices)
    
    
    simple_moving_average = get_simple_moving_average(prices)
    simple_moving_average_ratio = get_simple_moving_average_ratio(prices)
    #momentum = get_momentum(prices)
    boll_band_ratio = get_boll_band_ratio(prices, simple_moving_average, stdev)
    #MACD, MACD_signal, divergence = get_MACD(prices)
    df = pd.concat([
        prices, daily_returns, stdev,stdev_signal,stdev_divergence, simple_moving_average, simple_moving_average_ratio, boll_band_ratio
       
    ], axis=1)
    
   
    df.columns = prices.columns.tolist() + [
        "daily_returns", "stdev","stdev_signal","stdev_divergence","simple_moving_average", "simple_moving_averager", "boll_bandr",
    ]


    
    return df


def test_code():
    plt.rcParams['axes.grid'] = True

    # In sample data
    start_in_sample = dt.datetime(2010, 1, 1)
    end_in_sample = dt.datetime(2011, 12, 31)
    in_sample = pd.date_range(start_in_sample, end_in_sample)
    df_in_sample = get_data(["JPM"], in_sample)
    df_in_sample = df_in_sample / df_in_sample.iloc[0,:]
   

    # Indicators
    JPM_prices = df_in_sample[["JPM"]]
    df = get_indicators(JPM_prices)
   

 
  
    
    
   
    
    #### standard deviation  
    
    
    
    fig, ax = plt.subplots(3, 1, sharex=True)
    
    #using same period as in the "theoretically optimim
    start_period = JPM_prices.index.min() 
     
    end_period = JPM_prices.index.max() #
    df.loc[start_period:end_period][["JPM"]].plot(color="black", linewidth=0.95, ax=ax[0])
    
    df.loc[start_period:end_period][["stdev", "stdev_signal"]].plot(ax=ax[2])
    df.loc[start_period:end_period][["stdev_divergence"]].plot(ax=ax[1]); ax[1].axhline(y=0, color="r")
    plt.xlabel("Dates")
    plt.ylabel("Standard Deviation")
    plt.tight_layout()
    
    plt.savefig("indicators_stdev.png")
    
    
    
    

    ## Bollinger band ratio
    fig, ax = plt.subplots(2, 1, sharex=True)
    start_period = JPM_prices.index.min()
    end_period = JPM_prices.index.max()
    df.loc[start_period:end_period][["JPM"]].plot(color="black", linewidth=0.95, ax=ax[0])
    # calc upper and lower bands
    boll_band_low = df.loc[start_period:end_period]["simple_moving_average"] - 2 * df.loc[start_period:end_period]["stdev"]
    boll_band_up = df.loc[start_period:end_period]["simple_moving_average"] + 2 * df.loc[start_period:end_period]["stdev"]
    
    #plot bands
    df.loc[start_period:end_period][["simple_moving_average"]].plot(color="red", linewidth=0.95, ax=ax[0])
    boll_band_up.plot(color="green", linewidth=0.75, ax=ax[0])
    boll_band_low.plot(color="green", linewidth=0.75, ax=ax[0])
    #ax[0].fill_between(
     #   df.loc[start_period:end_period]["simple_moving_average"].index, boll_band_low.values, boll_band_up.values,
      #  color="b",
       # alpha=0.15
    #)
    # Plot ratio
    df.loc[start_period:end_period][["boll_bandr"]].plot(legend=None, ax=ax[1])
    ax[1].axhline(y=1, color="black"); ax[1].axhline(y=0, color="black")
    plt.xlabel("Dates")
    plt.ylabel("Normalized value Bollinger Band Ratio")
    plt.tight_layout()
    
    plt.savefig("indicators_boll_bandr.png")
    


    # simple_moving_average ratio
    fig, ax = plt.subplots(2, 1, sharex=True)
    start_period = JPM_prices.index.min()
    end_period = JPM_prices.index.max()
    df.loc[start_period:end_period][["JPM"]].plot(color="black", linewidth=0.95, ax=ax[0])
    df.loc[start_period:end_period][["simple_moving_average"]].plot(color="b", linewidth=0.95, ax=ax[0])
    # Plot ratio
    df.loc[start_period:end_period]["simple_moving_averager"].plot(color="r", linewidth=0.95, ax=ax[1])
    ax[1].axhline(y=1, color="black")
    plt.xlabel("Dates")
    plt.ylabel("Normalized value")
    plt.tight_layout()
    
    plt.savefig("indicators_simple_moving_averager.png")
    
    



if __name__ == "__main__":
    test_code()