"""  		   	  			  	 		  		  		    	 		 		   		 		  
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		   	  			  	 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		   	  			  	 		  		  		    	 		 		   		 		  
All Rights Reserved  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		   	  			  	 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		   	  			  	 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		   	  			  	 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		   	  			  	 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		   	  			  	 		  		  		    	 		 		   		 		  
or edited.  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		   	  			  	 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		   	  			  	 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		   	  			  	 		  		  		    	 		 		   		 		  
GT honor code violation.  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Student Name: Neil Watt (replace with your name)  		   	  			  	 		  		  		    	 		 		   		 		  
GT User ID: nwatt3 (replace with your User ID)  		   	  			  	 		  		  		    	 		 		   		 		  
GT ID: 903476861 (replace with your GT ID)  		   	  			  	 		  		  		    	 		 		   		 		  
"""  		   	  			  	 		  		  		    	 		 		   	
from indicators import get_indicators
from indicators import cum_return
from indicators import average_daily_ret
from indicators import standard_dev_daily_ret
from indicators import get_simple_moving_average_ratio
import datetime as dt
import pandas as pd
import util as ut
from util import get_data
import BagLearner as bl
#import QLearner as ql
import DTLearner as dl
import numpy as np
import random


def get_simple_moving_average(prices, window=20):
    return prices.rolling(window=window, min_periods=window).mean()


def get_stdev(prices, window=20, window_mean=40):
    stdev=prices.rolling(window=window, min_periods=window).std()
    sd_signal=prices.rolling(window=window_mean, min_periods=window_mean).std()
    stdev_divergence=stdev - sd_signal
    return stdev, sd_signal, stdev_divergence

class StrategyLearner(object):

    # constructor
    def __init__(self, verbose = False, impact=0.0):
        self.verbose = verbose
        self.impact = impact
        self.window_size=20
        self.feature_size = 5
        self.N = 10
        bag=20
        leaf_size = 5
        self.learner=bl.BagLearner(learner=dl.DTLearner, bags=bag, kwargs={"leaf_size":leaf_size})

    def author(self):
        return 'nwatt'

    def retrieve_price_data(self, symbol, dates):
         #from util, data file must be located in one directory level up from the active directory
        data = get_data([symbol], dates)
        prices = data[symbol]
        prices /= prices.iloc[0]
        prices_index = data["SPY"]
        prices_index /= prices_index.iloc[0]
        trading_days = prices_index.index
        return prices, prices_index, trading_days
    
    
    
    
    
    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 10000): 

        # add your code to do learning here
        window_size=self.window_size
        feature_size = self.feature_size
        N = self.N
        impact=self.impact
        threshold = max(0.05, 2 * impact)
        # example usage of the old backward compatible util function
        syms=[symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        norm_prices=prices.divide(prices.ix[0])
        #prices_SPY = prices_all['SPY']  # only SPY, for comparison later
        #if self.verbose: print prices
  
        

        #2. BB: Bollinger Band Index
        bb=norm_prices.copy()
        
        #bb['SMA']=norm_prices.rolling(window_size).mean()
        
        bb['SMA'] =get_simple_moving_average(norm_prices)
        
        #bb['STD']=norm_prices.rolling(window_size).std()
        bb['STD'],bb['stdev_signal'],bb['stdev_divergence']=get_stdev(norm_prices)
            
        bb['Upper BB']=bb['SMA']+2.0*bb['STD']
        bb['Lower BB']=bb['SMA']-2.0*bb['STD']
        bb['BBI']=(bb.ix[:, 0]-bb['Lower BB'])/(bb['Upper BB']-bb['Lower BB'])

        #3. MM: Momentum
        MM = norm_prices.copy()
        MM['Momentum'] = MM.divide(MM.shift(window_size)) - 1

        #Normalization
        # smap['SMA/P'] = (smap['SMA/P']-smap['SMA/P'].mean())/smap['SMA/P'].std()
        # bb['BBI'] = (bb['BBI']-bb['BBI'].mean())/bb['BBI'].std()
        # MM['Momentum'] = (MM['Momentum']-MM['Momentum'].mean())/MM['Momentum'].std()
        
        X_train=[]
        Y_train=[]

     
                
                
        for i in range(window_size + feature_size + 1, len(prices) - N):
            X_train.append( np.concatenate( (['SMA/P'][i - feature_size : i], bb['BBI'][i - feature_size : i], MM['Momentum'][i - feature_size : i]) ) )
            ret= (prices.values[i + N] - prices.values[i]) / prices.values[i]
            #Cal. N days return
            if ret > threshold:
                Y_train.append(1)
            elif ret < -threshold:
                Y_train.append(-1)
            else:
                Y_train.append(0)        

        X_train=np.array(X_train)
        Y_train=np.array(Y_train)

        self.learner.addEvidence(X_train, Y_train)
    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 10000):

        current_holding=0

        # here we build a fake set of trades
        # your code should return the same sort of data
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        trades = prices_all[[symbol,]]  # only portfolio symbols
        trades_SPY = prices_all['SPY']  # only SPY, for comparison later

        window_size=self.window_size
        feature_size = self.feature_size
        N=self.N

        syms=[symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        norm_prices=prices.divide(prices.ix[0])

        # Add My Indicators: SMA,BB,Momentum
        #1. SMA:
        
        prices, index_prices, trading_days = self.retrieve_price_data(symbol, pd.date_range(sd, ed))
        
        indicators = get_indicators(prices.to_frame(symbol))
        
        boll_bandr = indicators["boll_bandr"]
        simple_moving_averager = indicators["simple_moving_averager"]
        stdev_divergence=indicators["stdev_divergence"]
        
        smap=norm_prices.copy()
        smap['SMA/P']=prices.rolling(window_size).mean()/prices

        #2. BB: Bollinger Band Index
        bb=norm_prices.copy()
        bb['SMA']=norm_prices.rolling(window_size).mean()
        bb['STD']=norm_prices.rolling(window_size).std()
        bb['Upper BB']=bb['SMA']+2.0*bb['STD']
        bb['Lower BB']=bb['SMA']-2.0*bb['STD']
        bb['BBI']=(bb.ix[:, 0]-bb['Lower BB'])/(bb['Upper BB']-bb['Lower BB'])

        #3. MM: Momentum
        MM = norm_prices.copy()
        MM['Momentum'] = MM.divide(MM.shift(window_size)) - 1

        trades.values[:, :] = 0
        Xtest = []

        for i in range(window_size + feature_size + 1, len(prices) - N):
            data = np.concatenate( ( ['SMA/P'][i - feature_size : i], bb['BBI'][i - feature_size : i], MM['Momentum'][i - feature_size : i]) )
            Xtest.append(data)

        res = self.learner.query(Xtest)

        for i, r in enumerate(res):
            if r > 0:
                # Buy signal
                trades.values[i + window_size + feature_size + 1, :] = 1000 - current_holding
                current_holding = 1000
            elif r < 0:
                # Sell signal
                trades.values[i + window_size + feature_size + 1, :] = - 1000 - current_holding
                current_holding = -1000

        if self.verbose: print(type(trades)) # it better be a DataFrame!
        if self.verbose: print(trades)
        if self.verbose: print(prices_all)
        return trades

if __name__=="__main__":
    print("One does not simply think up a strategy")