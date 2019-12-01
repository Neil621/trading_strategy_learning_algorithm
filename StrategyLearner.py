"""  		   	  			  	 		  		  		    	 		 		   		 		  
Template for implementing QLearner  (c) 2015 Tucker Balch  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
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
  		   	  			  	 		  		  		    	 		 		   		 		  
Student Name: Neil Watt(replace with your name)  		   	  			  	 		  		  		    	 		 		   		 		  
GT User ID: nwatt3 (replace with your User ID)  		   	  			  	 		  		  		    	 		 		   		 		  
GT ID: 903476861  (replace with your GT ID)  		   	  			  	 		  		  		    	 		 		   		 		  
"""
import time
import datetime as dt
import matplotlib; matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from util import get_data
from indicators import get_daily_returns
from indicators import get_indicators
from indicators import cumulative_return
from marketsimcode import compute_portvals
import QLearner as ql


class StrategyLearner(object):

    def __init__(self, verbose=False, seed=False, impact=0.0, commission=0.0, min_iter=20, max_iter=100):
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.n_bins = 10
        self.num_states = self.n_bins ** 3
        self.mean = None
        self.std = None
        self.divergence_bins = None
        self.bbr_bins = None
        self.SMAr_bins = None
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.seed = seed

        # Seed for charts
        if seed:
            np.random.seed(903430342)

    def author(self):
        return "smarchienne3"

    def get_historical_data(self, symbol, dates):
        data = get_data([symbol], dates)
        prices = data[symbol]
        prices /= prices.iloc[0]
        index_prices = data["SPY"]
        index_prices /= index_prices.iloc[0]
        trading_days = index_prices.index
        return prices, trading_days

    def discretize(self, X, bins, n_bins):
        indices = np.digitize(X, bins, right=True) - 1
        indices = np.clip(indices, 0, n_bins-1)
        return indices

    def normalize_indicators(self, df, mu, sigma):
        return (df - mu) / sigma

    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 10000):

        # Grab in-sample data
        prices, trading_days = self.get_historical_data(symbol, pd.date_range(sd, ed))
        daily_returns = get_daily_returns(prices)

        # Indicators
        indicators = get_indicators(prices.to_frame(symbol))
        self.mean = indicators.mean()
        self.std = indicators.std()
        if (self.std == 0).any():
            self.std = 1
        std_indicators = self.normalize_indicators(indicators, self.mean, self.std)
        divergence = std_indicators["divergence"]
        bbr = std_indicators["bbr"]
        SMAr = std_indicators["SMAr"]

        # Discretize
        ## MACD
        _, self.divergence_bins = pd.qcut(divergence, self.n_bins, retbins=True, labels=False)
        divergence_ind = self.discretize(divergence, self.divergence_bins, self.n_bins)
        divergence_ind = pd.Series(divergence_ind, index=indicators.index)
        ## Bollinger Bands
        _, self.bbr_bins = pd.qcut(bbr, self.n_bins, retbins=True, labels=False)
        bbr_ind = self.discretize(bbr, self.bbr_bins, self.n_bins)
        bbr_ind = pd.Series(bbr_ind, index=indicators.index)
        ## SMA
        _, self.SMAr_bins = pd.qcut(SMAr, self.n_bins, retbins=True, labels=False)
        SMAr_ind = self.discretize(SMAr, self.SMAr_bins, self.n_bins)
        SMAr_ind = pd.Series(SMAr_ind, index=indicators.index)

        # Compute states of in-sample data
        discretized_indicators = pd.DataFrame(index=indicators.index)
        discretized_indicators["divergence"] = divergence_ind.values
        discretized_indicators["bbr"] = bbr_ind.values
        discretized_indicators["SMAr"] = SMAr_ind.values
        discretized_indicators["mapping"] = divergence_ind.astype(str) + bbr_ind.astype(str) + SMAr_ind.astype(str)
        discretized_indicators["state"] = discretized_indicators["mapping"].astype(np.int)
        states = discretized_indicators["state"]

        # QLearner
        self.learner = ql.QLearner(
            num_states=self.num_states,
            num_actions=3,
            alpha=0.2,
            gamma=0.9,
            rar=0.5,
            radr=0.99,
            dyna=0,
            verbose=self.verbose,
            seed=self.seed
        )

        # Training loop
        i = 0
        converged = False
        df_trades_previous = None
        while (i <= self.min_iter) or (i <= self.max_iter and not converged):

            # Set state with indicators of this first day
            action = self.learner.querysetstate(states.iloc[0])

            holding = 0
            df_trades = pd.Series(index=states.index)
            for day, state in states.iteritems():
                reward = holding * daily_returns.loc[day]
                if action != 2: # LONG or SHORT?
                    reward *= (1 - self.impact)
                action = self.learner.query(state, reward)
                if action == 0:  # SHORT
                    df_trades.loc[day] = {
                        -1000: 0,
                        0: -1000,
                        1000: -2000,
                    }.get(holding)
                elif action == 1:  # LONG
                    df_trades.loc[day] = {
                        -1000: 2000,
                        0: 1000,
                        1000: 0,
                    }.get(holding)
                elif action == 2:  # DO NOTHING
                    df_trades.loc[day] = 0
                else:
                    raise Exception("Unknown trading action to take: {}".format(action))

                holding += df_trades.loc[day]

            # Check for convergence
            if (df_trades_previous is not None) and (df_trades.equals(df_trades_previous)):
                converged = True

            df_trades_previous = df_trades
            i += 1


    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 10000):

        # Grab out-of-sample data
        prices, trading_days = self.get_historical_data(symbol, pd.date_range(sd, ed))

        # Indicators
        indicators = get_indicators(prices.to_frame(symbol))
        std_indicators = self.normalize_indicators(indicators, self.mean, self.std)
        divergence = std_indicators["divergence"]
        bbr = std_indicators["bbr"]
        SMAr = std_indicators["SMAr"]

        # Discretize
        ## MACD
        divergence_ind = self.discretize(divergence, self.divergence_bins, self.n_bins)
        divergence_ind = pd.Series(divergence_ind, index=indicators.index)
        ## Bollinger Bands
        bbr_ind = self.discretize(bbr, self.bbr_bins, self.n_bins)
        bbr_ind = pd.Series(bbr_ind, index=indicators.index)
        ## SMA
        SMAr_ind = self.discretize(SMAr, self.SMAr_bins, self.n_bins)
        SMAr_ind = pd.Series(SMAr_ind, index=indicators.index)

        # Compute states of out-of-sample data
        discretized_indicators = pd.DataFrame(index=indicators.index)
        discretized_indicators["divergence"] = divergence_ind.values
        discretized_indicators["bbr"] = bbr_ind.values
        discretized_indicators["SMAr"] = SMAr_ind.values
        discretized_indicators["mapping"] = divergence_ind.astype(str) + bbr_ind.astype(str) + SMAr_ind.astype(str)
        discretized_indicators["state"] = discretized_indicators["mapping"].astype(np.int)
        states = discretized_indicators["state"]

        holding = 0
        df_trades = pd.Series(index=states.index)
        for day, state in states.iteritems():
            action = self.learner.querysetstate(state, random=False)
            if action == 0:  # SHORT
                df_trades.loc[day] = {
                    -1000: 0,
                    0: -1000,
                    1000: -2000,
                }.get(holding)
            elif action == 1:  # LONG
                df_trades.loc[day] = {
                    -1000: 2000,
                    0: 1000,
                    1000: 0,
                }.get(holding)
            elif action == 2:  # DO NOTHING
                df_trades.loc[day] = 0
            else:
                raise Exception("Unknown trading action to take: {}".format(action))

            holding += df_trades.loc[day]

        return df_trades.to_frame(symbol)


def test_code():

    sv = 100000
    commission = 0.0
    impact = 0.0
    sd_in = dt.datetime(2008, 1, 1)
    ed_in = dt.datetime(2009, 12, 31)
    sd_out = dt.datetime(2010, 1, 1)
    ed_out = dt.datetime(2011, 12, 31)
    bench = lambda trading_days: [1000] + [0] * (len(trading_days) - 2) + [-1000]

    for symbol in ["JPM", "ML4T-220", "AAPL", "UNH", "SINE_FAST_NOISE"]:

        print("#######################")
        print("{}".format(symbol))
        print("#######################")

        ############################
        #
        # In sample
        #
        ############################

        # In sample
        period = pd.date_range(sd_in, ed_in)
        trading_days = get_data(["SPY"], dates=period).index

        # Benchmark in-sample
        benchmark_trade = pd.DataFrame(bench(trading_days), columns=[symbol], index=trading_days)
        benchmark = compute_portvals(benchmark_trade, start_val=sv, commission=commission, impact=impact)
        benchmark /= benchmark.iloc[0, :]

        # Train
        print("Training...")
        learner = StrategyLearner(verbose=False, seed=True, impact=impact, commission=commission)
        start = time.time()
        learner.addEvidence(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv)
        print("addEvidence() on in-sample completes in in {}sec".format(time.time()-start))

        # Test : in-sample
        print("Testing in-sample...")
        start = time.time()
        df_trades = learner.testPolicy(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv)
        print("testPolicy() on in-sample completes in in {}sec".format(time.time() - start))
        portvals_train = compute_portvals(df_trades, start_val=sv, commission=commission, impact=impact)
        portvals_train /= portvals_train.iloc[0, :]
        
        if cumulative_return(portvals_train) > 1.0:
            print("cumulative return in-sample greater than 1.0: {}".format(cumulative_return(portvals_train)))
        else:
            print("ERROR cumulative return in-sample NOT greater than 1.0: {}".format(cumulative_return(portvals_train)))

        if cumulative_return(portvals_train) > cumulative_return(benchmark):
            print("cumulative return in-sample greater than benchmark: {} vs {}".format(
                cumulative_return(portvals_train), cumulative_return(benchmark))
            )
        else:
            print("ERROR cumulative return in-sample NOT greater than benchmark: {} vs {}".format(
                cumulative_return(portvals_train), cumulative_return(benchmark)
            ))

        fig, ax = plt.subplots()
        benchmark.plot(ax=ax, color="g")
        portvals_train.plot(ax=ax, color="r")
        plt.legend(["Benchmark", "QLearner Strategy"])
        plt.title("QLearner Strategy on {} stock over in-sample period".format(symbol))
        plt.xlabel("Dates")
        plt.ylabel("Normalized value")
        for day, order in df_trades[df_trades[symbol] != 0].iterrows():
            if order[symbol] < 0:  # SHORT
                ax.axvline(day, color="k", alpha=0.5)
            elif order[symbol] > 0:  # LONG
                ax.axvline(day, color="b", alpha=0.5)
        plt.tight_layout()
        plt.savefig("QLearner_{}_in_sample.png".format(symbol))

        ############################
        #
        # Out of sample
        #
        ############################

        # Test : out-of-sample
        print("Testing out-of-sample...")
        period = pd.date_range(sd_out, ed_out)
        trading_days = get_data(["SPY"], dates=period).index

        # Benchmark out of sample
        benchmark_trade = pd.DataFrame(bench(trading_days), columns=[symbol], index=trading_days)
        benchmark = compute_portvals(benchmark_trade, start_val=sv, commission=commission, impact=impact)
        benchmark /= benchmark.iloc[0, :]

        # testPolicy out of sample
        start = time.time()
        df_trades = learner.testPolicy(symbol=symbol, sd=sd_out, ed=ed_out, sv=sv)
        print("testPolicy() on out-of-sample completes in in {}sec".format(time.time() - start))
        portvals_test = compute_portvals(df_trades, start_val=sv, commission=commission, impact=impact)
        portvals_test /= portvals_test.iloc[0, :]

        if cumulative_return(portvals_test) > 1.0:
            print("cumulative return out-of-sample greater than 1.0: {}".format(cumulative_return(portvals_test)))
        else:
            print("ERROR cumulative return out-of-sample NOT greater than 1.0: {}".format(cumulative_return(portvals_test)))

        if cumulative_return(portvals_test) > cumulative_return(benchmark):
            print("cumulative return out-of-sample greater than benchmark: {} vs {}".format(
                cumulative_return(portvals_test), cumulative_return(benchmark)
            ))
        else:
            print("ERROR cumulative return out-of-sample NOT greater than benchmark: {} vs {}".format(
                cumulative_return(portvals_test), cumulative_return(benchmark)
            ))

        fig, ax = plt.subplots()
        benchmark.plot(ax=ax, color="g")
        portvals_test.plot(ax=ax, color="r")
        plt.legend(["Benchmark", "QLearner Strategy"])
        plt.title("QLearner Strategy on {} stock over out-of-sample period".format(symbol))
        plt.xlabel("Dates")
        plt.ylabel("Normalized value")
        for day, order in df_trades[df_trades[symbol] != 0].iterrows():
            if order[symbol] < 0:  # SHORT
                ax.axvline(day, color="k", alpha=0.5)
            elif order[symbol] > 0:  # LONG
                ax.axvline(day, color="b", alpha=0.5)
        plt.tight_layout()
        plt.savefig("QLearner_{}_out_of_sample.png".format(symbol))


def tests():
    import time
    symbol = "JPM"
    sd_in = dt.datetime(2008, 1, 1)
    ed_in = dt.datetime(2009, 12, 31)
    sv = 10000

    # Make sure testPolicy() always return the same result
    learner = StrategyLearner(verbose=False, seed=False)
    learner.addEvidence(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv)
    df_trades_previous = None
    for i in range(20):
        df_trades = learner.testPolicy(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv)
        if (df_trades_previous is not None) and not (df_trades_previous.equals(df_trades)):
            raise Exception("testPolicy() does not always return the same result")
    print("OK: testPolicy() always returns the same result")

    # testPolicy() method should be much faster than your addEvidence() method
    learner = StrategyLearner(verbose=False, seed=False)
    start = time.time()
    learner.addEvidence(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv)
    time_addEvidence = time.time() - start

    start = time.time()
    learner.testPolicy(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv)
    time_testPolicy = time.time() - start

    if time_testPolicy >= time_addEvidence:
        print("testPolicy() is not faster than addEvidence()!!! {} VS {}".format(time_testPolicy, time_addEvidence))
    print("OK: testPolicy() faster than addEvidence(): {} VS {}".format(time_testPolicy, time_addEvidence))


if __name__ == "__main__":
    test_code()
    #tests()
