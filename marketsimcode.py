from datetime import datetime
import pandas as pd
from util import get_data


def author():
    return "nwatt3"


def compute_portvals(df_trades, start_val=1000000, commission=9.95, impact=0.005):
    """Improved version of marketsim, accepting trading Dataframes"""
    start_date = df_trades.index.min()
    end_date = df_trades.index.max()

    # Fetch stocks prices
    stocks = df_trades.columns.tolist()
    stocks_dict = {}
    for symbol in stocks:
        stocks_dict[symbol] = get_data([symbol], pd.date_range(start_date, end_date), colname='Adj Close')
        stocks_dict[symbol] = stocks_dict[symbol].resample("D").fillna(method="ffill")
        stocks_dict[symbol] = stocks_dict[symbol].fillna(method="bfill")

    # List the trading days
    SPY = get_data(['SPY'], pd.date_range(start_date, end_date))
    trading_days = pd.date_range(start_date, end_date, freq="D")
    not_trading_days = []
    for day in trading_days:
        if day not in SPY.index:
            not_trading_days.append(day)
    trading_days = trading_days.drop(not_trading_days)

    for day in df_trades.index:
        if day not in trading_days:
            raise Exception("One of the order day is missing in trading days")

    # Initialization of portfolio DataFrame
    portvals = pd.DataFrame(index=trading_days, columns=["portfolio_value"] + stocks)

    # Compute portfolio value for each trading day in the period
    current_value = start_val
    previous_day = None
    for today in trading_days:

        # Copy previous trading day's portfolio state
        if previous_day is not None:
            portvals.loc[today, :] = portvals.loc[previous_day, :]
            portvals.loc[today, "portfolio_value"] = 0
        else:
            portvals.loc[today, :] = 0

        # Execute orders
        if today in df_trades.index:
            today_orders = df_trades.loc[[today]]
            for symbol in today_orders.columns:
                order = today_orders.iloc[0].loc[symbol] # assume one order per day per symbol
                shares = abs(order)
                stock_price = stocks_dict[symbol].loc[today, symbol]

                if order > 0: # BUY
                    stock_price = (1 + impact) * stock_price
                    current_value -= stock_price * shares
                    current_value -= commission
                    portvals.loc[today, symbol] += shares
                elif order < 0: # SELL
                    stock_price = (1 - impact) * stock_price
                    current_value += stock_price * shares
                    current_value -= commission
                    portvals.loc[today, symbol] -= shares

        # Update portfolio value
        for symbol in stocks:
            stock_price = stocks_dict[symbol].loc[today, symbol]
            portvals.loc[today, "portfolio_value"] += portvals.loc[today, symbol] * stock_price
        portvals.loc[today, "portfolio_value"] += current_value

        previous_day = today

    # Remove empty lines
    portvals = portvals.sort_index(ascending=True)
    return portvals.iloc[:,0].to_frame()


if __name__ == "__main__":
    start_date = datetime(2008, 1, 2)
    end_date = datetime(2008, 1, 4)
    df_trades = pd.DataFrame([1000, 0, -1000], columns=["JPM"], index=pd.date_range(start_date, end_date))
    portvals = compute_portvals(df_trades)
    print(portvals)