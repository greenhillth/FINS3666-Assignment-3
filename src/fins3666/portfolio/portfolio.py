import pandas as pd
import numpy as np
import uuid
import logging
from typing import Union, List
from datetime import datetime, timedelta

from fins3666.defines import Order, AssetSpreads


class Portfolio:

    def __init__(self, timestamp: datetime, balance: Union[dict, List[dict]], fx_data: None):
        """
        Initialize a Portfolio.

        Args:
            starting_balance (List[AssetDict], optional):
                A list of initial assets to populate the portfolio.

        Example:
            >>> portfolio = Portfolio(timestamp=datetime(2020,09,1), starting_balance=[{
            >>>     "asset": "USD",
            >>>     "units": 12000,
            >>>     "yield": 2,
            >>>     "unit_value_USD": 1},
            >>>     {
            >>>     "asset": "AUD",
            >>>     "units": 0,
            >>>     "unit_value_USD": 1}
            >>> ])

        Returns:
            None

        """

        # Portfolio Characteristics (Scalar)
        self.sharpe_ratio = None
        self.sortino_ratio = None
        self.max_drawdown = None
        self.volatility = None
        self.beta = None
        self.alpha = None

        # Create Position Ledger with initial balance, if supplied
        self.ledger = self._build_ledger_df(balance)
        # Create transaction record
        self.trades = self._build_trade_df()
        # Create forex data

        self.forex_spreads = AssetSpreads(None, None, None)
        self.updateMarketData(fx_data)

        self.orders = []
        self.trade_log = []

        self.t_log = logging.getLogger("TradeLogger")
        self.t_log.setLevel(logging.INFO)

        if not self.t_log.handlers:
            file_handler = logging.FileHandler("trades.log", mode='w')
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.t_log.addHandler(file_handler)

        self.timestamp = timestamp

    def new_order(self, order: Order):
        self.orders.append(order)

    def update(self, timestamp: datetime):
        self._index_positions(timestamp)
        self.process_orders(timestamp)

    def _index_positions(self, newTime):
        deltaYears = (newTime - self.timestamp).days/365.25

        self.ledger.at['USD', 'Units'] -= 12*1e6

        self.ledger['Units'] = self.ledger.apply(
            lambda row: row['Units'] * ((row['YieldPA']) * deltaYears +
                                        1) if abs(row['YieldPA']) != 0 else row['Units'],
            axis=1
        )

        self.ledger.at['USD', 'Units'] += 12*1e6

        self.timestamp = newTime

    def process_orders(self, timestamp: datetime):
        remaining = []
        for order in self.orders:
            if (order.asset == order.currency):
                continue
            bid, ask, mid = self.asset_lookup(self.forex_spreads, order.tic())
            if order.expiry is not None and order.expiry < timestamp:
                self.trade_log.append(
                    order.log(timestamp=timestamp, status='cancelled'))
            elif order.order_type == 'market':
                price = mid
                if order.order == 'buy':
                    price = ask
                    self.execute_trade(order, price, timestamp)
                elif order.order == 'sell':
                    price = bid
                    self.execute_trade(order, price, timestamp)
                self.trade_log.append(
                    order.log(timestamp=timestamp, status='executed', price=price))
            elif order.order_type == 'limit':
                if order.order == 'buy' and ask <= order.limit:
                    self.execute_trade(order, ask, timestamp)
                    self.trade_log.append(
                        order.log(timestamp=timestamp, status='executed', price=ask))
                elif order.order == 'sell' and bid >= order.limit:
                    self.execute_trade(order, bid, timestamp)
                    self.trade_log.append(
                        order.log(timestamp=timestamp, status='executed', price=bid))
            else:
                remaining.append(order)

        self.orders = remaining

    def execute_trade(self, order, price, timestamp):
        msg = str()
        if order.order == 'buy':
            msg += (
                f'Executed {order.order_type} buy order of {order.units:>18,.4f} units of {order.asset}'
                f' for {order.currency} {order.sym():>4}{price:,.4f} per unit.'
                f' Transaction cost: {order.sym()}{(order.units*price):,.2f}'
            )
        else:
            msg += (
                f'Executed {order.order_type} sell order of {order.units:>18,.4f} units of {order.asset}'
                f' for {order.currency} {order.sym():>4}{price:,.4f} per unit.'
                f' Transaction gain: {order.sym()}{(order.units*price):,.2f}'
            )
        self.t_log.info(msg)

        self._add_trade(order=order, price=price, timestamp=timestamp)

    def updateMarketData(self, currentData: Union[None, dict, List[dict]], yields=None):
        """
        Updates the market data for the portfolio.

        Args:
            currentData (Union[dict, List[dict]]): A dictionary or list of dictionaries containing market data.
                Each dictionary should have the following structure, formatted by the `format_market_data` function:
                {
                    "tic": str,            # The asset identifier (e.g., "CBA.AX", "USD/EUR").
                    "timestamp": datetime, # The timestamp of the market data.
                    "bid": float,          # The bid price of the asset.
                    "ask": float,          # The ask price of the asset.
                    "mid": float           # The mid price of the asset.
                }

        Returns:
            None
        """
        if currentData is None:
            return

        if isinstance(currentData, dict):
            currentData = [currentData]

        def preprocess_fx_pair(d):
            if isinstance(d.get('tic', ''), str) and 'USD' in d['tic'] and '/' in d['tic']:
                base, quote = d['tic'].split('/')
                if base == 'USD':
                    return {
                        'tic': quote,
                        'timestamp': pd.to_datetime(d['timestamp']),
                        'bid': d['bid'],
                        'ask': d['ask'],
                        'mid': d['mid']
                    }
                else:
                    return {
                        'tic': base,
                        'timestamp': pd.to_datetime(d['timestamp']),
                        'bid': (1/d['ask']),
                        'ask': (1/d['bid']),
                        'mid': d['mid']
                    }
            return None

        usd_row = pd.DataFrame([{'tic': 'USD', 'timestamp': datetime(
            2000, 1, 1), 'bid': 1, 'ask': 1, 'mid': 1}])
        fx_df = pd.DataFrame(
            filter(None, (preprocess_fx_pair(d) for d in currentData)))

        fx_df = pd.concat([usd_row, fx_df], ignore_index=True)
        self.forex_spreads = AssetSpreads(*Portfolio.buildForexMatrix(fx_df))
        self.yields = yields

    def summary(self):
        """
        Builds a portfolio snapshot by getting current market values of assets from the ledger.
        """

        fx = self.forex_spreads
        summary_df = pd.DataFrame([
            {
                'Asset': asset,
                'Units': row['Units'],
                'USD Unit Val': 1/fx.mid['USD'].get(asset, 1),
                'Liquidation Value': fx.bid.at['USD', asset] * row['Units'] if row['Units'] > 0 else fx.ask.at['USD', asset] * row['Units'],
                'USD Total Val': row['Units'] / fx.mid['USD'].get(asset, 0),
                'Yield': row['YieldPA']
            }
            for asset, row in self.ledger.iterrows()])

        return summary_df

    def to_string(self):
        """
        Returns a string representation of the portfolio.

        Args:
            None

        Returns:
            str: String representation of the portfolio.
        """
        return self.__str__()

    """Private methods for Portfolio class"""

    def _build_ledger_df(self, startingBal: Union[None, dict, List[dict]]):
        """
        Builds the internal pandas DataFrame representing the ledger containing all open and closed positions.
        Args:
            startingAssets (list of AssetDict, optional):
                A single asset dictionary or a list of asset dictionaries containing:
                    - 'name' (str): asset name or identifier
                    - 'units' (np.float64): quantity of asset held
                    - 'yield' (np.float64): p.a return
                    - 'timestamp' (datetime): time of asset acquisition
        Returns:
            pd.DataFrame: DataFrame representing the ledger.
        """
        if startingBal is not None:
            if isinstance(startingBal, dict):
                startingBal = [startingBal]
            df = pd.DataFrame([{
                'Asset': d['asset'],
                'Units': d['units'],
                'YieldPA': 1,
                'TransactionIdxs': []}
                for d in startingBal])
        else:
            df = pd.DataFrame({
                'Asset': pd.Series(dtype='str'),
                'Units': pd.Series(dtype='float64'),
                'YieldPA': pd.Series(dtype='float64'),
                'TransactionIdxs': pd.Series(dtype='object')})

        df.set_index('Asset', inplace=True)

        return df

    def _build_mkt_df(self):
        """
        Builds an empty market data DataFrame.

        Args:
            None

        Returns:
            pd.DataFrame: An empty market data dataframe with predefined columns.
        """
        df = pd.DataFrame({
            'Asset': pd.Series(dtype='str'),
            'Timestamp': pd.Series(dtype='datetime64[ns]'),
            'Bid': pd.Series(dtype='float64'),
            'Ask': pd.Series(dtype='float64'),
            'Mid': pd.Series(dtype='float64')})
        df.set_index('Asset', inplace=True)

        df.loc['USD'] = [datetime(1999, 1, 1), 1.0, 1.0, 1.0]

        return df

    def _build_trade_df(self):
        """
        Builds an empty trades DataFrame.

        Args:
            None

        Returns:
            pd.DataFrame: An empty trades dataframe with predefined columns.
        """

        df = pd.DataFrame({
            'Timestamp': pd.Series(dtype='datetime64[ns]'),
            'Asset': pd.Series(dtype='str'),
            'Buy/Sell': pd.Series(dtype='str'),
            'Units': pd.Series(dtype='float64'),
            'UnitPrice': pd.Series(dtype='float64'),
            'Balance': pd.Series(dtype='float64'),
            'BaseCurrency': pd.Series(dtype='float64'),
            'OrderValue': pd.Series(dtype='float64'),
            'ForecastReturn': pd.Series(dtype='float64'),
            'i_USD': pd.Series(dtype='float64'),
            'OrderType': pd.Series(dtype='str'),
            'Exchange': pd.Series(dtype='str'),
            'AssetType': pd.Series(dtype='str'),
            'ID': pd.Series(dtype='str')})
        df.set_index('ID', inplace=True)
        return df

    def _add_trade(self, timestamp: datetime, order: Order, price):
        """
        Appends an executed order to the trades record and updates the internal ledger.

        Args:
            order (AssetDict):
                A dictionary representing a single asset trade, containing:
                    - 'timestamp' (datetime): trade timestamp
                    - 'asset' (str): asset name
                    - 'units' (float): number of units traded
                    - 'unit_price_usd' (float): price per unit
                    - 'order_type' (str, optional): 'market', 'limit'.
                    - 'exchange' (str, optional): exchange name
                    - 'asset_type' (str): type of asset
                    - (Optional fields): 'order_value_usd', 'forecast_return', 'i_usd'

        Example:
            >>> trade = {
            >>>     "timestamp": datetime.now(),
            >>>     "asset": "ETH",
            >>>     "units": 10,
            >>>     "unit_price_usd": 2000,
            >>>     "order_type": "buy",
            >>>     "exchange": "Coinbase",
            >>>     "asset_type": "Crypto"
            >>> }
            >>> portfolio.add_trade(trade)

        Returns:
            None
        """
        id = str(uuid.uuid1())

        '''
            trade is just an increase of one asset for a decrease of another

            order units is in terms of order.asset (i.e Order(order='buy', asset='JPY', units=100, currency='USD'), price=0.1)
            means an increase of 100 units of JPY corresponds to a decrease of price*100 units of USD 

            conversely (Order(order='sell', asset='JPY', units=50, currency='USD'), price=0.09) would correspond with a decrease
            of 50 units of JPY and an increase of price*50 units of USD

            This function needs to adjust the position ledger to account for these increases and decreases
        '''

        # TODO - consolodate after debugging
        if order.order == 'buy':
            buyunits = order.units
            sellunits = -order.units*price
            self._update_ledger(order.asset, buyunits, id,
                                self._yields(order.asset))
            self._update_ledger(order.currency, sellunits,
                                id, self._yields(order.currency))
        elif order.order == 'sell':
            sellunits = -order.units
            buyunits = order.units*price
            self._update_ledger(order.asset, sellunits, id,
                                self._yields(order.asset))
            self._update_ledger(order.currency, buyunits,
                                id, self._yields(order.currency))
        units = order.units

        trade = pd.DataFrame([{'Timestamp': timestamp, 'Asset': order.asset, 'Units': units, 'UnitPrice': price, 'BaseCurrency': order.currency,
                               'OrderValue': units*price, 'OrderType': order.order, 'Exchange': order.exchange, 'AssetType': order.asset_type}], index=[id])

        self.trades = pd.concat([self.trades, trade], ignore_index=True)

    def _yields(self, asset):
        if self.yields is None:
            return (0, 0)

        yields = self.yields[[f'{asset} SHORT', f'{asset} LONG']].iloc[-1]
        return tuple(yields/100)

    def _update_ledger(self, asset, units, tradeID, yield_pa=(0, 0)):
        """
        Update the position ledger with details of trade. Should only be called by the add_trade method.
        Args:
            asset (str): Asset identifier.
            units (float): Units to add/remove from asset
            transaction_id (int): Unique identifier for the transaction.
            yield_pa (Tuple(float)): Two - element tuple containing [LONG, SHORT] return profiles (i.e holding and borrowing costs).
        Returns:
            None
        """
        if asset in self.ledger.index:
            self.ledger.at[asset, 'Units'] += units
            self.ledger.at[asset, 'YieldPA'] = 0
            self.ledger.at[asset, 'TransactionIdxs'].append(tradeID)
        else:
            self.ledger.loc[asset] = {
                'Units': units,
                'YieldPA': 0.0,
                'TransactionIdxs': [tradeID]
            }
        if 1e-6 > np.fabs(self.ledger.at[asset, 'Units']):
            self.ledger.at[asset, 'Units'] = 0
        elif self.ledger.at[asset, 'Units'] > 0:
            self.ledger.at[asset, 'YieldPA'] = yield_pa[0]
        elif self.ledger.at[asset, 'Units'] < 0:
            self.ledger.at[asset, 'YieldPA'] = yield_pa[1]

    """Namespace methods for Portfolio class"""

    def __str__(self):
        df = self.summary()
        linewidth = 100

        totalValue = df['USD Total Val'].sum()

        df['Weight'] = df['USD Total Val']/totalValue

        out = f"\n\n{str('Portfolio Overview'):^100}" + "\n"
        out += "=" * linewidth + "\n"
        out += f"{'Asset':<9}{'Value':<11}{'Units':<15}{'Total Value (USD)':<28}{'Position':<15}{'Yield p.a':<15}{'Weight':<6}\n"
        out += "=" * linewidth + "\n"
        for asset, row in df.iterrows():
            out += (
                f"{str(row['Asset']):<7}"
                f"{f'${row['USD Unit Val']:,.2f}':>7}"
                f"{abs(row['Units']):>14,.0f}"
                f"{'-$' if row['USD Total Val'] < 0 else '$':>6}"
                f"{f'{abs(row['USD Total Val']):,.2f}':>18}"
                "          "
                f"{'Short' if row['Units'] < 0 else 'Long':>7}"
                f"{row['Yield']:>16.2%}"
                f"{row['Weight']:>13.2%}\n"
            )
        out += f"\nPortfolio - {len(df)} assets\n"
        out += f"Timestamp: {self.timestamp},\n"
        out += f"Net Value ($USD): ${totalValue:,.2f}"

        # get sum of
        return str(out)

    def __repr__(self):
        return f"{len(self.assets)} Asset, USD${self.total_value:,.2f} Nominal Value Portfolio Object"

    """Static Methods"""
    @staticmethod
    def buildForexMatrix(df):
        """
        Builds a forex matrix from a row of data containing bid, ask, and mid prices for various currencies.
        Args:
            df (pd.DataFrame): A forex dataframe with bid, ask, and mid prices for currencies, quoted in top row.
            Example:
               tic  timestamp         bid         ask         mid
            0  USD 2000-01-01    1.000000    1.000000    1.000000
            1  EUR 2023-01-02    0.936856    0.937119    0.936986
            2  JPY 2023-01-02  131.930000  131.960000  131.944990
            3  SEK 2023-01-02   10.414800   10.424200   10.419528
            4  CAD 2023-01-02    1.354800    1.355100    1.354934
            5  CHF 2023-01-02    0.925000    0.925400    0.925181
            6  NZD 2023-01-02    1.580528    1.581778    1.581137
            7  NOK 2023-01-02    9.848000    9.854000    9.850985
            8  GBP 2023-01-02    0.831186    0.831463    0.831324
            9  AUD 2023-01-02    1.474274    1.474926    1.474603
        Returns:
            tuple: A tuple containing three DataFrames: bid matrix, ask matrix, and mid matrix.
        """
        currencies = df.loc[:, 'tic']
        bid_vec = df.loc[:, 'bid'].to_numpy()
        ask_vec = df.loc[:, 'ask'].to_numpy()
        mid_vec = df.loc[:, 'mid'].to_numpy()

        # Vectorized outer division
        bid_matrix = bid_vec[:, None] / ask_vec[None, :]
        ask_matrix = ask_vec[:, None] / bid_vec[None, :]
        mid_matrix = mid_vec[:, None] / mid_vec[None, :]

        # Wrap in DataFrames with labels
        bid_df = pd.DataFrame(bid_matrix, index=currencies, columns=currencies)
        ask_df = pd.DataFrame(ask_matrix, index=currencies, columns=currencies)
        mid_df = pd.DataFrame(mid_matrix, index=currencies, columns=currencies)

        return bid_df, ask_df, mid_df

    @staticmethod
    def asset_lookup(spread_dfs, asset):
        bid_spread = spread_dfs.bid
        ask_spread = spread_dfs.ask
        mid_spread = spread_dfs.mid
        if '/' in asset:    # Currency lookup
            base, quote = asset.split('/')
            bid = bid_spread.at[quote, base]
            ask = ask_spread.at[quote, base]
            mid = mid_spread.at[quote, base]
        else:
            tic = asset.split('.')[0]
            bid = spread_dfs.bid.at[tic]
            ask = spread_dfs.ask.at[tic]
            mid = spread_dfs.mid.at[tic]
        return bid, ask, mid
