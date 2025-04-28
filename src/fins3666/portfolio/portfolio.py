import pandas as pd
import numpy as np
import uuid
from typing import Union, List, Optional, Literal
from datetime import datetime
from dataclasses import dataclass, field
from collections import namedtuple

AssetSpreads = namedtuple('AssetSpreads', ['bid', 'ask', 'mid'])


@dataclass(frozen=True)
class Order:

    asset: str
    units: Union[int, float]
    order: str
    timestamp: datetime

    currency: Optional[str] = 'USD'
    order_type: Optional[str] = 'market'
    expiry: Optional[str] = None
    limit: Optional[float] = np.nan
    exchange: Optional[str] = None
    asset_type: Optional[str] = "Currency"
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self):
        if self.units <= 0:
            raise ValueError(
                "Units must be greater than zero. Specify order='sell' for sale")
        if self.limit <= 0:
            raise ValueError("Purchase price must be positive.")
        if not isinstance(self.timestamp, datetime):
            raise TypeError("Timestamp must be a datetime object.")
        if self.order_type == 'limit' and np.isnan(self.limit):
            raise ValueError("No limit specified for limit order")

    def tic(self):
        return f'{self.asset}/{self.currency}' if self.asset_type == "Currency" else f'{self.asset}.{self.exchange}'

    def sym(self):
        syms = {'CHF': 'SFr.', 'EUR': '€', 'GBP': '£', 'JPY': '¥', 'SEK': 'kr'}
        return syms.get(self.currency, '$')


class Portfolio:

    def __init__(self, timestamp: datetime, balance: Union[dict, List[dict]]):
        """
        Initialize a Portfolio.

        Args:
            starting_balance (List[AssetDict], optional):
                A list of initial assets to populate the portfolio.

        Example:
            >>> portfolio = Portfolio(timestamp=datetime(2020,09,1), starting_balance=[{
            >>>     "asset": "USD",
            >>>     "units": 12000,
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
        self.ledger = self._build_ledger_df(timestamp, balance)
        # Create transaction record
        self.trades = self._build_trade_df()
        # Create empty market data
        self.mkt = self._build_mkt_df()
        self.forex_spreads = AssetSpreads(None, None, None)

        self.orders = []

    def new_order(self, order: Order):
        self.orders.append(order)

    def update(self, timestamp: datetime):
        self.process_orders(timestamp)

    def process_orders(self, timestamp: datetime):
        cancelled = []
        executed = []
        remaining = []
        for order in self.orders:
            bid, ask, mid = self.asset_lookup(self.forex_spreads, order.tic())
            if order.expiry is not None and order.expiry < timestamp:
                cancelled.append(order)
            elif order.order_type == 'market':
                if order.order == 'buy':
                    self.execute_trade(order, ask)
                elif order.order == 'sell':
                    self.execute_trade(order, bid)
                executed.append(order)
            elif order.order_type == 'limit':
                if order.order == 'buy' and ask <= order.limit:
                    self.execute_trade(order, ask)
                elif order.order == 'sell' and bid >= order.limit:
                    self.execute_trade(order, bid)
                executed.append(order)
            else:
                remaining.append(order)

        self.orders = remaining

    def execute_trade(self, order, price):
        msg = str()
        if order.order == 'buy':
            msg += (
                f'Executed {order.order_type} buy order of {order.units} units of {order.asset}'
                f' for {order.currency}{order.sym()}{price:,.4f} per unit.'
                f' Transaction cost: {order.sym()}{(order.units*price):,.2f}'
            )
        else:
            msg += (
                f'Executed {order.order_type} {order.order} order of {order.units} units of {order.asset}'
                f' for {order.currency}{order.sym()}{price:,.4f} per unit.'
                f' Transaction gain: {order.sym()}{(order.units*price):,.2f}'
            )
        print(msg)

    def updateMarketData(self, currentData: Union[dict, List[dict]]):
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

    def summary(self):
        """
        Builds a portfolio snapshot by getting current market values of assets from the ledger.

        Returns:
            pd.DataFrame: DataFrame containing the current portfolio values.
        """

        # Filter out closed positions
        summary = self.ledger[self.ledger['Open']].copy()
        if summary.empty:
            return pd.DataFrame()

        # Merge ledger with latest market data for each asset
        summary = summary.merge(
            self.mkt,
            left_on='Asset',
            right_index=True,
            how='left')

        summary['PurchaseCost'] = -summary['Units'] * summary['UnitWAP']

        summary['CurrentUnitVal'] = summary['Mid']
        summary['CurrentAssetValue'] = summary['CurrentUnitVal'] * \
            summary['Units']

        summary['Size'] = summary['Units'].abs()
        summary['Position'] = np.select([summary['Units'] > 0, summary['Units'] < 0],
                                        ['Long', 'Short'], default='None')
        total_value = summary['CurrentAssetValue'].sum(skipna=True)
        summary['Weight'] = np.where(
            total_value != 0, summary['CurrentAssetValue'] / total_value, 0)

        # Sort Columns
        columns_to_keep = [
            'Asset', 'Units', 'UnitWAP', 'PurchaseCost', 'CurrentUnitVal',
            'CurrentAssetValue', 'Size', 'Position', 'Weight', 'AssetType',
            'OpenTimestamp', 'UpdateTimestamp'
        ]
        summary = summary[columns_to_keep]

        return summary

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

    def _build_ledger_df(self, timestamp, startingBal: Union[None, dict, List[dict]]):
        """
        Builds the internal pandas DataFrame representing the ledger containing all open and closed positions.
        Args:
            startingAssets (list of AssetDict, optional):
                A single asset dictionary or a list of asset dictionaries containing:
                    - 'name' (str): asset name or identifier
                    - 'units' (np.float64): quantity of asset held
                    - 'purchase_price' (np.float64): price per unit
                    - 'timestamp' (datetime): time of asset acquisition
        Returns:
            pd.DataFrame: DataFrame representing the ledger.
        """
        if startingBal is not None:
            if isinstance(startingBal, dict):
                startingBal = [startingBal]
            df = pd.DataFrame([{
                'OpenTimestamp': timestamp,
                'UpdateTimestamp': timestamp,
                'Asset': d['asset'],
                'Units': d['units'],
                'UnitWAP': d['unit_value_USD'],
                'UnitLastPrice': d['unit_value_USD'],
                'TransactionIdxs': [],
                'AssetType': 'Currency',
                'PositionUSD': d['units']*d['unit_value_USD'],
                'CloseTimestamp': None,
                'Open': True}
                for d in startingBal])
        else:
            df = pd.DataFrame({
                'OpenTimestamp': pd.Series(dtype='datetime64[ns]'),
                'UpdateTimestamp': pd.Series(dtype='datetime64[ns]'),
                'Asset': pd.Series(dtype='str'),
                'Units': pd.Series(dtype='float64'),
                'UnitWAP': pd.Series(dtype='float64'),
                'UnitLastPrice': pd.Series(dtype='float64'),
                'TransactionIdxs': pd.Series(dtype='object'),
                'AssetType': pd.Series(dtype='str'),
                'PositionUSD': pd.Series(dtype='float64'),
                'CloseTimestamp': pd.Series(dtype='datetime64[ns]'),
                'Open': pd.Series(dtype='bool')})

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

    def _update_ledger(self, transaction_id, timestamp, asset, units, unit_price_usd, order_value_usd, asset_type):
        """
        Update the position ledger with details of trade. Should only be called by the add_trade method.
        Args:
            transaction_id (int): Unique identifier for the transaction.
            timestamp (datetime): Time of the transaction.
            asset (str): Asset identifier.
            units (float): Number of units acquired or sold.
            unit_price_usd (float): Price per unit in USD.
            order_value_usd (float): Total value of the order in USD.
            asset_type (str): Type of asset (e.g., 'stock', 'bond', etc.).
        Example:
            >>> portfolio._update_ledger(0, datetime.now(), "BTC", 2.0, 30000, 60000, "Crypto")
        Returns:
            None
        """
        '''
        {
                'OpenTimestamp': timestamp,
                'UpdateTimestamp': timestamp,
                'Asset': d['asset'],
                'Units': d['units'],
                'UnitWAP': d['unit_value_USD'],
                'UnitLastPrice': d['unit_value_USD'],
                'TransactionIdxs': [],
                'AssetType': 'Currency',
                'PositionUSD': d['units']*d['unit_value_USD'],
                'CloseTimestamp': None,
                'Open': True}

        '''

        # Check if open position for asset already exists
        existing_asset = self.ledger[self.ledger['Asset'] == asset]
        if not existing_asset.empty and existing_asset['Open'].values[0]:
            # Update existing asset entry
            idx = existing_asset.index[0]

            existingUnits = self.ledger.loc[idx, 'Units']
            newUnits = existingUnits + units
            newUnits = 0 if (abs(newUnits) < 1e-6) else newUnits
            self.ledger.at[idx, 'UpdateTimestamp'] = timestamp

            self.ledger.at[idx, 'Units'] = newUnits
            self.ledger.at[idx, 'UnitWAP'] = (
                self.ledger.loc[idx, 'UnitWAP'] * existingUnits + order_value_usd)/newUnits
            self.ledger.at[idx, 'UnitLastPrice'] = unit_price_usd
            self.ledger.at[idx, 'TransactionIdxs'].append(transaction_id)
            self.ledger.at[idx, 'AssetType'] = asset_type
            if newUnits == 0:  # maybe implement a close position function instead??
                self.ledger.at[idx, 'CloseTimestamp'] = timestamp
                self.ledger.at[idx, 'Open'] = False
        else:
            self.ledger = self.ledger.append(
                {'OpenTimestamp': timestamp, 'UpdateTimestamp': timestamp,
                 'Asset': asset, 'Units': units, 'UnitWAP': unit_price_usd, 'UnitLastPrice': unit_price_usd, 'TransactionIdxs': [transaction_id],
                 'AssetType': asset_type, 'CloseTimestamp': None, 'Open': True},
                ignore_index=True)

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
        asset = order.asset
        typ = order.order
        units = order.units
        unit_price = price
        currency = order.curr
        order_value = units * unit_price
        bal = -order_value if typ == 'buy' else order_value
        forecast_return = 0.0
        i_usd = 0.0
        order_type = order.order_type
        exchange = order.exchange
        asset_type = order.get('asset_type', 'Currency')

        trade = pd.DataFrame([{'Timestamp': timestamp, 'Asset': asset, 'Buy/Sell': typ, 'Units': units, 'UnitPrice': unit_price, 'Balance': bal, 'BaseCurrency': currency,
                               'OrderValue': order_value, 'ForecastReturn': forecast_return, 'i_USD': i_usd,
                               'OrderType': order_type, 'Exchange': exchange, 'AssetType': asset_type}], index=[id])

        self.trades = pd.concat([self.trades, trade], ignore_index=True)
        self._update_ledger(order)

    """Namespace methods for Portfolio class"""

    def __str__(self):
        df = self.summary()
        out = f"\n\n{str('Portfolio Overview'):^80}" + "\n"
        out += "=" * 80 + "\n"
        out += f"{'Asset':<10}{'Units':>9}{'Unit Value':>18}{'Asset Value':>15}{'Position':>15}{'Weight':>10}\n"
        out += "=" * 80 + "\n"
        for asset, row in df.iterrows():
            out += (
                f"{asset:<10}"
                f"{row['Size']:>10.2f}"
                f"{f'${row['UnitWAP']:,.2f}':>15}"
                f"{f'${row['CurrentAssetValue']:,.2f}':>15}"
                f"{str(row['Position']):>15}"
                f"{row['Weight']:>10.2%}\n"
            )
        out += f"\nPortfolio - {len(df)} assets\n"
        out += f"Unit Price Timestamp: {df['UpdateTimestamp'].max()},\n"
        out += f"Net Value ($USD): ${df['CurrentAssetValue'].sum(skipna=True):,.2f}"

        # get sum of
        return out

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
            currencies (list): List of currency codes to include in the matrix. Defaults to G10 list.
        Returns:
            tuple: A tuple containing three DataFrames: bid matrix, ask matrix, and mid matrix.
        """
        currencies = df.loc[:, 'tic']
        bid_vec = df.loc[:, 'bid'].to_numpy()
        ask_vec = df.loc[:, 'ask'].to_numpy()
        mid_vec = df.loc[:, 'mid'].to_numpy()

        # # Initialize bid/ask/mid vectors
        # bid_vec = np.array([pow.get(f"{currency} Bid", 1.0)
        #                    for currency in currencies])
        # ask_vec = np.array([row.get(f"{currency} Ask", 1.0)
        #                    for currency in currencies])
        # mid_vec = np.array([row.get(f"{currency} Mid", 1.0)
        #                    for currency in currencies])

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
