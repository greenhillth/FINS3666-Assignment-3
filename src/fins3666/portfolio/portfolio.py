import pandas as pd
import numpy as np
from typing import Union, List, TypedDict, NotRequired

"""
CURRENTLY REFACTORING THE PORTFOLIO TO RELY ON A TRANSACTION LEDGER.
THIS WILL ENABLE BETTER TRACKING OF ASSET VALUES AND TRANSACTIONS.
WORK IN PROGRESS.

I AM A COMPLETE DISASTER.
I AM A MONUMENTAL FAILURE.
I AM A GLORIFIED PAPERWEIGHT.
I AM A NEVER-ENDING SOURCE OF REGRET.
"""


class AssetDict(TypedDict):
    """
    Represents a single asset entry for the portfolio.

    Attributes:
        name (str): The name or symbol of the asset (e.g., "AUD", "USD", ).
        units (np.float64): The number of units in the order. Negative values indicate a sale, unless otherwise specified in the `sales`.
        purchase_price (np.float64): The price paid per unit at time of purchase, in USD.
        timestamp (datetime): The datetime when the asset was purchased.
        forecast_return (np.float64, optional): The expected return on the asset. Negative results in compounding liability. Defaults to 0%.
        order_type (str, optional): The type of order ('market', 'limit', etc.). Defaults to None.
        exchange (str, optional): The exchange where the asset was purchased. Defaults to None.
        asset_type (str, optional): The type of asset (e.g., 'stock', 'bond', etc.). Defaults to Currency.

    """
    name: str
    units: np.float64
    purchase_price: np.float64
    timestamp: pd.datetime
    # Optional attributes
    forecast_return: NotRequired[np.float64] = 0.0
    order_type: NotRequired[str] = None
    exchange: NotRequired[str] = None
    asset_type: NotRequired[str] = 'Currency'


class Portfolio:

    def __init__(self, startingAssets: Union[None, AssetDict, List[AssetDict]]):
        """
        Initialize a Portfolio.

        Args:
            statingAssets (list of AssetDict, optional):
                A single asset dictionary or a list of asset dictionaries containing:
                    - 'name' (str): asset name or identifier
                    - 'units' (np.float64): quantity of asset held
                    - 'purchase_price' (np.float64): price per unit
                    - 'timestamp' (datetime): time of asset acquisition
        """

        # Portfolio Characteristics (Scalar)
        self.sharpe_ratio = None
        self.sortino_ratio = None
        self.max_drawdown = None
        self.volatility = None
        self.beta = None
        self.alpha = None

        # Create Position Ledger with initial assets, if supplied
        self.ledger = self._build_ledger_df(startingAssets)
        # Create transaction record
        self.trades = self._build_trade_df()

    def _build_trade_df(self):
        """Builds the internal pandas DataFrame representing the trades."""
        df = pd.DataFrame({
            'Timestamp': pd.Series(dtype='pd.datetime64[ns]'),
            'Asset': pd.Series(dtype='str'),
            'Units': pd.Series(dtype='float64'),
            'UnitPriceUSD': pd.Series(dtype='float64'),
            'OrderValueUSD': pd.Series(dtype='float64'),
            'ForecastReturn': pd.Series(dtype='float64'),
            'i_USD': pd.Series(dtype='float64'),
            'OrderType': pd.Series(dtype='str'),
            'Exchange': pd.Series(dtype='str'),
            'AssetType': pd.Series(dtype='str')})
        return df

    def add_trade(self, order: AssetDict):
        """
        Appends an executed order to the trade record and updates internal ledger.

        Args:
            order (AssetDict):
                A single asset dictionary containing:
                    - 'name' (str): asset name or identifier
                    - 'units' (np.float64): quantity of asset held
                    - 'purchase_price' (np.float64): price per unit
                    - 'timestamp' (datetime): time of asset acquisition

        Example:
            >>> asset = {"name": "BTC", "units": np.float64(2), "purchase_price": np.float64(30000), "timestamp": datetime.now()}
            >>> my_function(asset)

        Returns:
            None
        """
        timestamp = order.get('timestamp', None)
        asset = order['name']
        units = order['units']
        unit_price_usd = order['purchase_price']
        order_value_usd = units * unit_price_usd
        forecast_return = order.get('forecast_return', 0)
        i_usd = order.get('i_USD', 0)
        order_type = order.get('order_type', 'Market')
        exchange = order.get('exchange', None)
        asset_type = order.get('asset_type', 'Currency')

        if units == 0:
            raise ValueError("Invalid trade data: Units cannot be zero.")
        if timestamp is None:
            raise ValueError("Invalid trade data: Timestamp cannot be None.")

        self.trades = self.trades.append(
            {'Timestamp': timestamp, 'Asset': asset, 'Units': units, 'UnitPriceUSD': unit_price_usd,
             'OrderValueUSD': order_value_usd, 'ForecastReturn': forecast_return, 'i_USD': i_usd,
             'OrderType': order_type, 'Exchange': exchange, 'AssetType': asset_type},
            ignore_index=True)
        self.update_ledger(order)

    def update_ledger(self, transaction_id, timestamp, asset, units, unit_price_usd, order_value_usd, asset_type):
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
        Returns:
            None
        """
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
            self.ledger.at[idx, 'TransactionIdxs'].append(transaction_id)
            self.ledger.at[idx, 'AssetType'] = asset_type
            if newUnits == 0:  # maybe implement a close position function instead??
                self.ledger.at[idx, 'CloseTimestamp'] = timestamp
                self.ledger.at[idx, 'Open'] = False
        else:
            self.ledger = self.ledger.append(
                {'OpenTimestamp': timestamp, 'UpdateTimestamp': timestamp,
                 'Asset': asset, 'Units': units, 'UnitWAP': unit_price_usd, 'TransactionIdxs': [transaction_id],
                 'AssetType': asset_type, 'CloseTimestamp': None, 'Open': True},
                ignore_index=True)

    def get_current_portfolio(self, timestamp=None):
        """
        Builds portfolio by getting current values of assets in the ledger.
        Args:
            Timestamp (datetime, optional): The timestamp for to base the asset values on. If not provided, asset data will be assumed current.
        Returns:
            pd.DataFrame: DataFrame containing the current portfolio values.
        """
        # Filter out closed positions
        df = self.ledger[self.ledger['Open'] == True]
        open_positions = df.groupby('Asset').agg(
            {'Units': 'sum', 'UnitWAP': 'mean'}).reset_index()
        open_positions['PurchaseCost'] = -open_positions['Units'] * \
            open_positions['UnitWAP']
        open_positions['AssetValue'] =
        open_positions['Position'] = np.where(
            open_positions['Units'] >= 0, 'Long', 'Short')
        open_positions['Weight'] = open_positions['AssetValue'] / \
            np.sum(open_positions['AssetValue'])
        return open_positions

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

    def _build_ledger_df(self, startingAssets: Union[None, AssetDict, List[AssetDict]]):
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

        df = pd.DataFrame({
            'OpenTimestamp': pd.Series(dtype='pd.datetime64[ns]'),
            'UpdateTimestamp': pd.Series(dtype='pd.datetime64[ns]'),
            'Asset': pd.Series(dtype='str'),
            'Units': pd.Series(dtype='float64'),
            'UnitWAP': pd.Series(dtype='float64'),
            'TransactionIdxs': pd.Series(dtype='list'),
            'AssetType': pd.Series(dtype='str'),
            'PositionUSD': pd.Series(dtype='float64'),
            'CloseTimestamp': pd.Series(dtype='pd.datetime64[ns]'),
            'Open': pd.Series(dtype='Bool')})

        if startingAssets is not None:
            if isinstance(startingAssets, dict):
                startingAssets = [startingAssets]
            for asset in startingAssets:
                df = df.append(
                    {'OpenTimestamp': asset['timestamp'], 'UpdateTimestamp': asset['timestamp'],
                     'Asset': asset['name'], 'Units': asset['units'], 'UnitWAP': asset['purchase_price'],
                     'TransactionIdxs': [], 'AssetType': asset.get('asset_type', 'Currency'),
                     'PositionUSD': 0, 'CloseTimestamp': None, 'Open': True},
                    ignore_index=True)
        return df

    def summary(self):
        """Returns the full portfolio DataFrame."""
        return self.get_current_portfolio()

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
                f"{f'${row['UnitValue']:,.2f}':>15}"
                f"{f'${row['AssetValue']:,.2f}':>15}"
                f"{str(row['Position']):>15}"
                f"{row['Weight']:>10.2%}\n"
            )
        out += f"\nPortfolio - {len(self.assets)} assets\n"
        out += f"Unit Price Timestamp: {self.timestamp},\n"
        out += f"Net Value ($USD): ${self.total_value:,.2f}"
        return out

    def __repr__(self):
        return f"{len(self.assets)} Asset, USD${self.total_value:,.2f} Nominal Value Portfolio Object"

    @staticmethod
    def buildForexMatrix(row, currencies=['AUD', 'CAD', 'CHF', 'EUR', 'GBP', 'JPY', 'NZD', 'SEK', 'SGD']):
        """
        Builds a forex matrix from a row of data containing bid, ask, and mid prices for various currencies.
        Args:
            row (pd.Series): A row of data containing bid, ask, and mid prices for various currencies.
            currencies (list): List of currency codes to include in the matrix. Defaults to a predefined list.
        Returns:
            tuple: A tuple containing three DataFrames: bid matrix, ask matrix, and mid matrix.
        """

        # Add USD to list of currencies
        currencies.append('USD')

        # Initialize bid/ask/mid series
        bids = {currency: row.get(f"{currency} Bid", 1.0)
                for currency in currencies}
        asks = {currency: row.get(f"{currency} Ask", 1.0)
                for currency in currencies}
        mids = {currency: row.get(f"{currency} Mid", 1.0)
                for currency in currencies}

        bids_series = pd.Series(bids)
        asks_series = pd.Series(asks)
        mids_series = pd.Series(mids)

        # Vectorized outer division
        bid_matrix = bids_series[:, None] / asks_series[None, :]
        ask_matrix = asks_series[:, None] / bids_series[None, :]
        mid_matrix = mids_series[:, None] / mids_series[None, :]

        # Wrap in DataFrames with labels
        bid_df = pd.DataFrame(bid_matrix, index=currencies, columns=currencies)
        ask_df = pd.DataFrame(ask_matrix, index=currencies, columns=currencies)
        mid_df = pd.DataFrame(mid_matrix, index=currencies, columns=currencies)

        return bid_df, ask_df, mid_df

    @staticmethod
    def get_asset_values(src: Union[pd.DataFrame, pd.Series, str], timestamp, asset: Union[str, List[str]], atype='C', otype='Mid', ref='USD'):
        """
        Returns the value(s) of an asset or list of assets for a given timestamp.
        Args:
            src (pd.DataFrame, pd.Series or str): The source of the asset data. Can be a full DataFrame (in which a lookup is performed), a Series or a file path (CSV or Excel). Dataframe must contain `Timestamp` column (not index).
            timestamp (datetime): The timestamp for which to retrieve the asset value.
            asset (str or list of str): The name(s) of the asset(s) to retrieve values for.
            atype (str): The type of asset. Defaults to 'C' (Currency). Other types include Bonds('B'), Stocks('S') and Commodities ('O'). Used to determine lookup key format.
            otype (str): The order price used to determine valuation. Defaults to 'Mid'. Other valid options are 'Bid' and 'Ask'.
            ref (str): The reference currency. Defaults to 'USD'.

        Returns:
            list: A list of asset values for the given timestamp, expressed in terms of the optionally specified `ref` param. Unknown assets return 0.
        """
        if isinstance(src, str):
            if src.endswith('.csv'):
                df = pd.read_csv(src)
            elif src.endswith('.xlsx'):
                df = pd.read_excel(src)
            elif src.toupper() == 'FACTSET':
                raise ValueError(
                    "Factset data source not implemented yet. :-(. Please provide a .csv or .xlsx file.")
            else:
                raise ValueError(
                    "Unsupported file format. Please provide a .csv or .xlsx file.")
        else:
            df = src
        # Validation on dataframe
        if isinstance(df, pd.DataFrame):
            if 'Timestamp' not in df.columns:
                raise ValueError(
                    "DataFrame must contain a 'Timestamp' column.")

            # Ensure the 'Timestamp' column is in datetime format
            if not pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
                df['Timestamp'] = pd.to_datetime(
                    df['Timestamp'], errors='coerce')

            df = df.sort_values(by='Timestamp')
            # filter out values occuring BEFORE the given timestamp and get the top row
            df = df[df['Timestamp'] >= timestamp]
            df = pd.Series(df.iloc[0:1])

        # at this point, df should be a Series

        if isinstance(asset, str):
            asset = [asset]
        # Get the value of each asset

        # generate bid, ask and mid matricies
        bid, _, mid = Portfolio.buildForexMatrix(df)

        if atype == 'C':
            if otype == 'Mid':
                return [mid.loc[ref, a] for a in asset]
            else:
                return [bid.loc[ref, a] for a in asset]

        # for other asset types, lookup the value in the dataframe
        return [df.get(f"{a} {otype}", 0) for a in asset]
