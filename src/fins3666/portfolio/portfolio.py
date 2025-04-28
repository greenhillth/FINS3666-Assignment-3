import pandas as pd
import numpy as np
from typing import Union, List, TypedDict, NotRequired
from datetime import datetime

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
        name (str): 
            The name or symbol of the asset (e.g., "AUD", "USD", "BTC").

        units (Union[int, float]): 
            The number of units in the order. Negative values typically indicate a sale, unless otherwise specified.

        purchase_price (float): 
            The price paid per unit at the time of purchase, in USD.

        timestamp (datetime): 
            The datetime when the asset was purchased.

        forecast_return (float, optional): 
            The expected return on the asset. Negative values imply compounding liability. 
            Defaults to 0.0 (no forecasted return).

        order_type (str, optional): 
            The type of order (e.g., 'market', 'limit'). Defaults to None.

        exchange (str, optional): 
            The exchange where the asset was purchased. Defaults to None.

        asset_type (str, optional): 
            The type of asset (e.g., 'stock', 'bond', 'currency'). Defaults to 'Currency'.

    Example:
        >>> asset = {
        >>>     "name": "BTC",
        >>>     "units": 1.5,
        >>>     "purchase_price": 30000.0,
        >>>     "timestamp": datetime.now(),
        >>>     "forecast_return": 0.07,
        >>>     "order_type": "market",
        >>>     "exchange": "Binance",
        >>>     "asset_type": "Crypto"
        >>> }

        >>> currency_asset = {
        >>>     "name": "USD",
        >>>     "units": 1000,
        >>>     "purchase_price": 1.0,
        >>>     "timestamp": datetime.now()
        >>> }

    Returns:
        None
    """
    name: str
    units: Union[int, float]
    purchase_price: float
    timestamp: datetime
    # Optional attributes
    forecast_return: NotRequired[float] = 0.0
    order_type: NotRequired[str] = None
    exchange: NotRequired[str] = None
    asset_type: NotRequired[str] = 'Currency'


class Portfolio:

    def __init__(self, assets: Union[None, AssetDict, List[AssetDict]]):
        """
        Initialize a Portfolio.

        Args:
            starting_assets (List[AssetDict], optional):
                A list of initial assets to populate the portfolio.

        Example:
            >>> portfolio = Portfolio(starting_assets=[{
            >>>     "timestamp": datetime.now(),
            >>>     "asset": "BTC",
            >>>     "units": 1.5,
            >>>     "unit_price_usd": 30000,
            >>>     "order_type": "buy",
            >>>     "exchange": "Binance",
            >>>     "asset_type": "Crypto"
            >>> }])

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

        # Create Position Ledger with initial assets, if supplied
        self.ledger = self._build_ledger_df(assets)
        # Create transaction record
        self.trades = self._build_trade_df()
        # Create empty market data
        self.mkt = self._build_mkt_df()

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
        Example:
            >>> portfolio.update_ledger(0, datetime.now(), "BTC", 2.0, 30000, 60000, "Crypto")
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

    def portfolio_summary(self):
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
        summary['Position'] = np.where(summary['Units'] >= 0, 'Long', 'Short')
        total_value = summary['CurrentAssetValue'].sum()
        summary['Weight'] = summary['CurrentAssetValue'] / total_value

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

        if startingAssets is not None:
            if isinstance(startingAssets, dict):
                startingAssets = [startingAssets]
            for asset in startingAssets:
                new_row = pd.DataFrame([{
                    'OpenTimestamp': asset['timestamp'], 'UpdateTimestamp': asset['timestamp'],
                    'Asset': asset['name'], 'Units': asset['units'], 'UnitWAP': asset['purchase_price'], 'UnitLastPrice': asset['purchase_price'],
                    'TransactionIdxs': [], 'AssetType': asset.get('asset_type', 'Currency'),
                    'PositionUSD': 0, 'CloseTimestamp': None, 'Open': True
                }])
                df = pd.concat([df, new_row], ignore_index=True)
        return df

    def summary(self):
        """Returns the full portfolio DataFrame."""
        return self.portfolio_summary()

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
        out += f"Net Value ($USD): ${sum(df['CurrentAssetValue']):,.2f}"

        # get sum of
        return out

    def __repr__(self):
        return f"{len(self.assets)} Asset, USD${self.total_value:,.2f} Nominal Value Portfolio Object"

    @staticmethod
    def buildForexMatrix(row, currencies=['AUD', 'CAD', 'CHF', 'EUR', 'GBP', 'JPY', 'NZD', 'SEK', 'SGD']):
        """
        Builds a forex matrix from a row of data containing bid, ask, and mid prices for various currencies.
        Args:
            row (pd.Series): A row of data containing bid, ask, and mid prices for various currencies.
            currencies (list): List of currency codes to include in the matrix. Defaults to G10 list.
        Returns:
            tuple: A tuple containing three DataFrames: bid matrix, ask matrix, and mid matrix.
        """

        # Add USD to list of currencies
        currencies.append('USD')

        # Initialize bid/ask/mid vectors
        bid_vec = np.array([row.get(f"{currency} Bid", 1.0)
                           for currency in currencies])
        ask_vec = np.array([row.get(f"{currency} Ask", 1.0)
                           for currency in currencies])
        mid_vec = np.array([row.get(f"{currency} Mid", 1.0)
                           for currency in currencies])

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
    def assetSaleValue():
        """
        Returns the value of an asset sale.
        Args:
            None
        Returns:
            float: The value of the asset sale.
        """
        return 0.0

    def updateMarketData(self, currentData: Union[dict, List[dict]]):
        """
        Updates the market data for the portfolio.

        Args:
            currentData (Union[dict, List[dict]]): A dictionary or list of dictionaries containing market data.
                Each dictionary should have the following structure, formatted by the `format_market_data` function:
                {
                    "asset": str,          # The name of the asset (e.g., "BTC", "USD").
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

        for data in currentData:
            asset = data["asset"]
            if asset not in self.mkt.index:
                new_data = pd.DataFrame([{
                    "Asset": asset,
                    "Timestamp": data["timestamp"],
                    "Bid": data["bid"],
                    "Ask": data["ask"],
                    "Mid": data["mid"],
                }]).set_index("Asset")
                self.mkt = pd.concat([self.mkt, new_data])
            elif data["timestamp"] > self.mkt.loc[asset, "Timestamp"]:
                self.mkt.loc[asset, "Timestamp"] = data["timestamp"]
                self.mkt.loc[asset, "Bid"] = data["bid"]
                self.mkt.loc[asset, "Ask"] = data["ask"]
                self.mkt.loc[asset, "Mid"] = data["mid"]

    # this function FUCKING SUCKS

    def get_asset_values(src: Union[pd.DataFrame, pd.Series, str], asset: Union[str, List[str]], timestamp=None, atype='C', otype='Mid', ref='USD'):
        """
        Returns the value(s) of an asset or list of assets for a given timestamp.
        Args:
            src (pd.DataFrame, pd.Series or str): The source of the asset data. Can be a full DataFrame (in which a lookup is performed), a Series or a file path (CSV or Excel). Dataframe must contain `Timestamp` column (not index).
            timestamp (datetime, optional): The timestamp for which to retrieve the asset value. Required if `src` is a DataFrame or file path. If `src` is a Series, the timestamp is ignored.
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
