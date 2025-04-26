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

    def __init__(self, assets, sizes=None, positions=None, unit_values=None, timestamp=None):
        """
        Initialize a Portfolio.

        Args:
            assets (list): List of asset identifiers (e.g., ["USD", "GBP", "ASX.CBA"]).
            sizes (list or array, optional): Quantity of each asset held.
            positions (list or array, optional): Position value or reference per asset.
            unit_values (list or array, optional): Value of one unit of each asset.
            weights (list or array, optional): Portfolio weight for each asset. Should sum to 1.
        """
        n = len(assets)
        p = len(positions) if positions is not None else 0
        pos = np.array([-1 if p.tolower.contains('short')
                       else 1 for p in positions]) if p is n else np.ones(n)

        # Asset Characteristics (Assigned)
        self.assets = assets
        sizes = self._adjust_array(sizes, default=0)
        self.positions = self._adjust_array(pos, default=1)
        self.sizes = np.array(
            [-s if p < 0 else s for s, p in zip(sizes, self.positions)])
        self.unit_values = self._adjust_array(unit_values, default=0)

        # Asset Characteristics (Determined)
        self.asset_values = self.sizes * self.unit_values
        total_value = np.sum(self.asset_values)
        self.weights = self.asset_values / \
            total_value if total_value > 0 else np.zeros(len(assets))

        # Portfolio Characteristics (Scalar)
        self.total_value = total_value
        self.timestamp = timestamp
        self.sharpe_ratio = None
        self.sortino_ratio = None
        self.max_drawdown = None
        self.volatility = None
        self.beta = None
        self.alpha = None

        # Create Purchase Ledger
        self.ledger = self._build_ledger_df()

        # Update ledger with initial data, if applicable

        # Generate portfolio from Ledger
        self.portfolio_df = self._build_portfolio_df()

    def update_ledger(self, orders: Union[List[AssetDict], AssetDict]):
        """
        Appends one or more executed orders to the internal ledger. Must be called to update ledger internally only!!

        Args:
            data (Union[AssetDict, List[AssetDict]]): 
                A single asset dictionary or a list of asset dictionaries.
                Each dictionary must contain:
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
        if isinstance(orders, dict):
            orders = [orders]
        for order in orders:
            timestamp = order['timestamp']
            asset = order['name']
            units = order['units']
            unit_price_usd = order['purchase_price']

        order_value_usd = units * unit_price_usd
        position_usd = units * unit_price_usd
        self.purchase_ledger = self.purchase_ledger.append(
            {'Timestamp': timestamp, 'Asset': asset, 'Units': units, 'UnitPriceUSD': unit_price_usd,
             'OrderType': order_type, 'OrderValueUSD': order_value_usd, 'PositionUSD': position_usd},
            ignore_index=True)
        self.purchase_ledger['Timestamp'] = pd.to_datetime(
            self.purchase_ledger['Timestamp'], format='%Y-%m-%d %H:%M:%S')

        # TODO - implement this function to update the portfolio_df with the current values of the assets in the ledger
        self.portfolio_df = self.get_current_portfolio()

    def add_asset(self, asset, size, unit_value, position='long'):
        """
        Add a new asset to the portfolio.

        Args:
            asset (str): Asset identifier.
            unit_value (float): Sale value of one unit of the asset, in USD.
            size (float): Units of asset aquired.
            position (string, optional): Defaults to 'long'. Specify 'short' in the case of a short-sale.
        """

        # get or generate index of asset
        idx = self._get_asset_idx(asset)

        if position.tolower.contains('short'):
            size = -size

        self.sizes[idx] += size
        self.unit_values[idx] = unit_value

        self._update()

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

    def _get_asset_idx(self, asset):
        """
        Get the index of an asset in the portfolio, or add one if it doesn't exist.

        Args:
            asset (str): Asset identifier.

        Returns:
            int: Index of the asset in the portfolio.
        """
        if asset not in self.assets:
            self.assets.append(asset)
            self.sizes = np.append(self.sizes, 0)
            self.positions = np.append(self.positions, 1)
            self.unit_values = np.append(self.unit_values, 0)
            self.weights = np.append(self.weights, np.nan)
        return self.assets.index(asset)

    def _update(self):
        """
        Update derived portfolio characteristics.

        Args:
            None
        """
        self.positions = np.where(self.sizes >= 0, 1, -1)
        self.unit_values *= self.positions
        self.asset_values = np.sum(self.sizes * self.unit_values)
        self.total_value = np.sum(self.asset_values)
        self.weights = self.asset_values / \
            self.total_value if np.fabs(
                self.total_value) > 1e-6 else np.zeros(len(self.assets))
        self.data = self._build_dataframe()

    def _adjust_array(self, arr, default=0):
        """Ensure an array is the same length as assets, filling or truncating as needed."""
        n = len(self.assets)
        if arr is None:
            return np.full(n, default)
        arr = np.asarray(arr)
        if arr.size < n:
            return np.concatenate([arr, np.full(n - arr.size, default)])
        return arr[:n]

    def _build_portfolio_df(self):
        """Builds the internal pandas DataFrame representing the portfolio."""
        df = pd.DataFrame(index=self.assets)
        df["Size"] = self.sizes
        df["UnitValue"] = self.unit_values
        df["AssetValue"] = self.asset_values
        df["Position"] = ['Short' if s < 0 else 'Long' for s in self.sizes]
        df["Weight"] = self.weights
        return df

    def _build_ledger_df(self):
        """Builds the internal pandas DataFrame representing the purchase ledger."""

        df = pd.DataFrame({
            'Timestamp': pd.Series(dtype='pd.datetime64[ns]'),
            'Asset': pd.Series(dtype='str'),
            'Units': pd.Series(dtype='float64'),
            'UnitPriceUSD': pd.Series(dtype='float64'),
            'OrderValueUSD': pd.Series(dtype='float64'),
            'CurrentReturn': pd.Series(dtype='float64'),
            'ForecastReturn': pd.Series(dtype='float64'),
            'i_USD': pd.Series(dtype='float64'),
            'OrderType': pd.Series(dtype='str'),
            'Exchange': pd.Series(dtype='str'),
            'AssetType': pd.Series(dtype='str'),
            'PositionUSD': pd.Series(dtype='float64')})

        return df

    def summary(self):
        """Returns the full portfolio DataFrame."""
        return self.data

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
