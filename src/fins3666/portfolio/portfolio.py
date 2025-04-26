import pandas as pd
import numpy as np


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
        self.assets = assets
        self.sizes = self._adjust_array(sizes, default=0)
        self.positions = self._adjust_array(positions, default=0)
        self.unit_values = self._adjust_array(unit_values, default=0)
        self.asset_values = np.sum(self.sizes * self.unit_values)
        self.total_value = np.sum(self.asset_value)
        self.weights = self.asset_values / \
            self.total_value if self.total_value > 0 else np.zeros(len(assets))
        self.data = self._build_dataframe()

        # Portfolio Characteristics
        self.timestamp = timestamp
        self.sharpe_ratio = None
        self.sortino_ratio = None
        self.max_drawdown = None
        self.volatility = None
        self.beta = None
        self.alpha = None

    """Private methods for Portfolio class"""

    def _update(self, sizes=None, positions=None, unit_values=None):
        """
        Update the portfolio with new values.

        Args:
            sizes (list or array, optional): New sizes for each asset.
            positions (list or array, optional): New positions for each asset.
            unit_values (list or array, optional): New unit values for each asset.
            weights (list or array, optional): New weights for each asset.
        """
        self.sizes = self._adjust_array(sizes, default=self.sizes)
        self.positions = self._adjust_array(positions, default=self.positions)
        self.unit_values = self._adjust_array(
            unit_values, default=self.unit_values)

    def _adjust_array(self, arr, default=0):
        """Ensure an array is the same length as assets, filling or truncating as needed."""
        n = len(self.assets)
        if arr is None:
            return np.full(n, default)
        arr = np.asarray(arr)
        if arr.size < n:
            return np.concatenate([arr, np.full(n - arr.size, default)])
        return arr[:n]

    def _build_dataframe(self):
        """Builds the internal pandas DataFrame representing the portfolio."""
        df = pd.DataFrame(index=self.assets)
        df["Size"] = self.sizes
        df["UnitValue"] = self.unit_values
        df["AssetValue"] = self.sizes * self.unit_values
        df["Position"] = self.positions
        df["Weight"] = self.weights
        return df

    def summary(self):
        """Returns the full portfolio DataFrame."""
        return self.data

    def __repr__(self):
        return self.data.to_string() + "\n" + f"Portfolio - ({len(self.assets)} assets)"
