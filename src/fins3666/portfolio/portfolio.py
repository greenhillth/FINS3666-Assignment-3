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

        self.data = self._build_dataframe()

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

    def _build_dataframe(self):
        """Builds the internal pandas DataFrame representing the portfolio."""
        df = pd.DataFrame(index=self.assets)
        df["Size"] = self.sizes
        df["UnitValue"] = self.unit_values
        df["AssetValue"] = self.asset_values
        df["Position"] = ['Short' if s < 0 else 'Long' for s in self.sizes]
        df["Weight"] = self.weights
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
