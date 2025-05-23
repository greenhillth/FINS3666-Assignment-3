import pandas as pd
import numpy as np
import uuid
import os
from typing import Union, Optional
from datetime import datetime
from dataclasses import dataclass, field
from collections import namedtuple

current_dir = os.getcwd()

DIR_DATA_RAW = os.path.join(current_dir, 'data/raw')
DIR_DATA_PROCESSED = os.path.join(current_dir, 'data/processed')
DIR_OUT = os.path.join(current_dir, 'out')

ACCOUNT_SIZE_USD = 12e6
NET_POSITION = 0
BORROWING_COST_PA = 0.02
BORROWING_COST_PM = (1+BORROWING_COST_PA) ** (1/12) - 1

AssetSpreads = namedtuple('AssetSpreads', ['bid', 'ask', 'mid'])


def load_dataframe(path):
    """
    Loads a CSV file into a pandas DataFrame and converts the 'Date' column to datetime.
    Args:
        path (str): Relative path to CSV file.
    Returns:
        pd.DataFrame: The loaded DataFrame with 'Period' column converted to datetime.
    """
    df = pd.read_csv(path)
    df['Timestamp'] = pd.to_datetime(df['Date'], dayfirst=True)
    df.sort_values(by='Timestamp', ascending=True, inplace=True)
    df['Period'] = df['Timestamp'].dt.to_period('D')
    df.drop(columns=['Date'], inplace=True)
    df.reset_index(drop=True)
    return df


fx = load_dataframe(os.path.join(DIR_DATA_RAW, 'fx-historical-daily.csv'))
mkt = load_dataframe(os.path.join(DIR_DATA_RAW, 'rates-historical-daily.csv'))
benchmarks = load_dataframe(os.path.join(DIR_DATA_RAW, 'benchmarks.csv'))


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
    order_id: str = field(default_factory=lambda: str(uuid.uuid1()))

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

    def __str__(self) -> str:
        parts = [
            f"Order ID: {self.order_id}",
            f"Asset: {self.asset}",
            f"Units: {self.units}",
            f"Order Type: {self.order_type}",
            f"Order Side: {self.order}",
            f"Timestamp: {self.timestamp.isoformat()}",
            f"Currency: {self.currency}",
            f"Limit Price: {self.limit if not np.isnan(self.limit) else 'N/A'}",
            f"Expiry: {self.expiry or 'N/A'}",
            f"Exchange: {self.exchange or 'N/A'}",
            f"Asset Type: {self.asset_type}",
        ]
        return " | ".join(parts)

    def summary_str(self) -> str:
        limit_part = f" @ {self.limit:.5f}" if not np.isnan(
            self.limit) else " @ MARKET"
        return f"{self.order.upper()} {self.units} {self.asset}{limit_part} (ID: {self.order_id[:8]})"

    def log(self, timestamp: datetime, status: str, price: Optional[float] = None) -> str:
        logmsg = f'{timestamp.isoformat()}: \t Order <{self.order_id}> [{self.summary_str()}] {status.upper()}'
        logmsg += f' @ {price:.5f}{self.currency}' if price is not None else ''
        return logmsg


def index_inflation(value, timestamp):
    cpi = benchmarks.loc[benchmarks['Timestamp']
                         <= timestamp, 'CPI'].tail(1)
    multiplier = cpi.values[0]/100 + 1
    return value*multiplier


def index_benchmark(value, timestamp):
    benchmark = benchmarks.loc[benchmarks['Timestamp']
                               <= timestamp, ['SP500', 'CPI']].tail(1)
    cpi = benchmark['CPI'].values[0]
    indx = benchmark['SP500'].values[0]
    multiplier = cpi if cpi > indx else indx
    return value*(multiplier/100+1)


__all__ = ['pd', 'np', 'os', 'datetime',
           'fx', 'mkt', 'benchmarks',
           'DIR_DATA_RAW', 'DIR_DATA_PROCESSED', 'DIR_OUT', 'ACCOUNT_SIZE_USD',
           'AssetSpreads', 'Order',
           'index_inflation', 'index_benchmark']
