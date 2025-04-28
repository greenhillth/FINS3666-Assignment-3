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


__all__ = ['pd', 'np', 'os', 'datetime',
           'DIR_DATA_RAW', 'DIR_DATA_PROCESSED', 'DIR_OUT', 'ACCOUNT_SIZE_USD',
           'AssetSpreads', 'Order']
