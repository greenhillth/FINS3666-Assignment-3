import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

current_dir = os.getcwd()

DIR_DATA_RAW = os.path.join(current_dir, 'data/raw')
DIR_DATA_PROCESSED = os.path.join(current_dir, 'data/processed')
DIR_OUT = os.path.join(current_dir, 'out')

load_dotenv()

FS_API_USER = os.getenv('API_USER')
FS_API_KEY = os.getenv('API_KEY')

ACCOUNT_SIZE_USD = 12e6
NET_POSITION = 0
BORROWING_COST_PA = 0.02
BORROWING_COST_PM = (1+BORROWING_COST_PA) ** (1/12) - 1


__all__ = ["pd", "np", "os", "DIR_DATA_RAW", "DIR_DATA_PROCESSED", "DIR_OUT",
           "ACCOUNT_SIZE_USD", "NET_POSITION", "BORROWING_COST_PA", "BORROWING_COST_PM"]
