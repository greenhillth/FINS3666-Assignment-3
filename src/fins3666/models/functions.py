import pandas as pd
import numpy as np
from fins3666.defines import Order


def execute_strategy(strategy, data):
    """
    Executes a given strategy on the provided data.

    Args:
        strategy (callable): The strategy function to execute.
        data (pd.DataFrame): The data to apply the strategy on.

    Returns:
        pd.DataFrame: The results of the strategy execution.
    """
    return strategy(data)


def carry_trade(data):
    """
    Example strategy function for a carry trade.

    Args:
        data (pd.DataFrame): The data to apply the strategy on.

    Returns:
        pd.DataFrame: The results of the carry trade strategy.
    """
    # Implement the carry trade logic here
    pass


def run(strategy, data):
    """
    Runs a given strategy on the provided data.

    Args:
        strategy (callable): The strategy function to execute.
        data (pd.DataFrame): The data to apply the strategy on.

    Returns:
        pd.DataFrame: The results of the strategy execution.
    """
    return execute_strategy(strategy, data)


__all__ = ["execute_strategy"]
