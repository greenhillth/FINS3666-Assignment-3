import pandas as pd
import datetime

"""
functions.py

This module contains utility functions to assist with common tasks in the project.
"""


def format_market_data(data: pd.Series) -> list[dict]:
    """
    Formats market data from a pandas Series into a dictionary.

    Args:
        data (pd.Series): The market data Series.

    Returns:
        dict: A dictionary with formatted market data.
    """
    if isinstance(data, pd.DataFrame):
        data = data.iloc[-1]

    def extract_assets(headers: list[str]) -> list[str]:
        asset_set = set()
        asset_set = {header.split(' ', 1)[0]
                     for header in headers if ' ' in header}

        return list(asset_set)

    assetList = extract_assets(data.index.tolist())

    return [{
        'timestamp': data['Timestamp'],
        'asset': a.upper(),
        'tic': str("USD/") + a.upper(),
        'bid': data[f"{a.upper()} Bid"],
        'ask': data[f"{a.upper()} Ask"],
        'mid': data[f"{a.upper()} Mid"]
    } for a in assetList]


def current_fx_data(fx: pd.DataFrame, timestamp: datetime):
    """
    Gets the current FX data for a given timestamp.
    Args:
        fx (pd.DataFrame): DataFrame containing FX data.
        timestamp (str): The timestamp for the FX data.
    Returns:
        list[dict]: List of dictionaries with formatted FX data.
    """
    hist = fx[fx['Timestamp'] <= timestamp]
    if hist.empty:
        raise ValueError(
            f"Unable to locate sufficiently accurate FX data for time {timestamp}.")
    # format the corresponding row
    return format_market_data(hist.iloc[-1, :])


def getExchangeRate(df, termCurrency, baseCurrency='USD'):
    """
    Gets the exchange rate between two currencies from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing exchange rates.
        termCurrency (str): The term currency.
        baseCurrency (str, optional): The base currency. Defaults to 'USD'.

    Returns:
        float: The exchange rate.
    """
    if baseCurrency == termCurrency:
        return 1.0
    invert = False
    # if dataframe contains multiple entries, get the most recent quote
    if isinstance(df, pd.DataFrame):
        df = df.iloc[-1]
    # check if the column exists in the series
    col = f'{baseCurrency}/{termCurrency}'
    if col not in df.keys():
        col = f'{termCurrency}/{baseCurrency}'
        invert = True
    if col not in df.keys():
        raise ValueError(
            f"Exchange rate for {baseCurrency} to {termCurrency} not found in DataFrame.")
    else:
        return df[col] if not invert else df[col]**(-1)


def currencyUSDvals(df, timestamp, currencies):
    """
    Converts currency values to USD using exchange rates from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing exchange rates.
        timestamp (str): The timestamp for the exchange rate.
        currencies (list): List of currencies to convert.

    Returns:
        dict: Dictionary with currency values in USD.
    """
    period = pd.Period(timestamp, freq='M')
    # round period down to the closest index present in the DataFrame
    period = df.index[df.index <= period].max()
    if period is pd.NaT:
        raise ValueError(
            f"Unable to locate sufficiently accurate FX data for time {timestamp.to_string()}.")

    df = df.loc[period]
    return [getExchangeRate(df, baseCurrency=c, termCurrency='USD') for c in currencies]


def format_currency(value):
    """
    Formats a number as a currency string.

    Args:
        value (float): The numeric value to format.

    Returns:
        str: The formatted currency string.
    """
    return "${:,.2f}".format(value)


__all__ = ["format_market_data", "current_fx_data",
           "getExchangeRate", "currencyUSDvals", "format_currency"]
