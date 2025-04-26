import pandas as pd
import numpy as np

"""
helpers.py

This module contains utility functions to assist with common tasks in the project.
"""


def load_dataframe(path):
    """
    Loads a CSV file into a pandas DataFrame and converts the 'Period' column to datetime.
    Args:
        path (str): Relative path to CSV file.
    Returns:
        pd.DataFrame: The loaded DataFrame with 'Period' column converted to datetime.
    """
    df = pd.read_csv(path)
    df['Period'] = pd.to_datetime(df['Period'], format='%m/%Y')
    df.sort_values(by='Period', ascending=True, inplace=True)
    df['Period'] = df['Period'].dt.to_period('M')
    df.set_index('Period', inplace=True)
    return df


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


def calculate_percentage(part, whole):
    """
    Calculates the percentage of a part relative to the whole.

    Args:
        part (float): The numerator value.
        whole (float): The denominator value.

    Returns:
        float: The percentage value.
    """
    if whole == 0:
        return 0
    return (part / whole) * 100


def read_file(file_path):
    """
    Reads the content of a file and returns it as a string.

    Args:
        file_path (str): The path to the file.

    Returns:
        str: The content of the file.
    """
    with open(file_path, 'r') as file:
        return file.read()


def write_file(file_path, content):
    """
    Writes content to a file.

    Args:
        file_path (str): The path to the file.
        content (str): The content to write.

    Returns:
        None
    """
    with open(file_path, 'w') as file:
        file.write(content)


__all__ = ["load_dataframe", "getExchangeRate", "currencyUSDvals", "format_currency",
           "calculate_percentage", "read_file", "write_file"]
