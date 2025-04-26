import pandas as pd

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
    df['Period'] = df['Period'].dt.to_period('M')
    df.set_index('Period', inplace=True)
    return df


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


__all__ = ["load_dataframe", "format_currency",
           "calculate_percentage", "read_file", "write_file"]
