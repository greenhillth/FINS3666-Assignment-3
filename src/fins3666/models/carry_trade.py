import numpy as np
import pandas as pd

from fins3666.defines import Order, AssetSpreads, ACCOUNT_SIZE_USD


def generate_orders(yields, current_positions, fx, min_balance=0, size=None):
    """
    Returns a list of orders required to rebalance position

    Args:
        yields: `pd.DataFrame` containing asset long and short yields.
        current_positions: `pd.DataFrame` of the form (asset, units).
        fx: `AssetSpread`  named tuple class containing current FX rates
    """

    # Generate list of orders as according to carry trade spec
    timestamp = yields.iloc[-1]['Timestamp']
    # Current position value
    df = current_positions[['Asset', 'Liquidation Value', 'Units']].copy()

    position = df['Liquidation Value'].sum()
    long_positions = df.loc[df['Liquidation Value']
                            > 0, 'Liquidation Value'].sum()
    short_positions = position - long_positions

    df['Weight'] = df['Liquidation Value']/(long_positions)

    df['Short'] = df['Liquidation Value'] < 0

    # Determine USD value for optimal portfolio weighting with asset and number
    size = long_positions if size is None else size
    per_asset_value = (size)/3

    # sort all rates in ascending order
    yields = yields.drop(columns=['Period', 'Timestamp'])
    yields_asc = yields.iloc[-1].sort_values().index.tolist()

    # Three lowest borrow rates
    short_asc = [k.removesuffix(' SHORT')
                 for k in yields_asc if ('SHORT') in k]
    # Three highest invest rates
    long_desc = list(reversed([k.removesuffix(' LONG')
                     for k in yields_asc if ('LONG') in k]))

    c_short = short_asc[:3]

    c_long = [x for x in long_desc if x not in c_short][:3]

    # mark positions for closure if no longer in lists
    df['Close'] = (
        (df['Short'] & ~df['Asset'].isin(c_short)) |
        (~df['Short'] & ~df['Asset'].isin(c_long) & (df['Units'] != 0))
    )

    transactions = [(row.Asset, -row.Units)
                    for row in df.itertuples() if abs(row.Units) > 1e-6]

    for currency in c_long:
        units = fx.ask.at[currency, 'USD'] * per_asset_value
        # units -= df.loc[df['Asset'] == currency, 'Units'].values[0]
        transactions.append((currency, units))
    for currency in c_short:
        units = fx.bid.at[currency, 'USD'] * -per_asset_value
        # units -= df.loc[df['Asset'] == currency, 'Units'].values[0]
        transactions.append((currency, units))

    aggregated = {}
    for asset, units in transactions:
        if asset in aggregated:
            aggregated[asset] += units
        else:
            aggregated[asset] = units

    transactions = [(asset, units)
                    for asset, units in aggregated.items() if abs(units) > 1e-4]
    buy = [t for t in transactions if t[1] > 0]
    sell = [t for t in transactions if t[1] < 0]

    orders = []

    # set up currency pairs
    for quote, amt in buy:
        base = sell[0][0] if sell else None
        units = abs(sell[0][1]) if sell else None
        while base is not None:
            sellUnits = fx.ask.at[base, quote]*amt
            remainingUnits = units - sellUnits
            if remainingUnits < 0:
                soldQuoteUnits = units/fx.ask.at[base, quote]
                amt = amt - soldQuoteUnits
                orders.append(Order(base, round(units, 4),
                              'sell', timestamp, quote))
                sell.pop(0)
                if sell:
                    base = sell[0][0]
                    units = abs(sell[0][1])
                else:
                    base = None
            else:
                if (round(sellUnits, 4)) > 0:
                    orders.append(Order(base, round(sellUnits, 4),
                                        'sell', timestamp, quote))
                sell[0] = (base, remainingUnits)
                base = None

    for quote, units in sell:
        if (round(abs(units), 4)) > 0:
            orders.append(Order(quote, round(abs(units), 4),
                                'sell', timestamp, 'USD'))
    return orders


__all__ = ["generate_orders"]
