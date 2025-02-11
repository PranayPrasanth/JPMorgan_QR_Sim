import numpy as np
import pandas as pd


def price_gas_storage_contract(
        injection_dates,
        withdrawal_dates,
        market_prices,
        injection_rate,
        withdrawal_rate,
        max_storage,
        storage_cost_per_day
):
    """
    Prices a gas storage contract based on given parameters.

    Parameters:
    - injection_dates: List of dates when gas is injected.
    - withdrawal_dates: List of dates when gas is withdrawn.
    - market_prices: Dictionary {date: price} with gas market prices.
    - injection_rate: Rate at which gas can be injected per day.
    - withdrawal_rate: Rate at which gas can be withdrawn per day.
    - max_storage: Maximum gas storage capacity.
    - storage_cost_per_day: Cost of storing gas per unit per day.

    Returns:
    - Total value of the contract.
    """
    storage = 0  # Initial storage level
    cash_flows = []
    storage_costs = 0

    all_dates = sorted(set(injection_dates + withdrawal_dates))  # Consider all relevant dates

    for date in all_dates:
        if date in market_prices:  # Ensure the date exists in market_prices
            if date in injection_dates:
                injection_volume = min(injection_rate, max_storage - storage)  # Ensure we don't exceed capacity
                storage += injection_volume
                cost = -injection_volume * market_prices[date]  # Buying cost (negative cash flow)
                cash_flows.append(cost)

            if date in withdrawal_dates:
                withdrawal_volume = min(withdrawal_rate, storage)  # Ensure we don't withdraw more than available
                storage -= withdrawal_volume
                revenue = withdrawal_volume * market_prices[date]  # Selling revenue (positive cash flow)
                cash_flows.append(revenue)

            # Apply storage costs for each day
            storage_costs += storage * storage_cost_per_day

    total_value = sum(cash_flows) - storage_costs
    return total_value


# Load market prices CSV
market_prices = pd.read_csv('./Nat_Gas (1).csv')




# Ensure correct data types and formatting
market_prices['Dates'] = pd.to_datetime(market_prices['Dates']).dt.strftime('%Y-%m-%d')




# Convert DataFrame to dictionary {date: price}
market_prices_dict = dict(zip(market_prices['Dates'], market_prices['Prices']))  # Adjust column name if needed

# Define contract parameters
injection_dates = ['2020-12-31', '2021-02-28']
withdrawal_dates = ['2023-03-31']
injection_rate = 100
withdrawal_rate = 100
max_storage = 200
storage_cost_per_day = 0.1

# Compute contract value
contract_value = price_gas_storage_contract(
    injection_dates,
    withdrawal_dates,
    market_prices_dict,  # Pass dictionary, not DataFrame
    injection_rate,
    withdrawal_rate,
    max_storage,
    storage_cost_per_day
)

print(f"Contract Value: {contract_value}")



