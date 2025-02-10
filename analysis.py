import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt


# Load and preprocess the data
def load_data(file_path):
    df = pd.read_csv(file_path)  # Load CSV file
    df['Dates'] = pd.to_datetime(df['Dates'], errors="coerce").dt.strftime("%Y-%m-%d")
    df['Dates'] = pd.to_datetime(df['Dates'])# Convert to datetime
    df.set_index('Dates', inplace=True)  # Set Date as index
    df = df.asfreq('ME')  # Ensure monthly frequency
    df.interpolate(method='linear', inplace=True)  # Handle missing values
    return df


# Train an ARIMA model for forecasting
def train_model(df):
    model = ARIMA(df['Prices'], order=(2, 1, 2))  # ARIMA(p=2, d=1, q=2) for trend capture
    model_fit = model.fit()
    return model_fit


# Estimate price for a given date
def estimate_gas_price(input_date, df, model):
    input_date = pd.to_datetime(input_date)  # Convert input to datetime

    if input_date in df.index:  # If date exists in dataset
        return df.loc[input_date, 'Prices']

    elif input_date < df.index[0]:  # If the date is in the past
        return np.nan  # Cannot predict before data starts

    elif input_date > df.index[-1]:  # If the date is in the future
        forecast_steps = (input_date.year - df.index[-1].year) * 12 + (input_date.month - df.index[-1].month)
        future_forecast = model.forecast(steps=forecast_steps)
        return future_forecast.iloc[-1]  # Return last forecasted value


# Main function
def main():
    file_path = "./Nat_Gas (1).csv"  # Replace with your file
    df = load_data(file_path)

    model = train_model(df)

    input_date = input("Enter a date (YYYY-MM-DD): ")
    estimated_price = estimate_gas_price(input_date, df, model)
    print(f"Estimated Gas Price on {input_date}: {estimated_price}")

    # Optional: Plot forecast
    future_dates = pd.date_range(start=df.index[-1], periods=13, freq='M')[1:]
    future_forecast = model.forecast(steps=12)

    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['Prices'], label="Actual Prices", marker='o')
    plt.plot(future_dates, future_forecast, label="Forecasted Prices", linestyle="dashed", marker='x')
    plt.xlabel("Date")
    plt.ylabel("Gas Price")
    plt.title("Natural Gas Price Forecast")
    plt.legend()
    plt.show()


# Run the script
if __name__ == "__main__":
    main()
