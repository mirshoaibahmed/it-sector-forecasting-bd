import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os
import warnings
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
import numpy as np

warnings.filterwarnings('ignore')

def forecast_ict_exports_with_prophet(file_path):
    try:
        print(f"--- Loading and preparing data from '{file_path}' ---")
        df = pd.read_csv(file_path, index_col='Year', parse_dates=True)
        
        # Keep only ICT Exports column
        prophet_df = df.reset_index().rename(columns={'Year': 'ds', 'ICT_Exports': 'y'})
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        prophet_df = prophet_df.dropna(subset=['y'])
        
        # Train-test split
        train_df = prophet_df[prophet_df['ds'].dt.year <= 2018]
        test_df = prophet_df[prophet_df['ds'].dt.year > 2018]

        # Fit Prophet on training data
        print("\n--- Fitting Prophet model on training data (up to 2018) ---")
        model = Prophet(yearly_seasonality=True)
        model.fit(train_df)

        # Forecast on test data
        future_test = test_df[['ds']].copy()
        forecast_test = model.predict(future_test)
        
        # Accuracy metrics
        actual = test_df['y'].values
        predicted = forecast_test['yhat'].values
        mape = mean_absolute_percentage_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        r2 = r2_score(actual, predicted)

        print("\n--- Model Accuracy on Test Data (2019-2024) ---")
        print(f"MAPE: {mape*100:.2f}%")
        print(f"RMSE: {rmse:,.2f}")
        print(f"R²: {r2:.2f}")

        # Refit on full dataset
        print("\n--- Re-fitting on full dataset ---")
        full_model = Prophet(yearly_seasonality=True)
        full_model.fit(prophet_df)

        # Forecast 2025-2030
        future_dates = full_model.make_future_dataframe(periods=6, freq='Y')
        forecast = full_model.predict(future_dates)
        final_forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(6)
        final_forecast_df['Year'] = final_forecast_df['ds'].dt.year
        final_forecast_df = final_forecast_df[['Year', 'yhat', 'yhat_lower', 'yhat_upper']]
        final_forecast_df = final_forecast_df.rename(columns={
            'yhat': 'Forecasted ICT Exports (USD)',
            'yhat_lower': 'Lower 95% CI',
            'yhat_upper': 'Upper 95% CI'
        })

        print("\n--- Final Forecasted ICT Exports (2025-2030) ---")
        print(final_forecast_df.to_string(index=False))

        # Save forecast table
        csv_path = os.path.join(os.path.dirname(file_path), 'prophet_forecast_2025_2030.csv')
        final_forecast_df.to_csv(csv_path, index=False)
        print(f"\n--- Forecast table saved at: {csv_path} ---")

        # Plotting
        plt.figure(figsize=(12, 7))
        plt.plot(prophet_df['ds'].dt.year, prophet_df['y'], label='Historical Data', color='blue', marker='o')
        plt.plot(final_forecast_df['Year'], final_forecast_df['Forecasted ICT Exports (USD)'],
                 label='Forecast', color='red', linestyle='--', marker='o')
        plt.fill_between(final_forecast_df['Year'],
                         final_forecast_df['Lower 95% CI'],
                         final_forecast_df['Upper 95% CI'],
                         color='red', alpha=0.2, label='95% Confidence Interval')
        plt.title('Prophet Forecast for ICT Exports (2025-2030)', fontsize=16)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('ICT Exports (USD)', fontsize=12)
        plt.xticks(final_forecast_df['Year'])
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(os.path.dirname(file_path), 'prophet_forecast_plot_final.png')
        plt.savefig(plot_path)
        plt.show()
        print(f"\n--- Forecast plot saved at: {plot_path} ---")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Run the function
input_file = "data/prophet_ready_final_corrected.csv"
forecast_ict_exports_with_prophet(input_file)
