import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

def forecast_ict_exports_with_accuracy(file_path):
    """
    Loads prepared data, fits SARIMA(1,1,1) on log-transformed data,
    forecasts ICT exports from 2025 to 2030, calculates accuracy,
    and generates a thesis-ready plot.
    """
    try:
        warnings.filterwarnings('ignore')
        print(f"--- Loading prepared data from '{file_path}' ---")
        df = pd.read_csv(file_path, index_col='Year', parse_dates=True)
        
        # --- Log-transform ---
        if (df['ICT_Exports'] <= 0).any():
            raise ValueError("ICT_Exports contains zero or negative values. Log-transform is invalid.")
        endog = np.log(df['ICT_Exports'].dropna())

        # --- Train-test split for accuracy ---
        train_endog = endog.loc[:'2018']
        test_endog = endog.loc['2019':]

        # --- Fit SARIMA(1,1,1) ---
        print("\n--- Fitting SARIMA(1,1,1) model on log-transformed training data (up to 2018) ---")
        model = SARIMAX(train_endog, order=(1, 1, 1),
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        results = model.fit(disp=False)

        # --- Forecast test period ---
        forecast_test = results.get_forecast(steps=len(test_endog))
        forecast_test_log = forecast_test.predicted_mean
        forecast_actual = np.exp(forecast_test_log)
        actual_test_data = np.exp(test_endog)

        # --- Accuracy metrics ---
        mape = mean_absolute_percentage_error(actual_test_data, forecast_actual)
        rmse = np.sqrt(mean_squared_error(actual_test_data, forecast_actual))
        r2 = r2_score(actual_test_data, forecast_actual)

        print(f"\n--- Model Accuracy on Test Data (2019-2024) ---")
        print(f"MAPE: {mape*100:.2f}%")
        print(f"RMSE: {rmse:,.2f}")
        print(f"R²: {r2:.2f}")

        # --- Refit on full dataset ---
        print("\n--- Re-fitting on full dataset for future forecast ---")
        full_model = SARIMAX(endog, order=(1, 1, 1),
                             enforce_stationarity=False,
                             enforce_invertibility=False)
        full_results = full_model.fit(disp=False)

        # --- Manual step-by-step forecast for 2025-2030 ---
        print("\n--- Manually generating forecast for 2025-2030 ---")
        forecast_dates = pd.date_range(start=endog.index.max(), periods=6, freq='A')[1:]

        forecast_values_log = []
        lower_ci_log = []
        upper_ci_log = []

        for step in range(1, len(forecast_dates) + 1):
            one_step_forecast = full_results.get_forecast(steps=step)
            forecast_values_log.append(one_step_forecast.predicted_mean.iloc[-1])
            ci_frame = one_step_forecast.conf_int()
            lower_ci_log.append(ci_frame.iloc[-1].iloc[0])
            upper_ci_log.append(ci_frame.iloc[-1].iloc[1])

        forecast_series_log = pd.Series(forecast_values_log, index=forecast_dates)
        forecast_series = np.exp(forecast_series_log)
        lower_ci = np.exp(pd.Series(lower_ci_log, index=forecast_dates))
        upper_ci = np.exp(pd.Series(upper_ci_log, index=forecast_dates))

        forecast_df = pd.DataFrame({
            'Forecasted ICT Exports (USD)': forecast_series,
            'Lower Confidence Interval': lower_ci,
            'Upper Confidence Interval': upper_ci
        }, index=forecast_dates)

        forecast_df.index = forecast_df.index.year
        print("\n--- Final Forecasted ICT Exports (2025-2030) ---")
        print(forecast_df)

        # --- Plotting ---
        print("\n--- Generating forecast plot ---")
        plt.figure(figsize=(12, 7))
        historical_untransformed = df['ICT_Exports'].dropna()
        plt.plot(historical_untransformed.index.year, historical_untransformed.values,
                 label='Historical Data', color='blue', marker='o')
        plt.plot(forecast_df.index, forecast_df['Forecasted ICT Exports (USD)'],
                 label='Forecast', color='red', linestyle='--', marker='o')
        plt.fill_between(forecast_df.index,
                         forecast_df['Lower Confidence Interval'],
                         forecast_df['Upper Confidence Interval'],
                         color='red', alpha=0.2, label='95% Confidence Interval')

        plt.title('SARIMA Forecast for ICT Exports (2025-2030)', fontsize=16)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('ICT Exports (USD)', fontsize=12)
        plt.xticks(forecast_df.index)
        plt.gca().yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save plot
        output_plot_path = os.path.join(os.path.dirname(file_path), 'sarima_forecast_plot_final.png')
        plt.savefig(output_plot_path, dpi=300)
        plt.show()
        print(f"\n--- Forecast plot saved to: {output_plot_path} ---")

        print("\nForecasting complete. Results and plot are thesis-ready.")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Check the path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# --- Run the function ---
input_file = "data/sarima_ready_final_corrected.csv"
forecast_ict_exports_with_accuracy(input_file)
