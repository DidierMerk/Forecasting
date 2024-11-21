import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta

from datasetsforecast.losses import mae, rmse, mape, mase
from datasetsforecast.evaluation import accuracy
from sktime.performance_metrics.forecasting import MeanSquaredScaledError


# Class used to retrieve model statistics such as train and inference time
class Timer:
    def __init__(self):
        self.timestamps = {}

    def record_timestamp(self, label):
        self.timestamps[label] = time.time()

    def elapsed_time(self, start_label, end_label):
        return self.timestamps[end_label] - self.timestamps[start_label]


# Function which writes the model statistics to a file
def write_statistics(model_name, num_timeseries, train_time, inference_time, file_path):
    train_time_per_series = train_time / num_timeseries
    inference_time_per_series = inference_time / num_timeseries
    current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    stats = f"\n{model_name}, {num_timeseries}, {train_time:.4f}, {inference_time:.4f}, {train_time_per_series:.4f}, {inference_time_per_series:.4f}, {current_date_time}"

    with open(file_path, 'a') as file:
        file.write(stats)
        
    return


def plot_train_test_split(Y_train_df, Y_test_df, unique_id):
    # Pick a specific timeserie
    account_number = unique_id

    # Filter dataframes for the selected timeserie
    Y_train_sample = Y_train_df[Y_train_df['unique_id'] == account_number]
    Y_test_sample = Y_test_df[Y_test_df['unique_id'] == account_number]

    # Get the last 200 days from the training data
    Y_train_sample_recent = Y_train_sample.iloc[-150:]

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10))

    # First plot: full timeserie
    ax1.plot(Y_train_sample['ds'], Y_train_sample['y'], label=f'End-of-Day Balance')
    ax1.set_title(f'Three-Year End-of-Day Balances (account {account_number})')
    ax1.legend(loc='upper left')

    # Add vertical gray lines for each year
    start_year = int(Y_train_sample['ds'].dt.year.min())
    end_year = int(Y_train_sample['ds'].dt.year.max())

    for year in range(start_year + 1, end_year + 1):
        ax1.axvline(pd.Timestamp(f'{year}-01-01'), color='gray', linestyle='--')

    # Second plot: last 200 days of train data and test data
    ax2.plot(Y_train_sample_recent['ds'], Y_train_sample_recent['y'], label='Historic Data (Y_train_df)')
    ax2.plot(Y_test_sample['ds'], Y_test_sample['y'], 'b--', label='Ground Truth Data (Y_test_df)')

    # Add a vertical dotted line to separate train and test data
    ax2.axvline(x=Y_train_sample['ds'].iloc[-1], color='black', linestyle='--')

    # Add a connecting line between the last train point and the first test point
    last_train_point = Y_train_sample['ds'].iloc[-1], Y_train_sample['y'].iloc[-1]
    first_test_point = Y_test_sample['ds'].iloc[0], Y_test_sample['y'].iloc[0]

    ax2.plot([last_train_point[0], first_test_point[0]], [last_train_point[1], first_test_point[1]], 'b--')

    # Forecasting periods
    forecast_periods = [(0, 1), (1, 7), (7, 14), (14, 30)]
    colors = ['#ffeda0', '#feb24c', '#fd8d3c', '#f03b20']

    min_y = min(Y_train_sample_recent['y'].min(), Y_test_sample['y'].min())
    max_y = max(Y_train_sample_recent['y'].max(), Y_test_sample['y'].max())

    start_date = Y_train_sample['ds'].iloc[-1]

    for (start, end), color in zip(forecast_periods, colors):
        ax2.axvspan(start_date + pd.Timedelta(days=start), start_date + pd.Timedelta(days=end), color=color, alpha=0.15, label=f'Forecasting Horizon = {end} days')

    ax2.set_title('Forecasting Horizons')
    ax2.legend(loc='upper left')

    plt.tight_layout()
    plt.show()
    
    return


def plot_single_prediction(Y_train_df, Y_test_df, Y_pred_df, model_name, unique_id):
    # Pick a specific timeserie
    account_number = unique_id

    # Filter dataframes for the selected timeserie
    Y_train_sample = Y_train_df[Y_train_df['unique_id'] == account_number]
    Y_test_sample = Y_test_df[Y_test_df['unique_id'] == account_number]
    
    Y_pred_df = Y_pred_df.reset_index()
    Y_pred_sample = Y_pred_df[Y_pred_df['unique_id'] == account_number]

    # Get the last 150 days from the training data
    Y_train_sample_recent = Y_train_sample.iloc[-150:]

    # plotting
    plt.figure(figsize=(18, 5))

    # Plot the train and test data
    plt.plot(Y_train_sample_recent['ds'], Y_train_sample_recent['y'], label='Historic Data (Y_train_df)')
    plt.plot(Y_test_sample['ds'], Y_test_sample['y'], 'b--', label='Ground Truth Data (Y_test_df)')

    # Plot the prediction
    plt.plot(Y_pred_sample['ds'], Y_pred_sample[f'{model_name}'], label=f'{model_name}')

    # Add a vertical dotted line to separate train and test data
    plt.axvline(x=Y_train_sample['ds'].iloc[-1], color='black', linestyle='--')

    # Add a connecting line between the last train point and the first test point
    last_train_point = Y_train_sample['ds'].iloc[-1], Y_train_sample['y'].iloc[-1]
    first_test_point = Y_test_sample['ds'].iloc[0], Y_test_sample['y'].iloc[0]

    plt.plot([last_train_point[0], first_test_point[0]], [last_train_point[1], first_test_point[1]], 'b--')

    plt.title('Forecasting Horizons')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.show()
    
    return


def plot_multiple_forecasts(Y_train_df, Y_test_df, Y_pred_df, model_name, unique_ids):
    """
    Plot multiple time series forecasts in a 3x2 grid.
    
    Parameters:
    - Y_train_df: DataFrame containing the training data.
    - Y_test_df: DataFrame containing the test data.
    - Y_pred_df: DataFrame containing the predictions.
    - model_name: Name of the forecasting model used.
    - unique_ids: List of unique IDs for the time series to plot.
    """
    num_plots = len(unique_ids)
    fig, axes = plt.subplots(3, 2, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, account_number in enumerate(unique_ids):
        ax = axes[i]

        # Filter dataframes for the selected timeserie
        Y_train_sample = Y_train_df[Y_train_df['unique_id'] == account_number]
        Y_test_sample = Y_test_df[Y_test_df['unique_id'] == account_number]
        
        Y_pred_sample = Y_pred_df.reset_index()
        Y_pred_sample = Y_pred_sample[Y_pred_sample['unique_id'] == account_number]

        # Get the last 150 days from the training data
        Y_train_sample_recent = Y_train_sample.iloc[-150:]

        # Plot the train and test data
        ax.plot(Y_train_sample_recent['ds'], Y_train_sample_recent['y'], label='Historic Data (Y_train_df)')
        ax.plot(Y_test_sample['ds'], Y_test_sample['y'], 'b--', label='Ground Truth Data (Y_test_df)')

        # Plot the prediction
        ax.plot(Y_pred_sample['ds'], Y_pred_sample[model_name], label=f'{model_name}')

        # Add a vertical dotted line to separate train and test data
        ax.axvline(x=Y_train_sample['ds'].iloc[-1], color='black', linestyle='--')

        # Add a connecting line between the last train point and the first test point
        last_train_point = Y_train_sample['ds'].iloc[-1], Y_train_sample['y'].iloc[-1]
        first_test_point = Y_test_sample['ds'].iloc[0], Y_test_sample['y'].iloc[0]

        ax.plot([last_train_point[0], first_test_point[0]], [last_train_point[1], first_test_point[1]], 'b--')

        ax.set_title(f'Forecast for unique_id: "{account_number}"')
        ax.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

    return


def plot_multiple_model_forecasts(Y_train_df, Y_test_df, Y_pred_df, model_names, unique_ids):
    """
    Plot multiple time series forecasts in a 4x1 grid.
    
    Parameters:
    - Y_train_df: DataFrame containing the training data.
    - Y_test_df: DataFrame containing the test data.
    - Y_pred_df: DataFrame containing the predictions.
    - model_names: List of model names whose predictions need to be plotted.
    - unique_ids: List of unique IDs for the time series to plot.
    """
    num_plots = len(unique_ids)
    fig, axes = plt.subplots(num_plots, 1, figsize=(18, 20))
    axes = axes.flatten()
    
    for i, account_number in enumerate(unique_ids):
        ax = axes[i]

        # Filter dataframes for the selected timeserie
        Y_train_sample = Y_train_df[Y_train_df['unique_id'] == account_number]
        Y_test_sample = Y_test_df[Y_test_df['unique_id'] == account_number]
        
        Y_pred_sample = Y_pred_df[Y_pred_df['unique_id'] == account_number]

        # Get the last 150 days from the training data
        Y_train_sample_recent = Y_train_sample.iloc[-150:]

        # Plot the train and test data
        ax.plot(Y_train_sample_recent['ds'], Y_train_sample_recent['y'], label='Historic Data (Y_train_df)')
        ax.plot(Y_test_sample['ds'], Y_test_sample['y'], 'b--', label='Ground Truth Data (Y_test_df)')

        # Plot the predictions for each model
        for model_name in model_names:
            if model_name in Y_pred_sample.columns:
                ax.plot(Y_pred_sample['ds'], Y_pred_sample[model_name], label=f'{model_name}', linewidth=0.9)

        # Add a vertical dotted line to separate train and test data
        ax.axvline(x=Y_train_sample['ds'].iloc[-1], color='black', linestyle='--')

        # Add a connecting line between the last train point and the first test point
        last_train_point = Y_train_sample['ds'].iloc[-1], Y_train_sample['y'].iloc[-1]
        first_test_point = Y_test_sample['ds'].iloc[0], Y_test_sample['y'].iloc[0]

        ax.plot([last_train_point[0], first_test_point[0]], [last_train_point[1], first_test_point[1]], 'b--')

        ax.set_title(f'Forecast for unique_id: "{account_number}"')
        ax.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

    return


# # FUNCTIONS FOR EVALUATION ##

# Function to retrieve the cut off date (for 1, 7, 14, 30 days)
def retrieve_cutoff_date(pred_df, num_days):
    first_day = min(pred_df['ds'])
    cut_off_day = first_day + timedelta(days = num_days - 1)
    
    return cut_off_day


# Function to calculate RMSSE for each model
def calculate_rmsse(group, y_true_group, Y_train_df, model_columns):
    unique_id = group['unique_id'].iloc[0]
    results = {'unique_id': unique_id}
    
    y_true = y_true_group['y'].values
    y_train = Y_train_df['y'].values
    
    # Initialize the RMSSE metric
    rmsse_metric = MeanSquaredScaledError(square_root=True)
    
    for model in model_columns:
        y_pred = group[model].values
        rmsse_value = rmsse_metric(y_true, y_pred, y_train=y_train)  # Using y_true as y_train for simplicity
        results[model] = rmsse_value
    
    return results


def full_rmsse_calculation(Y_pred_df, Y_test_df, Y_train_df, model_columns):
    # Merge predictions with true values based on 'unique_id' and 'ds'
    merged_df = pd.merge(Y_pred_df, Y_test_df, on=['unique_id', 'ds'], suffixes=('', '_true'))
    
    # Apply the RMSSE calculation to each group of unique_id
    results = []
    for unique_id, group in merged_df.groupby('unique_id'):
        y_true_group = group[['ds', 'y']]
        y_pred_group = group.drop(columns=['y'])
        rmsse_results = calculate_rmsse(y_pred_group, y_true_group, Y_train_df, model_columns)
        results.append(rmsse_results)
        
    # Create the final dataframe
    results_df = pd.DataFrame(results)
    results_df['metric'] = 'rmsse'

    # Reorder columns
    cols = ['metric', 'unique_id'] + model_columns
    results_df = results_df[cols]
    
    return results_df


def perform_evaluation(Y_train_df, Y_test_df, Y_pred_df):
    # Divide into dataframes for 1 day, 7 days, 14 days and 30 days
    
    # Create the 30 days predictions dataframe
    full_30_days_df = Y_pred_df.drop(columns=['y']).reset_index()

    # Create 1 day prediction dataframe
    cut_off_day = retrieve_cutoff_date(full_30_days_df, 1)
    first_day_df = full_30_days_df[full_30_days_df.ds <= cut_off_day]
    first_day_test_df = Y_test_df[Y_test_df.ds <= cut_off_day]

    # Create 7 day prediction dataframe
    cut_off_day = retrieve_cutoff_date(full_30_days_df, 7)
    one_week_df = full_30_days_df[full_30_days_df.ds <= cut_off_day]
    one_week_test_df = Y_test_df[Y_test_df.ds <= cut_off_day]

    # Create 14 day prediction dataframe
    cut_off_day = retrieve_cutoff_date(full_30_days_df, 14)
    two_week_df = full_30_days_df[full_30_days_df.ds <= cut_off_day]
    two_week_test_df = Y_test_df[Y_test_df.ds <= cut_off_day]
    
    # Metric calculation for each dataframe
    metrics = [mae, mape, rmse]

    # Calculate the metrics
    evaluation_1_day = accuracy(first_day_df, metrics, Y_test_df=first_day_test_df, agg_by=['unique_id'])
    evaluation_7_days = accuracy(one_week_df, metrics, Y_test_df=one_week_test_df, agg_by=['unique_id'])
    evaluation_14_days = accuracy(two_week_df, metrics, Y_test_df=two_week_test_df, agg_by=['unique_id'])
    evaluation_30_days = accuracy(full_30_days_df, metrics, Y_test_df=Y_test_df, agg_by=['unique_id'])
    
    # List of model columns
    model_columns = [col for col in full_30_days_df.columns if col not in ['unique_id', 'ds']]
    
    # Calculate the RMSE's
    rmsses_1_days = full_rmsse_calculation(first_day_df, first_day_test_df, Y_train_df, model_columns)
    rmsses_7_days = full_rmsse_calculation(one_week_df, one_week_test_df, Y_train_df, model_columns)
    rmsses_14_days = full_rmsse_calculation(two_week_df, two_week_test_df, Y_train_df, model_columns)
    rmsses_30_days = full_rmsse_calculation(full_30_days_df, Y_test_df, Y_train_df, model_columns)
    
    # Concatenate the metrics
    full_eval_1_day = pd.concat([evaluation_1_day, rmsses_1_days], ignore_index=True)
    full_eval_7_days = pd.concat([evaluation_7_days, rmsses_7_days], ignore_index=True)
    full_eval_14_days = pd.concat([evaluation_14_days, rmsses_14_days], ignore_index=True)
    full_eval_30_days = pd.concat([evaluation_30_days, rmsses_30_days], ignore_index=True)
    
    return full_eval_1_day, full_eval_7_days, full_eval_14_days, full_eval_30_days

# ############

def summarize_metrics(df1, df2, df3, df4):
    dataframes = [df1, df2, df3, df4]
    horizons = ['01 day', '07 days', '14 days', '30 days']

    # List to store the results
    results = []

    # Loop over each dataframe and its corresponding horizon
    for df, horizon in zip(dataframes, horizons):
        # # Group by 'metric' and calculate mean for each model
        # avg_scores = df.drop(columns=['unique_id']).groupby('metric').mean().reset_index()
        
        # Group by 'metric' and calculate median for each model
        avg_scores = df.drop(columns=['unique_id']).groupby('metric').median().reset_index()
        
        # Add the 'horizon' column
        avg_scores['horizon'] = horizon
        # Append to results list
        results.append(avg_scores)

    # Concatenate all results into a single dataframe
    combined_df = pd.concat(results)

    # Set the index to be a MultiIndex with 'metric' and 'horizon'
    combined_df.set_index(['metric', 'horizon'], inplace=True)

    # Sort the index to ensure it is organized properly
    combined_df.sort_index(inplace=True)
    
    # Highlight the minimum value
    combined_df = combined_df.style.highlight_min(color = 'palegreen', axis = 1)
    
    # Add a line between rows
    line_separator = [{'selector': 'tr', 'props': [('border-bottom', '1px solid black')]}]
    combined_df = combined_df.set_table_styles(line_separator, overwrite=False)

    return combined_df


def filter_high_mape(eval_df, Y_test_df, mape_threshold=500, approx_0_threshold=0.02):
    # Check where for PatchTST the MAPE exceeds the threshold
    exceeding_error_ids = eval_df[(eval_df['metric'] == 'mape') & (eval_df['PatchTST'] > mape_threshold)]['unique_id'].unique()
    
    # Check where in the test data values get close to 0
    filtered_ids = Y_test_df[(Y_test_df['y'] >= -approx_0_threshold) & (Y_test_df['y'] <= approx_0_threshold)]['unique_id'].unique()
    
    # Find the common ids
    final_unique_ids = set(exceeding_error_ids).intersection(set(filtered_ids))
    
    # Remove those unique_ids from the metrics
    filtered_eval_df = eval_df[~eval_df['unique_id'].isin(final_unique_ids)]
    
    return filtered_eval_df

def filtered_metric_summary(df1, df2, df3, df4, Y_test_df, mape_threshold=500, approx_0_threshold=0.02):
    # Filter the weird mapes out
    filtered_df1 = filter_high_mape(df1, Y_test_df, mape_threshold, approx_0_threshold)
    filtered_df2 = filter_high_mape(df2, Y_test_df, mape_threshold, approx_0_threshold)
    filtered_df3 = filter_high_mape(df3, Y_test_df, mape_threshold, approx_0_threshold)
    filtered_df4 = filter_high_mape(df4, Y_test_df, mape_threshold, approx_0_threshold)
    
    filtered_results = summarize_metrics(filtered_df1, filtered_df2, filtered_df3, filtered_df4)
    
    return filtered_results


def merging_preds(Y_pred_df, model_preds, model_name):
    # Change the 'unique_id' column to string
    model_preds['unique_id'] = model_preds['unique_id'].astype('string')
    
    # Change the 'ds' column to datetime object
    model_preds['ds'] = model_preds['ds'].astype('datetime64[ns]')
    
    # Set the 'unique_id' column as index
    model_preds = model_preds.set_index('unique_id')
    
    # Add the Chronos predictions to the full prediction dataframe
    Y_pred_df = Y_pred_df.merge(model_preds[['ds', f'{model_name}']], on=['unique_id', 'ds'], how='left')
    
    return Y_pred_df


def define_eval_dfs(eval_1_day, eval_7_days, eval_14_days, eval_30_days, horizons):
    eval_dfs = []
    
    if '1 day' in horizons:
        eval_dfs.append(eval_1_day)
    if '7 days' in horizons:
        eval_dfs.append(eval_7_days)
    if '14 days' in horizons:
        eval_dfs.append(eval_14_days)
    if '30 days' in horizons:
        eval_dfs.append(eval_30_days)
        
    if len(eval_dfs) == 0:
        raise ValueError("You didn't add any forecasting horizons to evaluate.")
    
    return eval_dfs


num_timeseries_dict = {'tiny' : 20,
                       'xsmall' : 80,
                       'small' : 400,
                       'medium' : 800,
                       'large' : 2000,
                       'xlarge' : 8000,
                       'full' : 32000,
                    }
