import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm

from datasetsforecast.losses import mae, rmse, mape, mase
from datasetsforecast.evaluation import accuracy
from sktime.performance_metrics.forecasting import MeanSquaredScaledError

# Entropy calculation
import EntropyHub as EH

# Scaling
from sklearn.preprocessing import MinMaxScaler

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


def compare_single_model_rmsse(df, trained_model, metric, n_days, baselines, outlier_percentage=10):
    # Number of rows and columns in the grid
    n_cols = len(baselines)
    
    # Create a figure and a grid of subplots
    fig, axes = plt.subplots(1, n_cols, figsize=(15, 5), sharey=False)
    
    for col, baseline_model in enumerate(baselines):
        ax = axes[col]
        
        # Filter the dataframe for the given metric
        metric_df = df[df['metric'] == f'{metric}']
        
        # Extract the metric values for the baseline and trained models
        baseline_metric = metric_df[baseline_model].values
        trained_metric = metric_df[trained_model].values
        
        # Determine the color for each point
        colors = ['green' if trained < baseline else 'red' for trained, baseline in zip(trained_metric, baseline_metric)]
        
        # Calculate the percentage of timeseries where each model is better
        better_count_trained = sum(trained < baseline for trained, baseline in zip(trained_metric, baseline_metric))
        total_count = len(trained_metric)
        better_percentage_trained = (better_count_trained / total_count) * 100
        better_percentage_baseline = 100 - better_percentage_trained
        
        # Set the log scale
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Set axis limits based on the specified outlier percentage
        x_min = np.percentile(baseline_metric, outlier_percentage / 2)
        x_max = np.percentile(baseline_metric, 100 - outlier_percentage / 2)
        
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([x_min, x_max])
        
        # Plot the y=x line
        ax.plot([x_min, x_max], [x_min, x_max], 'k--', lw=1)
        
        # Create a scatter plot
        ax.scatter(baseline_metric, trained_metric, c=colors, s=3, alpha=0.3)
        
        # Calculate the median point
        median_x = np.median(baseline_metric)
        median_y = np.median(trained_metric)
        
        # Plot the median point
        ax.scatter(median_x, median_y, c='black', s=10, label=f'Medians ({median_x:.2f}, {median_y:.2f})')
        
        # Add vertical and horizontal lines up to the median point
        ax.plot([x_min, median_x], [median_y, median_y], color='gray', linestyle='-', lw=1)
        ax.plot([median_x, median_x], [x_min, median_y], color='gray', linestyle='-', lw=1)
        
        # # Add vertical and horizontal lines from the median point to the axes
        # ax.axvline(median_x, color='gray', linestyle='--', lw=1)
        # ax.axhline(median_y, color='gray', linestyle='--', lw=1)
        
        # Label the axes
        if col == 0:
            ax.set_ylabel(f'{trained_model} {metric.upper()}')
        ax.set_xlabel(f'{baseline_model} {metric.upper()}')
        
        # Add a legend
        green_label = f'{trained_model} better: {better_percentage_trained:.2f}%'
        red_label = f'{baseline_model} better: {better_percentage_baseline:.2f}%'
        ax.legend([plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, alpha=0.5), 
                   plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, alpha=0.5),
                   plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10, alpha=0.5)], 
                  [green_label, red_label, f'Median {metric.upper()}'], loc='upper left')
    
    # Add a title above the row
    fig.suptitle(f'{trained_model} vs Baselines ({metric.upper()}, {n_days}-day horizon)', size='x-large', y=0.95)
    
    # Layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
    return


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


def plot_metric_distribution(eval_dfs, models, metric, Y_test_df, horizons, filtered='yes', grid='horizontal'):
    # horizons = ['1 day', '7 days', '14 days', '30 days']
    num_models = len(models)
    bins = 60
    
    if filtered == 'yes':
        eval_dfs = [filter_high_mape(eval_dfs[i], Y_test_df) for i in range(len(horizons))]

    # Initialize the figure and axis
    fig, axs = plt.subplots(num_models, len(horizons), figsize=(20, 5 * num_models), constrained_layout=True)
    
    # Ensure axs is a 2D array even if there's only one row or one column
    if num_models == 1:
        axs = np.expand_dims(axs, axis=0)
    if len(horizons) == 1:
        axs = np.expand_dims(axs, axis=1)
    
    # Function to calculate the 0th and 85th percentiles
    def get_percentile_limits(data, low=0, high=85):
        return np.percentile(data, low), np.percentile(data, high)

    # Calculate the metric dataframes and limits
    metric_data = [{horizon: df[df['metric'] == metric] for horizon, df in zip(horizons, eval_dfs)} for model in models]
    
    # Collect all metric values for computing global x-axis limits
    all_metric_values = np.concatenate([
        metric_data[i][horizon][model].values 
        for horizon in horizons 
        for i, model in enumerate(models)
    ])
    
    # Calculate global x-axis limits based on percentiles
    global_x_limits = get_percentile_limits(all_metric_values, 0, 85)
    
    # Calculate row-specific y-axis limits and medians
    row_y_limits = []
    medians = {model: [] for model in models}
    
    for i, model in enumerate(models):
        model_medians = []
        max_y_limit = 0
        for horizon in horizons:
            horizon_values = metric_data[i][horizon][model].values
            model_medians.append(np.median(horizon_values))
            max_y_limit = max(max_y_limit, np.histogram(horizon_values, bins=bins, range=global_x_limits)[0].max())
        row_y_limits.append((0, max_y_limit))
        medians[model] = model_medians

    for i, model in enumerate(models):
        for j, horizon in enumerate(horizons):
            ax = axs[i, j] if num_models > 1 else axs[j]

            # Plot the histogram within the specified global x-axis range
            ax.hist(metric_data[i][horizon][model], bins=bins, alpha=0.5, edgecolor='black', range=global_x_limits)
            ax.axvline(medians[model][j], color='black', linestyle='dashed', linewidth=1.5, label=f'Median {metric.upper()} = {medians[model][j]:.2f}%')
            ax.set_xlim(global_x_limits)
            ax.set_ylim(row_y_limits[i])
            ax.set_title(f"{model} (fh = {horizon})")
            if j == 0:
                ax.set_ylabel('Frequency')
            else:
                ax.set_yticklabels([])

            # Turn on horizontal grid lines
            ax.yaxis.grid(True)
            if grid == 'vertical':
                ax.xaxis.grid(True)
            elif grid == 'both':
                ax.yaxis.grid(True)
                ax.xaxis.grid(True)
            elif grid == 'none':
                ax.yaxis.grid(False)
                ax.xaxis.grid(False)

            ax.legend()

    plt.show()
    
    return


def plot_metrics_over_time(metric_df):
    # Ensure metric_df is a DataFrame and not a Styler
    if hasattr(metric_df, 'data'):
        metric_df = metric_df.data
    
    metrics = ['mae', 'mape', 'rmse', 'rmsse']
    horizons = ['01 day', '07 days', '14 days', '30 days']
    days = [1, 7, 14, 30]
    
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axs[i]
        try:
            metric_data = metric_df.loc[metric]
        except KeyError:
            print(f"Metric {metric} not found in the dataframe.")
            continue
        
        for j, model in enumerate(metric_df.columns):
            ax.plot(days, metric_data[model], marker=markers[j % len(markers)], label=model)
        
        ax.set_title(f'{metric.upper()} over Time')
        ax.set_xlabel('Forecasting Horizon (days)')
        ax.set_ylabel(metric.upper())
        ax.grid(False)
    
    # Create a global legend
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.02))
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
    return


def ApproxEntropy(data, num_groups):
    # Initialize a list to store the entropies
    entropies = []
    
    # Iterate over each timeseries column (excluding the 'date' column)
    for column in tqdm(data.columns[1:]):
        # Convert the timeseries to a numpy array and handle NaN values
        timeserie = data[column].dropna().values
        
        # Calculate Approximate Entropy
        entropy_value = EH.ApEn(timeserie[-4 * 30:-30])[0][2]
        
        # Save the calculated entropy
        entropies.append({'unique_id': str(column), 'ApproxEntropy': entropy_value})
    
    # Create an entropies dataframe
    entropies_df = pd.DataFrame(entropies)
    
    # Calculate the percentiles and assign groups
    entropies_df['Group'] = pd.qcut(entropies_df['ApproxEntropy'], q=num_groups, labels=list(range(1,num_groups+1)))
    
    return entropies_df


def RateOfChange(data, num_groups, num_horizons=3, fh=30):
    # Scale the data using a MinMax Scaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.iloc[: , 1:])
    
    # Change back to a pandas dataframe
    scaled_dataset = pd.DataFrame(scaled_data, columns=data.iloc[: , 1:].columns)
    
    # Take the last 3 horizons before to calculate rate of change
    data_length = len(data)
    scaled_dataset_short = scaled_dataset.loc[int(data_length) - (num_horizons + 1) * fh : int(data_length) - fh - 1]
    
    # Calculate the difference between each row and the row below it for each column
    differences = scaled_dataset.diff(periods=-1)
    
    # Drop the last row since it will have NaN values after the diff operation
    differences = differences.iloc[:-1]

    # Take the absolute value since that is what matters
    differences = abs(differences)

    # Calculate the mean of these differences for each column
    # average_daily_changes = differences.mean()
    average_daily_changes = differences.median()
    
    # Create a new DataFrame with unique_id and roc columns
    roc_df = pd.DataFrame({'unique_id': average_daily_changes.index, 'RoC': average_daily_changes.values})
    
    # Calculate the percentiles and assign groups
    roc_df['Group'] = pd.qcut(roc_df['RoC'], q=num_groups, labels=list(range(1,num_groups+1)))
    
    return roc_df


def GroupSeries(data, num_groups, group_method, num_horizons=3, fh=30):
    if group_method == 'ApproxEntropy':
        group_df = ApproxEntropy(data, num_groups)
    elif group_method == 'RateOfChange':
        group_df = RateOfChange(data, num_groups, num_horizons, fh)
    else:
        raise ValueError("Not a valid method of timeseries classification was entered. Has to be one of ['ApproxEntropy', 'RateOfChange']")
        
    return group_df


def PlotGroups(data, eval_df, num_horizons=3, fh=30):
    # Select one unique_id from each group at random
    selected_timeseries = eval_df.groupby('Group')['unique_id'].apply(lambda x: x.sample(1)).reset_index(drop=True)
    
    # Get the number of groups
    num_groups = selected_timeseries.shape[0]

    # Calculate the number of rows and columns for the grid
    num_cols = 5
    num_rows = int(np.ceil(num_groups / num_cols))
    
    # Create the subplot grid
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 3 * num_rows), sharey=True)
    axes = axes.flatten()
    
    # MinMaxScaler instance
    scaler = MinMaxScaler()

    # Plot each selected timeseries
    for i, unique_id in enumerate(selected_timeseries):
        group = eval_df[eval_df['unique_id'] == unique_id]['Group'].values[0]
        timeseries = data[unique_id]
        
        # MinMax scale the timeseries
        timeseries_scaled = scaler.fit_transform(timeseries.values.reshape(-1, 1)).flatten()
        
        axes[i].plot(timeseries_scaled[-num_horizons * fh : -fh])
        axes[i].set_title(f'Group {group}')
        axes[i].set_xticks([])
        # axes[i].set_yticks([])
        axes[i].grid(True)

    for j in range(i + 1, num_rows * num_cols):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
    
    return


# Function to compute the median mape for a model across horizons
def get_median_mape(df_list, model):
    median_mapes = []
    for df in df_list:
        median_mapes.append(df[df['metric'] == 'mape'][model].median())
    return median_mapes


# Function to compute the median mape per group for a model across horizons
def get_group_median_mape(df_list, model):
    group_median_mapes = {}
    for group in df_list[0]['Group'].unique():
        group_medians = []
        for df in df_list:
            group_medians.append(df[(df['metric'] == 'mape') & (df['Group'] == group)][model].median())
        group_median_mapes[group] = group_medians
    return group_median_mapes


def PlotGroupPerformance(dataframes, models, horizons):
    # Plotting
    num_models = len(models)
    num_cols = 3
    num_rows = int(np.ceil(num_models / num_cols))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 4 * num_rows), sharey=True)
    axes = axes.flatten()

    for idx, model in enumerate(models):
        ax = axes[idx]

        # Median mape per group
        group_median_mapes = get_group_median_mape(dataframes, model)
        num_groups = len(group_median_mapes)
        colors = [plt.cm.Greys(0.4 + 0.6 * i / (num_groups - 1)) for i in range(num_groups)]

        for group, group_medians in group_median_mapes.items():
            ax.plot(horizons, group_medians, color=colors[group-1], linestyle='--', linewidth=1.0)

        # Formatting
        ax.set_xlabel('Forecasting Horizon (days)')
        if idx % num_cols == 0:
            ax.set_ylabel('Median MAPE')
        ax.set_title(f'{model}')
        ax.set_xticks(horizons)
        ax.set_xticklabels(horizons)
        ax.grid(True)

        # Median mape across all groups
        median_mape = get_median_mape(dataframes, model)
        ax.plot(horizons, median_mape, label=f'{model} (all groups)', color='red', linewidth=2.5)

        # Legend
        handles = [
            plt.Line2D([0], [0], color='red', linewidth=2.5, label=f'Full Data'),
            plt.Line2D([0], [0], color='gray', linestyle='--', linewidth=1.0, label='ApEn Group')
        ]
        ax.legend(handles=handles, loc='upper left')

    for j in range(idx + 1, num_rows * num_cols):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
    
    return


def Plotting_MAPE(df1, df2, model_categories, colors):
    # Convert data to dataframe
    df1 = df1.data
    df2 = df2.data
    
    # Extract data for plotting and sort it
    mape_30d_df1 = df1.loc[('mape', '30 days')].sort_values(ascending=True).round(2)
    mape_30d_df2 = df2.loc[('mape', '30 days')].sort_values(ascending=True).round(2)

    models_df1 = mape_30d_df1.index
    models_df2 = mape_30d_df2.index
    
    num_models = len(df1.columns)

    # Plotting
    fig, axes = plt.subplots(ncols=2, figsize=(num_models * 14/8, num_models * 5/8))

    for ax, models, mape_data, xlabel in zip(axes, [models_df1, models_df2], [mape_30d_df1, mape_30d_df2], ['In-sample MAPE', 'Out-sample MAPE']):
        colors_list = [colors[model_categories[model]] for model in models]
        ax.barh(models, mape_data, color=colors_list)
        ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
        ax.invert_yaxis()  # Invert y-axis to have the lowest MAPE at the top
        for i, v in enumerate(mape_data):
            ax.text(v + 0.25, i, f'{v:.2f}', color='black', va='center')

        ax.set_xlim(0, max(mape_data) + 0.1 * max(mape_data))

    # Adjust the plot to ensure all numbers fit
    plt.subplots_adjust(left=0.2, right=1.95, top=0.9, bottom=0.1)

    # Custom legend
    handles = [plt.Rectangle((0,0),1,1, color=color) for color in colors.values()]
    labels = colors.keys()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
    return

