import pandas as pd
import numpy as np
import torch
import os
from chronos import BaseChronosPipeline
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import time
import sys
from tqdm import tqdm

from log import setup_logger

output_dir = "outputs/part3"
os.makedirs("outputs", exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
log_dir = os.path.join(output_dir, "log")

logger = setup_logger("part3", "run_log", log_dir)


def load_city_data(file):
    df = pd.read_csv(file, parse_dates=["From Date"])
    return df

def load_pipeline(model_variant):
    logger.info(f"\nLoading {model_variant}...")
    return BaseChronosPipeline.from_pretrained(
        model_variant,
        device_map="cpu",
        torch_dtype=torch.float32
    )

def forecast(pipeline, context, horizon):
    context = torch.tensor(context, dtype=torch.float32)
    _, mean = pipeline.predict_quantiles(context, prediction_length=horizon, quantile_levels=[0.5])
    return mean.squeeze().numpy()

def compute_rmse(pipeline, df, context_len_hours, horizon_hours):
    """
    Computes RMSE and collects data for plotting.
    Assumes non-overlapping windows (step = horizon_hours) for continuous plotting.
    """
    start_time = time.time()
    
    # Extract series and dates
    series = df["calibPM"].astype(float).to_numpy()
    dates = df["From Date"].to_numpy()
    
    total = len(series)
    step = horizon_hours # Non-overlapping windows
    
    results = {
        "rmses": [],
        "rmse_dates": [],    # Date corresponding to the specific window
        "preds": [],         # Flattened list of all predictions
        "pred_dates": [],    # Dates for the predictions
        "ground_truth": [],  # Ground truth for the prediction period
    }

    logger.info(f"Starting evaluation... Total len: {total}, Context: {context_len_hours}, Step: {step}")

    for i in tqdm(range(context_len_hours, total - horizon_hours, step)):
        # Prepare context and truth
        context = series[i - context_len_hours : i]
        true_window = series[i : i + horizon_hours]
        true_dates = dates[i : i + horizon_hours]
        
        # Forecast
        pred_window = forecast(pipeline, context, horizon_hours)
        
        # Calculate RMSE for this specific window
        window_rmse = np.sqrt(mean_squared_error(true_window, pred_window))
        
        # Store results
        results["rmses"].append(window_rmse)
        results["rmse_dates"].append(dates[i]) # Log the start time of the forecast window
        
        # Store series data for plotting
        results["preds"].extend(pred_window)
        results["pred_dates"].extend(true_dates)
        results["ground_truth"].extend(true_window)

    elapsed = time.time() - start_time
    logger.info(f"compute_rmse completed in {elapsed/60:.2f} min. Processed {len(results['rmses'])} windows.")
    return results


def plot_cdf(results, city_name, output_dir="outputs"):
    rmses = np.array(results["rmses"])
    
    # --- Plot 1: CDF of RMSE ---
    plt.figure(figsize=(8, 5))
    sorted_rmse = np.sort(rmses)
    cdf = np.arange(1, len(sorted_rmse) + 1) / len(sorted_rmse)
    
    plt.plot(sorted_rmse, cdf, marker='.', linestyle='none')
    plt.xlabel(f"RMSE (PM 2.5)")
    plt.ylabel("CDF (Probability)")
    plt.title(f"CDF of RMSE - {city_name}")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    cdf_path = os.path.join(output_dir, f"{city_name}_rmse_cdf.png")
    plt.savefig(cdf_path)
    plt.close()
    logger.info(f"Saved CDF plot to {cdf_path}")

def plot_rmse(results, city_name, output_dir="outputs", bucket_size_hours=24):
    """
    Plots time series of Ground Truth, Forecast, and RMSE.
    Aggregates data by `bucket_size_hours` to reduce plot density.
    """
    # 1. Prepare DataFrames for resampling
    df_ts = pd.DataFrame({
        "date": results["pred_dates"],
        "ground_truth": results["ground_truth"],
        "forecast": results["preds"]
    })
    df_ts["date"] = pd.to_datetime(df_ts["date"])
    df_ts.set_index("date", inplace=True)

    df_rmse = pd.DataFrame({
        "date": results["rmse_dates"],
        "rmse": results["rmses"]
    })
    df_rmse["date"] = pd.to_datetime(df_rmse["date"])
    df_rmse.set_index("date", inplace=True)

    # 2. Resample (Average over bucket_size_hours)
    resample_rule = f"{bucket_size_hours}h"
    
    # We use .mean() to average values within the bucket
    df_ts_agg = df_ts.resample(resample_rule).mean()
    df_rmse_agg = df_rmse.resample(resample_rule).mean()

    # 3. Plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # -- Primary Y-axis: PM 2.5 --
    ax1.plot(df_ts_agg.index, df_ts_agg["ground_truth"], label="Ground Truth", color="black", alpha=0.6, linewidth=1.5)
    ax1.plot(df_ts_agg.index, df_ts_agg["forecast"], label="Forecast", color="blue", alpha=0.8, linewidth=1.5)
    
    ax1.set_xlabel(f"Date (Avg over {bucket_size_hours}h)")
    ax1.set_ylabel("PM 2.5 Concentration", color="black")
    ax1.tick_params(axis='y', labelcolor="black")
    ax1.legend(loc="upper left")

    # -- Secondary Y-axis: RMSE --
    ax2 = ax1.twinx()
    # Handle case where RMSE might be sparse or NaN after resampling if no windows aligned perfectly (unlikely with mean)
    ax2.plot(df_rmse_agg.index, df_rmse_agg["rmse"], label="Window RMSE", color="red", linestyle="--", alpha=0.5, linewidth=1.5)
    
    ax2.set_ylabel("RMSE", color="red")
    ax2.tick_params(axis='y', labelcolor="red")
    # ax2.legend(loc="upper right") # Optional

    plt.title(f"Forecast Analysis: {city_name} (Aggregated {bucket_size_hours}h)")
    plt.tight_layout()
    
    ts_path = os.path.join(output_dir, f"{city_name}_forecast_analysis.png")
    plt.savefig(ts_path)
    plt.close()
    logger.info(f"Saved Time-Series plot to {ts_path}")

def plot_results(results, city_name, output_dir):
    """
    Generates:
    1. CDF of RMSE
    2. Time-series comparison (Test Day vs Ground Truth/Forecast/RMSE)
    """
    plot_cdf(results, city_name, output_dir)
    plot_rmse(results, city_name, output_dir)


def main():
    # Load Data
    df_gurgaon = load_city_data("df_ggn_covariates.csv")
    df_patna = load_city_data("df_patna_covariates.csv")

    model_variants = [
        "amazon/chronos-t5-mini",
        "amazon/chronos-t5-small",
        "amazon/chronos-bolt-small"
    ]

    # best params found from part2
    model = "amazon/chronos-t5-small"
    ctx_days = 4
    horizon_hours = 4
    # predownload
    # for model in model_variants:
    #     load_pipeline(model)
    # Load Model
    pipeline = load_pipeline(model)
    
    # Compute
    results = compute_rmse(
        pipeline,
        df_patna,
        context_len_hours=ctx_days * 24,
        horizon_hours=horizon_hours
    )
    
    # Plot
    plot_results(results, "patna", output_dir)

if __name__ == "__main__":
    main()