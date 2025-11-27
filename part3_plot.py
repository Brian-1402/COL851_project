import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from log import setup_logger

logger = None

def plot_cdf(df_metrics, city_name, output_dir):
    rmses = df_metrics["rmse"].sort_values().to_numpy()
    cdf = np.arange(1, len(rmses) + 1) / len(rmses)
    
    plt.figure(figsize=(8, 5))
    plt.plot(rmses, cdf, marker='.', linestyle='none')
    plt.xlabel(f"RMSE (PM 2.5)")
    plt.ylabel("CDF (Probability)")
    plt.title(f"CDF of RMSE - {city_name}")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    cdf_path = os.path.join(output_dir, f"{city_name}_rmse_cdf.png")
    plt.savefig(cdf_path)
    plt.close()
    logger.info(f"Saved CDF plot to {cdf_path}")

def plot_rmse_ts(df_preds, df_metrics, city_name, output_dir, bucket_size_hours):
    # Ensure datetime
    df_preds["timestamp"] = pd.to_datetime(df_preds["timestamp"])
    df_metrics["window_start"] = pd.to_datetime(df_metrics["window_start"])
    
    df_preds.set_index("timestamp", inplace=True)
    df_metrics.set_index("window_start", inplace=True)

    # Resample
    resample_rule = f"{bucket_size_hours}h"
    df_ts_agg = df_preds.resample(resample_rule).mean()
    df_metrics_agg = df_metrics.resample(resample_rule).mean() 

    # Plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # PM 2.5
    ax1.plot(df_ts_agg.index, df_ts_agg["ground_truth"], label="Ground Truth", color="black", alpha=0.6, linewidth=1.5)
    ax1.plot(df_ts_agg.index, df_ts_agg["forecast"], label="Forecast", color="blue", alpha=0.8, linewidth=1.5)
    
    ax1.set_xlabel(f"Date (Avg over {bucket_size_hours}h)")
    ax1.set_ylabel("PM 2.5 Concentration", color="black")
    ax1.tick_params(axis='y', labelcolor="black")
    ax1.legend(loc="upper left")

    # RMSE
    ax2 = ax1.twinx()
    ax2.plot(df_metrics_agg.index, df_metrics_agg["rmse"], label="Window RMSE", color="red", linestyle="--", alpha=0.5, linewidth=1.5)
    ax2.set_ylabel("RMSE", color="red")
    ax2.tick_params(axis='y', labelcolor="red")
    
    plt.title(f"Forecast Analysis: {city_name} (Aggregated {bucket_size_hours}h)")
    plt.tight_layout()
    
    ts_path = os.path.join(output_dir, f"{city_name}_forecast_analysis_{bucket_size_hours}h.png")
    plt.savefig(ts_path)
    plt.close()
    logger.info(f"Saved Time-Series plot to {ts_path}")

def main():
    global logger

    # --- Configuration Defaults ---
    DEFAULTS = {
        "city": "patna",
        "bucket_size": 24,
        "output_dir": "outputs/part3"
    }

    parser = argparse.ArgumentParser(description="Part 3 Plot: Visualize Chronos Results")
    
    # Path arguments default to None so we can detect if user provided them
    parser.add_argument("--preds_file", type=str, default=None, help="Path to predictions CSV. (Default: auto-constructed from dir+city)")
    parser.add_argument("--metrics_file", type=str, default=None, help="Path to metrics CSV. (Default: auto-constructed from dir+city)")
    
    parser.add_argument("--city", type=str, default=DEFAULTS["city"], help="City name for title/output")
    parser.add_argument("--bucket_size", type=int, default=DEFAULTS["bucket_size"], help="Hours to aggregate for TS plot")
    parser.add_argument("--output_dir", type=str, default=DEFAULTS["output_dir"], help="Directory to save plots")
    
    args = parser.parse_args()
    
    # --- Setup Logger ---
    os.makedirs("outputs", exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, "log")
    
    logger = setup_logger("part3_plot", "run_log", log_dir)

    # --- Dynamic Path Handling ---
    # If explicit files not provided, construct based on defaults/pipeline convention
    if args.preds_file is None:
        args.preds_file = os.path.join(args.output_dir, f"{args.city}_predictions.csv")
        logger.info(f"No preds_file specified, using default: {args.preds_file}")
        
    if args.metrics_file is None:
        args.metrics_file = os.path.join(args.output_dir, f"{args.city}_metrics.csv")
        logger.info(f"No metrics_file specified, using default: {args.metrics_file}")
    
    # Validation
    if not os.path.exists(args.preds_file):
        logger.error(f"Predictions file not found: {args.preds_file}")
        return
    if not os.path.exists(args.metrics_file):
        logger.error(f"Metrics file not found: {args.metrics_file}")
        return

    logger.info("Loading CSV data...")
    df_preds = pd.read_csv(args.preds_file)
    df_metrics = pd.read_csv(args.metrics_file)
    
    logger.info(f"Generating plots for {args.city} with bucket size {args.bucket_size}h...")
    plot_cdf(df_metrics, args.city, args.output_dir)
    plot_rmse_ts(df_preds, df_metrics, args.city, args.output_dir, args.bucket_size)

if __name__ == "__main__":
    main()