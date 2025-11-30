"""
Part 3 compute

Input:
    ./df_patna_covariates.csv
    ./df_ggn_covariates.csv (maybe)
    (Requires columns: "From Date", "calibPM")

Outputs (saved to ./outputs/part3/):
    1. patna_predictions.csv  - [timestamp, ground_truth, forecast]
    2. patna_metrics.csv      - [window_start, rmse]
    3. patna_performance.csv  - [window_start, latency_sec, throughput_preds_per_sec, cpu_util_pct, ram_util_mb]
    4. log/                   - Execution logs
"""
import argparse
import pandas as pd
import numpy as np
import torch
import os
import time
import sys
import psutil
from tqdm import tqdm
from chronos import BaseChronosPipeline
from sklearn.metrics import mean_squared_error

from log import setup_logger

logger = None

def load_pipeline(model_variant, device_map="cpu"):
    logger.info(f"Loading {model_variant} on {device_map}...")
    return BaseChronosPipeline.from_pretrained(
        model_variant,
        device_map=device_map,
        torch_dtype=torch.float32
    )

def forecast(pipeline, context, horizon):
    context_tensor = torch.tensor(context, dtype=torch.float32)
    _, mean = pipeline.predict_quantiles(context_tensor, prediction_length=horizon, quantile_levels=[0.5])
    return mean.squeeze().numpy()

def run_inference(pipeline, df, context_len_hours, horizon_hours):
    start_time = time.time()
    
    # --- Process Monitor Setup ---
    # We target the specific PID of this Python script
    current_process = psutil.Process(os.getpid())
    # First call initializes the counter; subsequent calls return avg usage since last call
    current_process.cpu_percent(interval=None)
    
    series = df["calibPM"].astype(float).to_numpy()
    dates = df["From Date"].to_numpy()
    total = len(series)
    step = horizon_hours 

    metrics_data = [] 
    preds_data = []   
    perf_data = []

    logger.info(f"Starting inference... Series len: {total}, Context: {context_len_hours}, Horizon: {horizon_hours}")

    for i in tqdm(range(context_len_hours, total - horizon_hours, step)):
        context = series[i - context_len_hours : i]
        true_window = series[i : i + horizon_hours]
        true_dates = dates[i : i + horizon_hours]
        
        # Performance measurement
        t_start = time.perf_counter()
        pred_window = forecast(pipeline, context, horizon_hours)
        t_end = time.perf_counter()
        
        # --- Capture Process Metrics ---
        # Can be > 100% if the process is multithreaded (e.g., PyTorch backend).
        cpu_util = current_process.cpu_percent(interval=None)
        
        # RSS (Resident Set Size): Actual physical memory used by the process
        ram_util_mb = current_process.memory_info().rss / (1024 * 1024)
        
        latency = t_end - t_start
        # Throughput as instances (prediction windows) per second
        throughput = 1.0 / latency if latency > 0 else 0.0

        perf_data.append({
            "window_start": dates[i],
            "latency_sec": latency,
            "throughput_preds_per_sec": throughput,
            "cpu_util_pct": cpu_util,
            "ram_util_mb": ram_util_mb
        })

        window_rmse = np.sqrt(mean_squared_error(true_window, pred_window))
        
        metrics_data.append({
            "window_start": dates[i],
            "rmse": window_rmse
        })
        
        for d, gt, p in zip(true_dates, true_window, pred_window):
            preds_data.append({
                "timestamp": d,
                "ground_truth": gt,
                "forecast": p
            })

    elapsed = time.time() - start_time
    logger.info(f"Inference completed in {elapsed/60:.2f} min.")
    
    return pd.DataFrame(preds_data), pd.DataFrame(metrics_data), pd.DataFrame(perf_data)

def main():
    global logger

    # --- Configuration Defaults ---
    DEFAULTS = {
        "file": "df_patna_covariates.csv",
        "city": "patna",
        "model": "amazon/chronos-bolt-small",
        "context_days": 4,
        "horizon_hours": 4,
        "output_dir": "outputs/part3"
    }

    parser = argparse.ArgumentParser(description="Part 3 Compute: Run Chronos Inference")
    parser.add_argument("--file", type=str, default=DEFAULTS["file"], help="Input CSV file")
    parser.add_argument("--city", type=str, default=DEFAULTS["city"], help="City name for output naming")
    parser.add_argument("--model", type=str, default=DEFAULTS["model"], help="Chronos model variant")
    parser.add_argument("--context_days", type=int, default=DEFAULTS["context_days"], help="Context length in days")
    parser.add_argument("--horizon_hours", type=int, default=DEFAULTS["horizon_hours"], help="Forecast horizon in hours")
    parser.add_argument("--output_dir", type=str, default=DEFAULTS["output_dir"], help="Directory to save CSV results")
    
    args = parser.parse_args()
    
    # --- Setup Logger ---
    os.makedirs("outputs", exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, "log")
    
    logger = setup_logger("part3_compute", "run_log", log_dir)
    
    if not os.path.exists(args.file):
        logger.error(f"Input file not found: {args.file}")
        return

    logger.info(f"Loading data from {args.file}")
    df = pd.read_csv(args.file, parse_dates=["From Date"])
    
    pipeline = load_pipeline(args.model)
    
    df_preds, df_metrics, df_perf = run_inference(
        pipeline, 
        df, 
        context_len_hours=args.context_days * 24, 
        horizon_hours=args.horizon_hours
    )
    
    preds_path = os.path.join(args.output_dir, f"{args.city}_predictions.csv")
    metrics_path = os.path.join(args.output_dir, f"{args.city}_metrics.csv")
    perf_path = os.path.join(args.output_dir, f"{args.city}_performance.csv")
    
    df_preds.to_csv(preds_path, index=False)
    df_metrics.to_csv(metrics_path, index=False)
    df_perf.to_csv(perf_path, index=False)
    
    logger.info(f"Saved predictions to {preds_path}")
    logger.info(f"Saved metrics to {metrics_path}")
    logger.info(f"Saved performance to {perf_path}")
    
    logger.info(f"Avg Latency: {df_perf['latency_sec'].mean():.4f}s | Avg Throughput: {df_perf['throughput_preds_per_sec'].mean():.2f} preds/sec")
    logger.info(f"Avg Process CPU: {df_perf['cpu_util_pct'].mean():.1f}% | Avg Process RAM: {df_perf['ram_util_mb'].mean():.1f} MB")

if __name__ == "__main__":
    main()