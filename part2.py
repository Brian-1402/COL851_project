"""
part2.py — Performance and Resource Analysis for Chronos Forecasting (CPU, Windows Compatible)

Usage:
    python part2.py

Optional Args:
    --data_gurgaon   Path to Gurgaon CSV (default: df_ggn_covariates.csv)
    --data_patna     Path to Patna CSV (default: df_patna_covariates.csv)
    --loop           Number of repeated inference runs per configuration (default: 3)
    --prometheus_port  Port for Prometheus metrics endpoint (default: 8000)

Description:
    Runs zero-shot inference using pre-trained Amazon Chronos models on PM data for two cities.
    Measures and exports metrics:
        - Average and P95 latency
        - Throughput
        - CPU utilization
        - Memory utilization
    Exposes live Prometheus metrics endpoint for Grafana dashboards.
    Saves detailed CSV logs to ./outputs_perf/.
"""

import argparse
import os
import time
import psutil
import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline
from sklearn.metrics import mean_squared_error
from prometheus_client import Gauge, start_http_server

# ===============================================================
# Argument parsing
# ===============================================================
parser = argparse.ArgumentParser(description="Chronos performance measurement")
parser.add_argument("--data_gurgaon", default="df_ggn_covariates.csv")
parser.add_argument("--data_patna", default="df_patna_covariates.csv")
parser.add_argument("--loop", type=int, default=10)
parser.add_argument("--prometheus_port", type=int, default=8000)
args = parser.parse_args()

os.makedirs("outputs_perf", exist_ok=True)
csv_path = "outputs_perf/perf_metrics.csv"

# ===============================================================
# Prometheus metric registration (Option B — separate scrape job)
# ===============================================================
METRICS = {
    "latency_avg": Gauge("latency_avg_seconds", "Average inference latency", labelnames=["job"]),
    "latency_p95": Gauge("latency_p95_seconds", "95th percentile inference latency", labelnames=["job"]),
    "throughput": Gauge("throughput_samples_per_sec", "Inference throughput", labelnames=["job"]),
    "cpu_util": Gauge("cpu_util_percent", "CPU utilization percent", labelnames=["job"]),
    "mem_util": Gauge("memory_util_percent", "Memory utilization percent", labelnames=["job"]),
}
start_http_server(args.prometheus_port)

# ===============================================================
# Data loading
# ===============================================================
def load_city_data(file):
    df = pd.read_csv(file, parse_dates=["From Date"])
    return df["calibPM"].astype(float).to_numpy()

gurgaon_series = load_city_data(args.data_gurgaon)
patna_series = load_city_data(args.data_patna)

# ===============================================================
# Experiment configuration
# ===============================================================
model_variants = [
    "amazon/chronos-t5-mini",
    "amazon/chronos-t5-small",
    "amazon/chronos-bolt-small",
]

context_days = [2, 4, 8, 10, 14]
horizons = [4, 8, 12, 24, 48]
context_fixed = 10 * 24

# ===============================================================
# Utility functions
# ===============================================================
def record_system_metrics():
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory().percent
    return cpu, mem

def forecast(pipeline, context, horizon):
    _, mean = pipeline.predict_quantiles(
        torch.tensor(context, dtype=torch.float32),
        prediction_length=horizon,
        quantile_levels=[0.5]
    )
    return mean.squeeze().numpy()

def run_forecasting_perf(series, pipeline, context_len, horizon, loops, city, model):
    latencies = []
    rmses = []

    print(f"\nRunning: City={city}, Model={model}, Context={context_len}h, Horizon={horizon}h, Loops={loops}")
    for i in range(loops):
        context = series[-context_len:]
        true = series[-(context_len + horizon):-context_len]  # simulate next horizon
        start_t = time.time()
        pred = forecast(pipeline, context, horizon)
        latency = time.time() - start_t
        latencies.append(latency)
        rmse = np.sqrt(mean_squared_error(true[:len(pred)], pred))
        rmses.append(rmse)
        print(f"  Loop {i+1}/{loops}: latency={latency:.3f}s, rmse={rmse:.4f}")

    avg_lat = np.mean(latencies)
    p95_lat = np.percentile(latencies, 95)
    thr = loops / np.sum(latencies)
    cpu, mem = record_system_metrics()

    job_label = f"{city}_{model.replace('/', '_')}"
    METRICS["latency_avg"].labels(job=job_label).set(avg_lat)
    METRICS["latency_p95"].labels(job=job_label).set(p95_lat)
    METRICS["throughput"].labels(job=job_label).set(thr)
    METRICS["cpu_util"].labels(job=job_label).set(cpu)
    METRICS["mem_util"].labels(job=job_label).set(mem)

    print(f"Summary → Avg latency={avg_lat:.3f}s, P95={p95_lat:.3f}s, "
          f"Throughput={thr:.3f}/s, CPU={cpu:.1f}%, MEM={mem:.1f}%")

    return {
        "city": city,
        "model": model,
        "context_hours": context_len,
        "horizon_hours": horizon,
        "loops": loops,
        "latency_avg": avg_lat,
        "latency_p95": p95_lat,
        "throughput": thr,
        "cpu_util": cpu,
        "mem_util": mem,
        "rmse_avg": np.mean(rmses),
    }

# ===============================================================
# Main experiment loop
# ===============================================================
results = []
for model_variant in model_variants:
    print(f"\n=== Loading model: {model_variant} ===")
    pipeline = BaseChronosPipeline.from_pretrained(model_variant, device_map="cpu", torch_dtype=torch.float32)

    for city_name, series in [("gurgaon", gurgaon_series), ("patna", patna_series)]:
        # --- RMSE vs Context Length (Fixed 24h Horizon)
        for days in context_days:
            context_len = days * 24
            results.append(run_forecasting_perf(series, pipeline, context_len, 24, args.loop, city_name, model_variant))

        # --- RMSE vs Horizon (Fixed 10-day Context)
        for h in horizons:
            results.append(run_forecasting_perf(series, pipeline, context_fixed, h, args.loop, city_name, model_variant))

# ===============================================================
# Save Results
# ===============================================================
df = pd.DataFrame(results)
df.to_csv(csv_path, index=False)
print(f"\nAll performance metrics saved to: {csv_path}")

# ===============================================================
# Step: Plot Performance Metrics
# ===============================================================
import matplotlib.pyplot as plt

def plot_perf_metrics(df):
    metrics = ["latency_avg", "latency_p95", "throughput", "cpu_util", "mem_util"]
    os.makedirs("outputs_perf", exist_ok=True)

    for metric in metrics:
        plt.figure(figsize=(8, 5))
        for city in df["city"].unique():
            subset = df[df["city"] == city]
            plt.scatter(subset["context_hours"], subset[metric], label=f"{city}-context", marker='o')
            plt.scatter(subset["horizon_hours"], subset[metric], label=f"{city}-horizon", marker='x')
        plt.xlabel("Context/Horizon (hours)")
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(f"{metric.replace('_', ' ').title()} vs Context/Horizon")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        path = f"outputs_perf/{metric}_plot.png"
        plt.savefig(path)
        plt.close()
        print(f"Saved: {path}")

plot_perf_metrics(df)
print("Performance plots saved under outputs_perf/")