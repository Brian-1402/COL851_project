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

# ===============================================================
# Step 0: Setup Output Directory & Logging
# ===============================================================
os.makedirs("outputs", exist_ok=True)
log_file = "outputs/experiment_log.txt"
summary_file = "outputs/summary_log.txt"

# Ensure both files exist
for f in [log_file, summary_file]:
    if not os.path.exists(f):
        open(f, "w").close()

# Custom print that logs to file + console
def log_print(msg):
    print(msg)
    with open(log_file, "a", encoding="utf-8") as f:   
        f.write(msg + "\n")

def write_summary(msg):
    with open(summary_file, "a", encoding="utf-8") as f:  
        f.write(msg + "\n")


# ===============================================================
# Step 1: Load Data
# ===============================================================
def load_city_data(file):
    df = pd.read_csv(file, parse_dates=["From Date"])
    series = df["calibPM"].astype(float)
    return series

gurgaon_series = load_city_data("df_ggn_covariates.csv")
patna_series = load_city_data("df_patna_covariates.csv")

# ===============================================================
# Step 2: Define Models
# ===============================================================
model_variants = [
    "amazon/chronos-t5-mini",
    "amazon/chronos-t5-small",
    "amazon/chronos-bolt-small"
]

# ===============================================================
# Step 3: Forecasting
# ===============================================================
def load_pipeline(model_variant):
    log_print(f"\nLoading {model_variant}...")
    return BaseChronosPipeline.from_pretrained(
        model_variant,
        device_map="cpu",
        torch_dtype=torch.float32
    )

def forecast(pipeline, context, horizon):
    context = torch.tensor(context, dtype=torch.float32)
    _, mean = pipeline.predict_quantiles(context, prediction_length=horizon, quantile_levels=[0.5])
    return mean.squeeze().numpy()

def compute_rmse(pipeline, series, context_len_hours, horizon_hours):
    start_time = time.time()
    series = series.to_numpy()
    total = len(series)
    step = horizon_hours
    rmses = []
    for i in tqdm(range(context_len_hours, total - horizon_hours, step)):
        context = series[i - context_len_hours:i]
        true = series[i:i + horizon_hours]
        pred = forecast(pipeline, context, horizon_hours)
        rmses.append(np.sqrt(mean_squared_error(true, pred)))
    elapsed = time.time() - start_time
    log_print(f"compute_rmse(context={context_len_hours}h, horizon={horizon_hours}h) took {elapsed/60:.2f} min")
    return np.mean(rmses)

# ===============================================================
# Step 4: Run Experiments
# ===============================================================
results = {"gurgaon": {}, "patna": {}}
context_days = [2, 4, 8, 10, 14]
horizons = [4, 8, 12, 24, 48]
context_fixed = 10 * 24

for model_variant in model_variants:
    pipeline = load_pipeline(model_variant)
    start_time = time.time()

    # --- RMSE vs Context ---
    log_print(f"\n{model_variant}: Context Length Experiments")
    rmses_g = []
    rmses_p = []
    for days in context_days:
        ctx_hours = days * 24
        log_print(f"→ Gurgaon | Context={days} days | Horizon=24h")
        rmses_g.append(compute_rmse(pipeline, gurgaon_series, ctx_hours, 24))
        log_print(f"→ Patna   | Context={days} days | Horizon=24h")
        rmses_p.append(compute_rmse(pipeline, patna_series, ctx_hours, 24))

    # --- RMSE vs Horizon ---
    log_print(f"\n{model_variant}: Forecast Horizon Experiments")
    rmses_h_g = []
    rmses_h_p = []
    for h in horizons:
        log_print(f"→ Gurgaon | Context=10 days | Horizon={h}h")
        rmses_h_g.append(compute_rmse(pipeline, gurgaon_series, context_fixed, h))
        log_print(f"→ Patna   | Context=10 days | Horizon={h}h")
        rmses_h_p.append(compute_rmse(pipeline, patna_series, context_fixed, h))

    results["gurgaon"][model_variant] = {
        "context_days": context_days,
        "rmse_context": rmses_g,
        "horizons": horizons,
        "rmse_horizon": rmses_h_g
    }
    results["patna"][model_variant] = {
        "context_days": context_days,
        "rmse_context": rmses_p,
        "horizons": horizons,
        "rmse_horizon": rmses_h_p
    }

    total_elapsed = (time.time() - start_time) / 60
    log_print(f"\n{model_variant} completed in {total_elapsed:.1f} min\n")

# ===============================================================
# Step 5: Plot and Summarize
# ===============================================================
def plot_rmse(city, city_results):
    for model, vals in city_results.items():
        safe_name = model.replace("/", "_")
        context_plot_path = f"outputs/{city}_context_{safe_name}.png"
        horizon_plot_path = f"outputs/{city}_horizon_{safe_name}.png"

        plt.figure(figsize=(8,5))
        plt.plot(vals["context_days"], vals["rmse_context"], marker='o')
        plt.xlabel("Context length (days)")
        plt.ylabel("RMSE (24h forecast)")
        plt.title(f"{city.title()} - Context vs RMSE ({model})")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(context_plot_path)
        plt.close()

        plt.plot(vals["horizons"], vals["rmse_horizon"], marker='o')
        plt.xlabel("Forecast horizon (hours)")
        plt.ylabel("RMSE (10-day context)")
        plt.title(f"{city.title()} - Horizon vs RMSE ({model})")
        plt.xticks(vals["horizons"], [f"{h}h" for h in vals["horizons"]]) 
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(horizon_plot_path)
        plt.close()

        log_print(f"Saved plots: {context_plot_path}, {horizon_plot_path}")

def summarize(city, city_results):
    summary = []
    for model, vals in city_results.items():
        best_ctx_idx = np.argmin(vals["rmse_context"])
        best_h_idx = np.argmin(vals["rmse_horizon"])
        summary.append({
            "Model": model,
            "Best Context (days)": vals["context_days"][best_ctx_idx],
            "Min RMSE (context)": vals["rmse_context"][best_ctx_idx],
            "Best Horizon (hours)": vals["horizons"][best_h_idx],
            "Min RMSE (horizon)": vals["rmse_horizon"][best_h_idx]
        })
    df = pd.DataFrame(summary)
    csv_path = f"outputs/{city}_summary.csv"
    df.to_csv(csv_path, index=False)
    log_print(f"\n{city.title()} Summary saved at {csv_path}:\n{df}")
    write_summary(f"{city.title()} Summary:\n{df.to_string(index=False)}\n")
    return df

# Generate plots and summaries
for city in results:
    plot_rmse(city, results[city])
    summarize(city, results[city])
