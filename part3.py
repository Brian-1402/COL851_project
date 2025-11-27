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
log_file = "outputs/part3_log.txt"
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


def load_city_data(file):
    df = pd.read_csv(file, parse_dates=["From Date"])
    series = df["calibPM"].astype(float)
    return series

gurgaon_series = load_city_data("df_ggn_covariates.csv")
patna_series = load_city_data("df_patna_covariates.csv")

model_variants = [
    "amazon/chronos-t5-mini",
    "amazon/chronos-t5-small",
    "amazon/chronos-bolt-small"
]

# best params found from part2
model = "amazon/chronos-t5-small"
ctx_days = 4
horizon_hours = 4



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
    test_hours = []
    for i in tqdm(range(context_len_hours, total - horizon_hours, step)):
        context = series[i - context_len_hours:i]
        true = series[i:i + horizon_hours]
        pred = forecast(pipeline, context, horizon_hours)
        rmses.append(np.sqrt(mean_squared_error(true, pred)))
        test_hours.append(i)
    elapsed = time.time() - start_time
    log_print(f"compute_rmse(context={context_len_hours}h, horizon={horizon_hours}h) took {elapsed/60:.2f} min")
    return rmses, test_hours

def main():
    pipeline = load_pipeline(model)
    rmses, test_hours = compute_rmse(
        pipeline,
        patna_series,
        context_len_hours=ctx_days * 24,
        horizon_hours=horizon_hours
    )
    test_days = [h // 24 for h in test_hours]
    print(test_days)
    print(rmses)

if __name__ == "__main__":
    main()
