import pandas as pd, torch
from chronos import BaseChronosPipeline

# Load model
print("Loading model...")
pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cpu",
    torch_dtype=torch.bfloat16,
)

# Load PM2.5 series
print("Loading data...")
df = pd.read_csv("df_ggn_covariates.csv", parse_dates=["From Date"])
pm = torch.tensor(df["calibPM"].values, dtype=torch.float32)

# Forecast next 24 hours (24 future steps)
print("Forecasting...")
quantiles, mean = pipeline.predict_quantiles(
    context=pm,
    prediction_length=24,
    quantile_levels=[0.1, 0.5, 0.9],
)
print("Quantiles shape:", quantiles.shape)
print("Mean shape:", mean.shape)
print("Quantiles:", quantiles)
print("Mean:", mean)